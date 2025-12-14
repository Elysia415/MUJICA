from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

import lancedb
import pandas as pd
import pyarrow as pa

from src.data_engine.chunker import chunk_text
from src.utils.llm import get_embedding, get_embeddings


class KnowledgeBase:
    """
    MUJICA 本地知识库：
    - LanceDB：存 chunk/paper 的向量索引（语义检索）
    - SQLite：存结构化元数据（评分/作者/决策/评审等）
    """

    def __init__(
        self,
        db_path: str = "data/lancedb",
        *,
        metadata_path: Optional[str] = None,
        papers_table: str = "papers",
        chunks_table: str = "chunks",
        embedding_model: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        chunk_max_tokens: int = 350,
        chunk_overlap_tokens: int = 60,
    ):
        self.db_path = db_path
        self.db: Optional[lancedb.db.LanceDBConnection] = None

        self.papers_table = papers_table
        self.chunks_table = chunks_table
        self.embedding_model = (
            (embedding_model or os.getenv("MUJICA_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
        )

        # Embedding 允许与 Chat 完全解耦（不同 key / 不同 base_url）
        env_embed_key = (os.getenv("MUJICA_EMBEDDING_API_KEY") or "").strip() or None
        env_embed_base = (os.getenv("MUJICA_EMBEDDING_BASE_URL") or "").strip() or None
        self.embedding_api_key = (embedding_api_key or "").strip() or env_embed_key
        self.embedding_base_url = (embedding_base_url or "").strip() or env_embed_base
        self.chunk_max_tokens = chunk_max_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

        # 默认将元数据库放在 lancedb 目录内，便于测试与清理
        self.metadata_path = metadata_path or os.path.join(self.db_path, "metadata.sqlite")
        self._meta_conn: Optional[sqlite3.Connection] = None
        self.metadata_df = pd.DataFrame()

    # ---------------------------
    # Init
    # ---------------------------
    def initialize_db(self) -> None:
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        print(f"Connected to LanceDB at {self.db_path}")

        # SQLite
        self._meta_conn = sqlite3.connect(self.metadata_path)
        self._meta_conn.row_factory = sqlite3.Row
        self._init_metadata_schema()
        self.metadata_df = self._load_metadata_df()

    def _init_metadata_schema(self) -> None:
        assert self._meta_conn is not None
        cur = self._meta_conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                tldr TEXT,
                authors_json TEXT,
                keywords_json TEXT,
                year INTEGER,
                venue_id TEXT,
                forum TEXT,
                number INTEGER,
                pdf_url TEXT,
                pdf_path TEXT,
                decision TEXT,
                presentation TEXT,
                rating REAL,
                raw_json TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                paper_id TEXT,
                idx INTEGER,
                rating REAL,
                rating_raw TEXT,
                confidence REAL,
                confidence_raw TEXT,
                summary TEXT,
                strengths TEXT,
                weaknesses TEXT,
                raw_json TEXT,
                PRIMARY KEY (paper_id, idx)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_paper_id ON reviews(paper_id)")
        self._meta_conn.commit()

        # schema migrate：为旧库补齐新列
        try:
            cols = [r["name"] for r in cur.execute("PRAGMA table_info(papers)").fetchall()]
            if "presentation" not in cols:
                cur.execute("ALTER TABLE papers ADD COLUMN presentation TEXT")
                self._meta_conn.commit()
        except Exception:
            # 忽略迁移失败（最坏情况：不会存 presentation）
            pass

    # ---------------------------
    # Ingest
    # ---------------------------
    def ingest_data(self, papers: List[Dict[str, Any]]) -> None:
        """
        Ingest 论文到知识库。

        兼容旧字段：
        - `rating`：论文评分（float 或 None）
        - `content`：全文/解析文本（可选）
        - `reviews`：评审列表（可选）
        """
        if not papers:
            print("No papers to ingest.")
            return
        if self.db is None or self._meta_conn is None:
            raise RuntimeError("KnowledgeBase not initialized. Call initialize_db() first.")

        # 1) 写入结构化元数据（SQLite）
        for p in papers:
            self._upsert_paper_and_reviews(p)

        # Embedding 批大小（部分 OpenAI-compatible 网关对单次请求 input 数量有限制，常见为 64）
        batch_size = int(os.getenv("MUJICA_EMBEDDING_BATCH_SIZE", "64") or 64)
        batch_size = max(1, min(batch_size, 256))

        # Embedding 日志频率：每 N 个 batch 打印一次进度（默认 10；设为 1 会非常详细）
        try:
            log_every = int(os.getenv("MUJICA_EMBEDDING_LOG_EVERY", "10") or 10)
        except Exception:
            log_every = 10
        log_every = max(0, min(log_every, 10_000))

        # 2) 写入 paper-level 向量（便于 fallback / 简单检索）
        paper_payload: List[Dict[str, Any]] = []
        paper_texts: List[str] = []
        for p in papers:
            pid = str(p.get("id") or "").strip()
            if not pid:
                continue
            title = (p.get("title") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            text_to_embed = f"Title: {title}\nAbstract: {abstract}".strip()
            if not text_to_embed:
                continue
            paper_payload.append(
                {
                    "id": pid,
                    "title": title,
                    "abstract": abstract,
                    "rating": p.get("rating"),
                    "year": p.get("year"),
                }
            )
            paper_texts.append(text_to_embed)

        paper_rows: List[Dict[str, Any]] = []
        if paper_texts:
            total = len(paper_texts)
            total_batches = int(math.ceil(total / batch_size)) if batch_size > 0 else 0
            print(
                f"[Embedding] papers: total={total} batch_size={batch_size} batches={total_batches} model={self.embedding_model}"
            )
            t0 = time.time()
            ok_total = 0
            empty_total = 0
            paper_vecs: List[list] = []
            for bi, start in enumerate(range(0, total, batch_size)):
                batch = paper_texts[start : start + batch_size]
                t1 = time.time()
                vecs = get_embeddings(
                    batch,
                    model=self.embedding_model,
                    api_key=self.embedding_api_key,
                    base_url=self.embedding_base_url,
                    tag=f"papers {bi+1}/{total_batches}",
                )
                dt = time.time() - t1
                ok = sum(1 for v in vecs if v)
                empty = len(vecs) - ok
                ok_total += ok
                empty_total += empty
                paper_vecs.extend(vecs)

                if log_every > 0 and (
                    bi == 0 or (bi + 1) % log_every == 0 or (bi + 1) == total_batches
                ):
                    done = min(start + len(batch), total)
                    pct = int(done * 100 / total) if total else 100
                    print(
                        f"[Embedding] papers: {done}/{total} ({pct}%) "
                        f"batch {bi+1}/{total_batches} size={len(batch)} ok={ok} empty={empty} dt={dt:.1f}s"
                    )

            print(
                f"[Embedding] papers done: ok={ok_total}/{total} empty={empty_total} elapsed={time.time()-t0:.1f}s"
            )
            for meta, vec in zip(paper_payload, paper_vecs):
                if not vec:
                    continue
                row = dict(meta)
                row["vector"] = vec
                paper_rows.append(row)

        if paper_rows:
            if self.papers_table in self.db.table_names():
                tbl = self.db.open_table(self.papers_table)

                # 如果历史表 schema 推断错误（例如 year 全是 None 被推断成 null），需要迁移/重建一次
                # 否则后续插入 int/float 会触发 Arrow cast 错误：Unsupported cast from int64 to null
                try:
                    year_type = str(tbl.schema.field("year").type) if "year" in tbl.schema.names else ""
                except Exception:
                    year_type = ""
                expected_dim = len(paper_rows[0].get("vector") or [])
                try:
                    existing_dim = int(tbl.schema.field("vector").type.list_size) if "vector" in tbl.schema.names else -1
                except Exception:
                    existing_dim = -1

                needs_rebuild = (year_type == "null") or (existing_dim > 0 and expected_dim > 0 and existing_dim != expected_dim)

                if needs_rebuild:
                    ids_set = set([r["id"] for r in paper_rows])
                    try:
                        old_rows = tbl.to_list()
                    except Exception:
                        old_rows = []

                    merged: List[Dict[str, Any]] = []
                    for r in old_rows:
                        pid = r.get("id")
                        if pid in ids_set:
                            continue
                        vec = r.get("vector") or []
                        if not isinstance(vec, list) or (expected_dim and len(vec) != expected_dim):
                            continue
                        merged.append(
                            {
                                "id": str(pid),
                                "title": r.get("title") or "",
                                "abstract": r.get("abstract") or "",
                                "rating": r.get("rating", None),
                                "year": None,  # 旧表 year 为空，迁移时保持 None（可从 SQLite 再补）
                                "vector": vec,
                            }
                        )

                    # 新 rows
                    for r in paper_rows:
                        vec = r.get("vector") or []
                        if not isinstance(vec, list) or (expected_dim and len(vec) != expected_dim):
                            continue
                        merged.append(r)

                    schema = pa.schema(
                        [
                            pa.field("id", pa.string()),
                            pa.field("title", pa.string()),
                            pa.field("abstract", pa.string()),
                            pa.field("rating", pa.float64()),
                            pa.field("year", pa.int64()),
                            pa.field("vector", pa.list_(pa.float32(), expected_dim)),
                        ]
                    )

                    # drop + recreate
                    try:
                        self.db.drop_table(self.papers_table, ignore_missing=True)
                    except Exception:
                        pass
                    self.db.create_table(self.papers_table, data=merged, schema=schema)
                else:
                    # 删除旧记录再插入，避免重复
                    ids = [r["id"] for r in paper_rows]
                    try:
                        ids_sql = ", ".join([f"'{i}'" for i in ids])
                        tbl.delete(f"id IN ({ids_sql})")
                    except Exception:
                        pass
                    tbl.add(paper_rows)
            else:
                # 显式 schema：避免第一批数据某列全是 None 被推断成 null，导致后续写入失败
                expected_dim = len(paper_rows[0].get("vector") or [])
                schema = pa.schema(
                    [
                        pa.field("id", pa.string()),
                        pa.field("title", pa.string()),
                        pa.field("abstract", pa.string()),
                        pa.field("rating", pa.float64()),
                        pa.field("year", pa.int64()),
                        pa.field("vector", pa.list_(pa.float32(), expected_dim)),
                    ]
                )
                self.db.create_table(self.papers_table, data=paper_rows, schema=schema)

        # 3) 写入 chunk-level 向量（用于证据溯源）
        chunk_rows = []
        for p in papers:
            pid = str(p.get("id") or "").strip()
            if not pid:
                continue

            sources: List[tuple[str, str]] = []
            title = (p.get("title") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            tldr = (p.get("tldr") or "").strip()
            content = (p.get("content") or "").strip()

            # 删除该 paper 旧 chunks（避免重复）
            # 关键：当本次没有提供 content（未解析全文）时，保留历史 full_text chunks，避免“补元数据”时把全文证据删掉。
            if self.chunks_table in self.db.table_names():
                try:
                    tbl = self.db.open_table(self.chunks_table)
                    safe_pid = pid.replace("'", "''")
                    if content:
                        tbl.delete(f"paper_id = '{safe_pid}'")
                    else:
                        tbl.delete(f"paper_id = '{safe_pid}' AND source != 'full_text'")
                except Exception:
                    pass

            # meta：将作者/关键词/评分等元数据也做成可检索、可引用的 chunk
            # 这样 Writer 可以用 chunk_id 来引用“作者/年份/评分/决策/领域(关键词)”等信息
            authors = p.get("authors") or []
            if not isinstance(authors, list):
                authors = []
            keywords = p.get("keywords") or []
            if not isinstance(keywords, list):
                keywords = []

            meta_lines: List[str] = [f"Paper ID: {pid}"]
            if title:
                meta_lines.append(f"Title: {title}")
            if authors:
                meta_lines.append(f"Authors: {', '.join([str(a) for a in authors][:25])}")
            if keywords:
                meta_lines.append(f"Keywords: {', '.join([str(k) for k in keywords][:30])}")
            if p.get("year") is not None:
                meta_lines.append(f"Year: {p.get('year')}")
            if p.get("venue_id"):
                meta_lines.append(f"Venue: {p.get('venue_id')}")
            if p.get("decision") is not None:
                meta_lines.append(f"Decision: {p.get('decision')}")
            if p.get("presentation") is not None:
                meta_lines.append(f"Presentation: {p.get('presentation')}")
            if p.get("rating") is not None:
                meta_lines.append(f"Rating: {p.get('rating')}")
            if p.get("pdf_url"):
                meta_lines.append(f"PDF: {p.get('pdf_url')}")
            if p.get("pdf_path"):
                meta_lines.append(f"PDF Path: {p.get('pdf_path')}")
            reviews = p.get("reviews") or []
            if isinstance(reviews, list) and reviews:
                meta_lines.append(f"Reviews: {len(reviews)}")

            meta_text = "\n".join([x for x in meta_lines if x]).strip()
            if meta_text:
                sources.append(("meta", meta_text))

            if title or abstract:
                sources.append(("title_abstract", f"Title: {title}\nAbstract: {abstract}".strip()))
            if tldr:
                sources.append(("tldr", tldr))
            if content:
                sources.append(("full_text", content))

            for source_name, text in sources:
                chunks = chunk_text(
                    text,
                    max_tokens=self.chunk_max_tokens,
                    overlap_tokens=self.chunk_overlap_tokens,
                )
                for i, c in enumerate(chunks):
                    chunk_rows.append(
                        {
                            "chunk_id": f"{pid}::{source_name}::{i}",
                            "paper_id": pid,
                            "source": source_name,
                            "chunk_index": i,
                            "text": c,
                        }
                    )

        if chunk_rows:
            texts = [r["text"] for r in chunk_rows]

            total = len(texts)
            total_batches = int(math.ceil(total / batch_size)) if batch_size > 0 else 0
            print(
                f"[Embedding] chunks: total={total} batch_size={batch_size} batches={total_batches} model={self.embedding_model}"
            )
            t0 = time.time()
            ok_total = 0
            empty_total = 0
            rows_to_insert = []
            for bi, start in enumerate(range(0, total, batch_size)):
                batch_texts = texts[start : start + batch_size]
                t1 = time.time()
                batch_vecs = get_embeddings(
                    batch_texts,
                    model=self.embedding_model,
                    api_key=self.embedding_api_key,
                    base_url=self.embedding_base_url,
                    tag=f"chunks {bi+1}/{total_batches}",
                )
                dt = time.time() - t1
                ok = sum(1 for v in batch_vecs if v)
                empty = len(batch_vecs) - ok
                ok_total += ok
                empty_total += empty
                for r, vec in zip(chunk_rows[start : start + batch_size], batch_vecs):
                    if not vec:
                        continue
                    rr = dict(r)
                    rr["vector"] = vec
                    rows_to_insert.append(rr)

                if log_every > 0 and (
                    bi == 0 or (bi + 1) % log_every == 0 or (bi + 1) == total_batches
                ):
                    done = min(start + len(batch_texts), total)
                    pct = int(done * 100 / total) if total else 100
                    rate = done / max(1e-9, (time.time() - t0))
                    print(
                        f"[Embedding] chunks: {done}/{total} ({pct}%) "
                        f"batch {bi+1}/{total_batches} size={len(batch_texts)} ok={ok} empty={empty} "
                        f"dt={dt:.1f}s rate={rate:.1f}/s"
                    )

            print(
                f"[Embedding] chunks done: ok={ok_total}/{total} empty={empty_total} elapsed={time.time()-t0:.1f}s"
            )

            if rows_to_insert:
                if self.chunks_table in self.db.table_names():
                    self.db.open_table(self.chunks_table).add(rows_to_insert)
                else:
                    self.db.create_table(self.chunks_table, data=rows_to_insert)

        # 4) 刷新 metadata_df
        self.metadata_df = self._load_metadata_df()
        print(f"✓ Ingested {len(papers)} papers (metadata) into SQLite and vectors into LanceDB.")

    def _upsert_paper_and_reviews(self, p: Dict[str, Any]) -> None:
        assert self._meta_conn is not None

        pid = str(p.get("id") or "").strip()
        if not pid:
            return

        title = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        tldr = (p.get("tldr") or "").strip()
        authors_json = json.dumps(p.get("authors") or [], ensure_ascii=False)
        keywords_json = json.dumps(p.get("keywords") or [], ensure_ascii=False)

        row = (
            pid,
            title,
            abstract,
            tldr,
            authors_json,
            keywords_json,
            p.get("year"),
            p.get("venue_id"),
            p.get("forum"),
            p.get("number"),
            p.get("pdf_url"),
            p.get("pdf_path"),
            p.get("decision"),
            p.get("presentation"),
            p.get("rating"),
            json.dumps(p, ensure_ascii=False),
        )

        cur = self._meta_conn.cursor()
        cur.execute(
            """
            INSERT INTO papers (
                id, title, abstract, tldr, authors_json, keywords_json,
                year, venue_id, forum, number, pdf_url, pdf_path,
                decision, presentation, rating, raw_json, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, CURRENT_TIMESTAMP
            )
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                abstract=excluded.abstract,
                tldr=excluded.tldr,
                authors_json=excluded.authors_json,
                keywords_json=excluded.keywords_json,
                year=excluded.year,
                venue_id=excluded.venue_id,
                forum=excluded.forum,
                number=excluded.number,
                pdf_url=excluded.pdf_url,
                pdf_path=excluded.pdf_path,
                decision=excluded.decision,
                presentation=excluded.presentation,
                rating=excluded.rating,
                raw_json=excluded.raw_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            row,
        )

        # reviews：简单策略，先删后插
        cur.execute("DELETE FROM reviews WHERE paper_id = ?", (pid,))
        reviews = p.get("reviews") or []
        for idx, r in enumerate(reviews):
            cur.execute(
                """
                INSERT INTO reviews (
                    paper_id, idx, rating, rating_raw, confidence, confidence_raw,
                    summary, strengths, weaknesses, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pid,
                    idx,
                    r.get("rating"),
                    r.get("rating_raw"),
                    r.get("confidence"),
                    r.get("confidence_raw"),
                    r.get("summary"),
                    r.get("strengths"),
                    r.get("weaknesses"),
                    json.dumps(r, ensure_ascii=False),
                ),
            )

        self._meta_conn.commit()

    # ---------------------------
    # Query
    # ---------------------------
    def _load_metadata_df(self) -> pd.DataFrame:
        if self._meta_conn is None:
            return pd.DataFrame()
        try:
            df = pd.read_sql_query("SELECT * FROM papers", self._meta_conn)

            # 解析 authors/keywords，便于结构化过滤与统计
            def _parse_list(v: Any) -> list:
                if v is None:
                    return []
                if isinstance(v, list):
                    return v
                if not isinstance(v, str):
                    return []
                s = v.strip()
                if not s:
                    return []
                try:
                    obj = json.loads(s)
                except Exception:
                    return []
                return obj if isinstance(obj, list) else []

            if "authors_json" in df.columns and "authors" not in df.columns:
                df["authors"] = df["authors_json"].apply(_parse_list)
                df["authors_text"] = df["authors"].apply(
                    lambda xs: ", ".join([str(x) for x in xs]) if isinstance(xs, list) else ""
                )

            if "keywords_json" in df.columns and "keywords" not in df.columns:
                df["keywords"] = df["keywords_json"].apply(_parse_list)
                df["keywords_text"] = df["keywords"].apply(
                    lambda xs: ", ".join([str(x) for x in xs]) if isinstance(xs, list) else ""
                )

            return df
        except Exception:
            return pd.DataFrame()

    def search_structured(self, query: str = None) -> pd.DataFrame:
        """
        返回结构化元数据 DataFrame（外部可用 pandas 自由过滤）。
        query 参数暂未启用（保留兼容）。
        """
        if self.metadata_df.empty:
            self.metadata_df = self._load_metadata_df()
        return self.metadata_df

    def _get_papers_by_ids(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not ids or self._meta_conn is None:
            return {}
        placeholders = ", ".join(["?"] * len(ids))
        rows = self._meta_conn.execute(f"SELECT * FROM papers WHERE id IN ({placeholders})", ids).fetchall()
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            d = dict(r)
            out[d["id"]] = d
        return out

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        if not paper_id or self._meta_conn is None:
            return None
        row = self._meta_conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return dict(row) if row else None

    def get_reviews(self, paper_id: str) -> List[Dict[str, Any]]:
        if not paper_id or self._meta_conn is None:
            return []
        rows = self._meta_conn.execute(
            "SELECT * FROM reviews WHERE paper_id = ? ORDER BY idx ASC",
            (paper_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_paper(self, paper_id: str, *, delete_pdf: bool = False) -> Dict[str, Any]:
        """
        删除单篇论文条目（不可逆）：
        - SQLite：papers + reviews
        - LanceDB：papers 向量表 + chunks 向量表
        - 可选：删除本地 PDF 文件（若 papers.pdf_path 存在）
        """
        pid = str(paper_id or "").strip()
        if not pid:
            return {"ok": False, "error": "paper_id 为空"}
        if self._meta_conn is None:
            raise RuntimeError("KnowledgeBase not initialized. Call initialize_db() first.")

        # 先取出 pdf_path（删除后就取不到了）
        pdf_path: Optional[str] = None
        try:
            row = self._meta_conn.execute("SELECT pdf_path FROM papers WHERE id = ?", (pid,)).fetchone()
            if row:
                try:
                    pdf_path = row["pdf_path"]
                except Exception:
                    pdf_path = row[0] if len(row) > 0 else None
        except Exception:
            pdf_path = None

        deleted_sql_papers = 0
        deleted_sql_reviews = 0

        # SQLite 删除（事务）
        try:
            cur = self._meta_conn.cursor()
            cur.execute("DELETE FROM reviews WHERE paper_id = ?", (pid,))
            deleted_sql_reviews = int(cur.rowcount or 0)
            cur.execute("DELETE FROM papers WHERE id = ?", (pid,))
            deleted_sql_papers = int(cur.rowcount or 0)
            self._meta_conn.commit()
        except Exception as e:
            try:
                self._meta_conn.rollback()
            except Exception:
                pass
            return {"ok": False, "error": f"SQLite 删除失败：{e}"}

        # LanceDB 删除（尽量删除，失败不影响 SQLite 成功）
        deleted_vec_papers = 0
        deleted_vec_chunks = 0
        if self.db is not None:
            safe_pid = pid.replace("'", "''")
            try:
                if self.papers_table in self.db.table_names():
                    tbl = self.db.open_table(self.papers_table)
                    # LanceDB delete 无 rowcount 返回，做 best-effort
                    tbl.delete(f"id = '{safe_pid}'")
                    deleted_vec_papers = 1
            except Exception:
                pass
            try:
                if self.chunks_table in self.db.table_names():
                    tbl = self.db.open_table(self.chunks_table)
                    tbl.delete(f"paper_id = '{safe_pid}'")
                    deleted_vec_chunks = 1
            except Exception:
                pass

        deleted_pdf = False
        pdf_error: Optional[str] = None
        if delete_pdf and pdf_path:
            try:
                p = str(pdf_path)
                if p and os.path.exists(p):
                    os.remove(p)
                    deleted_pdf = True
            except Exception as e:
                pdf_error = str(e)

        # 刷新 metadata_df（UI/过滤依赖）
        try:
            self.metadata_df = self._load_metadata_df()
        except Exception:
            pass

        return {
            "ok": True,
            "paper_id": pid,
            "deleted_sql_papers": deleted_sql_papers,
            "deleted_sql_reviews": deleted_sql_reviews,
            "deleted_vec_papers": deleted_vec_papers,
            "deleted_vec_chunks": deleted_vec_chunks,
            "pdf_path": pdf_path,
            "deleted_pdf": deleted_pdf,
            "pdf_error": pdf_error,
        }

    def delete_papers(self, paper_ids: List[str], *, delete_pdf: bool = False) -> Dict[str, Any]:
        """
        批量删除论文条目（不可逆）。
        返回统计信息（best-effort；LanceDB delete 不保证返回 rowcount）。
        """
        if self._meta_conn is None:
            raise RuntimeError("KnowledgeBase not initialized. Call initialize_db() first.")
        ids = [str(x).strip() for x in (paper_ids or []) if str(x).strip()]
        # 去重且保持顺序
        seen = set()
        uniq: List[str] = []
        for x in ids:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        if not uniq:
            return {"ok": False, "error": "paper_ids 为空"}

        placeholders = ", ".join(["?"] * len(uniq))

        # 先取出 pdf_path（可选删除）
        pdf_paths: Dict[str, str] = {}
        if delete_pdf:
            try:
                rows = self._meta_conn.execute(
                    f"SELECT id, pdf_path FROM papers WHERE id IN ({placeholders})",
                    uniq,
                ).fetchall()
                for r in rows:
                    try:
                        pid = r["id"]
                        pth = r["pdf_path"]
                    except Exception:
                        pid = r[0]
                        pth = r[1] if len(r) > 1 else None
                    if pid and pth:
                        pdf_paths[str(pid)] = str(pth)
            except Exception:
                pdf_paths = {}

        deleted_sql_papers = 0
        deleted_sql_reviews = 0

        # SQLite 删除（事务）
        try:
            cur = self._meta_conn.cursor()
            cur.execute(f"DELETE FROM reviews WHERE paper_id IN ({placeholders})", uniq)
            deleted_sql_reviews = int(cur.rowcount or 0)
            cur.execute(f"DELETE FROM papers WHERE id IN ({placeholders})", uniq)
            deleted_sql_papers = int(cur.rowcount or 0)
            self._meta_conn.commit()
        except Exception as e:
            try:
                self._meta_conn.rollback()
            except Exception:
                pass
            return {"ok": False, "error": f"SQLite 批量删除失败：{e}"}

        # LanceDB 删除（best-effort）
        deleted_vec_papers = 0
        deleted_vec_chunks = 0
        if self.db is not None:
            safe_ids = [x.replace("'", "''") for x in uniq]

            def _chunks(xs: List[str], n: int = 200):
                for i in range(0, len(xs), n):
                    yield xs[i : i + n]

            try:
                if self.papers_table in self.db.table_names():
                    tbl = self.db.open_table(self.papers_table)
                    for part in _chunks(safe_ids, 200):
                        ids_sql = ", ".join([f"'{x}'" for x in part])
                        tbl.delete(f"id IN ({ids_sql})")
                    deleted_vec_papers = 1
            except Exception:
                pass

            try:
                if self.chunks_table in self.db.table_names():
                    tbl = self.db.open_table(self.chunks_table)
                    for part in _chunks(safe_ids, 200):
                        ids_sql = ", ".join([f"'{x}'" for x in part])
                        tbl.delete(f"paper_id IN ({ids_sql})")
                    deleted_vec_chunks = 1
            except Exception:
                pass

        deleted_pdf = 0
        pdf_errors: List[Dict[str, Any]] = []
        if delete_pdf and pdf_paths:
            for pid, pth in pdf_paths.items():
                try:
                    if pth and os.path.exists(pth):
                        os.remove(pth)
                        deleted_pdf += 1
                except Exception as e:
                    pdf_errors.append({"paper_id": pid, "pdf_path": pth, "error": str(e)})

        # 刷新 metadata_df
        try:
            self.metadata_df = self._load_metadata_df()
        except Exception:
            pass

        return {
            "ok": True,
            "deleted_sql_papers": deleted_sql_papers,
            "deleted_sql_reviews": deleted_sql_reviews,
            "deleted_vec_papers": deleted_vec_papers,
            "deleted_vec_chunks": deleted_vec_chunks,
            "requested": len(uniq),
            "deleted_pdf": deleted_pdf,
            "pdf_errors": pdf_errors,
        }

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        通过 chunk_id 从 LanceDB 的 chunks 表中取回一条记录。

        LanceDB 当前版本的 Table API 不直接提供 where/filter 的非向量查询；
        这里使用 search + where(prefilter) 进行精确筛选。
        """
        if not chunk_id or self.db is None:
            return None
        if self.chunks_table not in (self.db.table_names() if self.db else []):
            return None

        tbl = self.db.open_table(self.chunks_table)
        try:
            dim = int(tbl.schema.field("vector").type.list_size)
        except Exception:
            return None

        # 仅用于触发查询；where(prefilter) 会先把候选集筛到 chunk_id 命中
        zero_vec = [0.0] * dim
        safe_id = str(chunk_id).replace("'", "''")
        try:
            hits = tbl.search(zero_vec).where(f"chunk_id = '{safe_id}'", prefilter=True).limit(1).to_list()
            return hits[0] if hits else None
        except Exception:
            return None

    def search_chunks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        返回 chunk 级别命中（包含 chunk_id / paper_id / text / 距离），并补充 paper 标题等元数据。
        """
        if self.db is None:
            raise RuntimeError("KnowledgeBase not initialized. Call initialize_db() first.")
        if self.chunks_table not in (self.db.table_names() if self.db else []):
            return []

        qv = get_embedding(
            query,
            model=self.embedding_model,
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url,
        )
        if not qv:
            return []

        tbl = self.db.open_table(self.chunks_table)
        hits = tbl.search(qv).limit(limit).to_list()
        if not hits:
            return []

        ids = list({h.get("paper_id") for h in hits if h.get("paper_id")})
        meta = self._get_papers_by_ids(ids)

        enriched: List[Dict[str, Any]] = []
        for h in hits:
            pid = h.get("paper_id")
            m = meta.get(pid, {}) if pid else {}
            # 解析 authors/keywords（用于 UI 展示/上层分析）
            authors = []
            keywords = []
            try:
                authors = json.loads(m.get("authors_json") or "[]") if isinstance(m.get("authors_json"), str) else []
                if not isinstance(authors, list):
                    authors = []
            except Exception:
                authors = []
            try:
                keywords = json.loads(m.get("keywords_json") or "[]") if isinstance(m.get("keywords_json"), str) else []
                if not isinstance(keywords, list):
                    keywords = []
            except Exception:
                keywords = []

            enriched.append(
                {
                    "paper_id": pid,
                    "title": m.get("title", ""),
                    "abstract": m.get("abstract", ""),
                    "authors": authors,
                    "keywords": keywords,
                    "year": m.get("year", None),
                    "venue_id": m.get("venue_id", None),
                    "rating": m.get("rating", None),
                    "decision": m.get("decision", None),
                    "presentation": m.get("presentation", None),
                    "chunk_id": h.get("chunk_id"),
                    "source": h.get("source"),
                    "chunk_index": h.get("chunk_index"),
                    "text": h.get("text"),
                    "_distance": h.get("_distance", None),
                }
            )
        return enriched

    def search_semantic(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：默认优先在 chunk 表中搜索，然后聚合为 paper 结果返回（兼容旧接口）。
        """
        if self.db is None:
            raise RuntimeError("KnowledgeBase not initialized. Call initialize_db() first.")

        print(f"Semantic searching for: {query}")
        query_vector = get_embedding(
            query,
            model=self.embedding_model,
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url,
        )
        if not query_vector:
            return []

        # 优先 chunk 表
        if self.chunks_table in self.db.table_names():
            tbl = self.db.open_table(self.chunks_table)
            # 多取一些 chunk，聚合后再截断
            raw = tbl.search(query_vector).limit(max(limit * 8, 20)).to_list()
            if not raw:
                return []

            best_by_paper: Dict[str, Dict[str, Any]] = {}
            for r in raw:
                pid = r.get("paper_id")
                if not pid:
                    continue
                dist = r.get("_distance", None)
                if pid not in best_by_paper or (dist is not None and dist < best_by_paper[pid].get("_distance", 1e9)):
                    best_by_paper[pid] = r

            # 按距离排序（越小越相似）
            ranked = sorted(best_by_paper.items(), key=lambda kv: kv[1].get("_distance", 1e9))[:limit]
            paper_ids = [pid for pid, _ in ranked]
            meta = self._get_papers_by_ids(paper_ids)

            results: List[Dict[str, Any]] = []
            for pid, best_chunk in ranked:
                m = meta.get(pid, {})
                results.append(
                    {
                        "id": pid,
                        "title": m.get("title", ""),
                        "abstract": m.get("abstract", ""),
                        "rating": m.get("rating", None),
                        "_distance": best_chunk.get("_distance", None),
                        "best_chunk": {
                            "chunk_id": best_chunk.get("chunk_id"),
                            "source": best_chunk.get("source"),
                            "chunk_index": best_chunk.get("chunk_index"),
                            "text": best_chunk.get("text"),
                            "_distance": best_chunk.get("_distance", None),
                        },
                    }
                )
            return results

        # fallback：paper 表
        if self.papers_table not in self.db.table_names():
            print("No vector table found. Please ingest data first.")
            return []

        tbl = self.db.open_table(self.papers_table)
        return tbl.search(query_vector).limit(limit).to_list()
