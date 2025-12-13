from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

import lancedb
import pandas as pd

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
        embedding_model: str = "text-embedding-3-small",
        chunk_max_tokens: int = 350,
        chunk_overlap_tokens: int = 60,
    ):
        self.db_path = db_path
        self.db: Optional[lancedb.db.LanceDBConnection] = None

        self.papers_table = papers_table
        self.chunks_table = chunks_table
        self.embedding_model = embedding_model
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

        # 2) 写入 paper-level 向量（便于 fallback / 简单检索）
        paper_rows = []
        for p in papers:
            pid = str(p.get("id") or "").strip()
            if not pid:
                continue
            title = (p.get("title") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            text_to_embed = f"Title: {title}\nAbstract: {abstract}".strip()
            vec = get_embedding(text_to_embed, model=self.embedding_model)
            if not vec:
                continue
            paper_rows.append(
                {
                    "id": pid,
                    "title": title,
                    "abstract": abstract,
                    "rating": p.get("rating"),
                    "year": p.get("year"),
                    "vector": vec,
                }
            )

        if paper_rows:
            if self.papers_table in self.db.table_names():
                tbl = self.db.open_table(self.papers_table)
                # 删除旧记录再插入，避免重复
                ids = [r["id"] for r in paper_rows]
                try:
                    ids_sql = ", ".join([f"'{i}'" for i in ids])
                    tbl.delete(f"id IN ({ids_sql})")
                except Exception:
                    pass
                tbl.add(paper_rows)
            else:
                self.db.create_table(self.papers_table, data=paper_rows)

        # 3) 写入 chunk-level 向量（用于证据溯源）
        chunk_rows = []
        for p in papers:
            pid = str(p.get("id") or "").strip()
            if not pid:
                continue

            # 删除该 paper 旧 chunks（避免重复）
            if self.chunks_table in self.db.table_names():
                try:
                    self.db.open_table(self.chunks_table).delete(f"paper_id = '{pid}'")
                except Exception:
                    pass

            sources: List[tuple[str, str]] = []
            title = (p.get("title") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            tldr = (p.get("tldr") or "").strip()
            content = (p.get("content") or "").strip()

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
            embeddings = get_embeddings([r["text"] for r in chunk_rows], model=self.embedding_model)
            rows_to_insert = []
            for r, vec in zip(chunk_rows, embeddings):
                if vec:
                    rr = dict(r)
                    rr["vector"] = vec
                    rows_to_insert.append(rr)

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
            p.get("rating"),
            json.dumps(p, ensure_ascii=False),
        )

        cur = self._meta_conn.cursor()
        cur.execute(
            """
            INSERT INTO papers (
                id, title, abstract, tldr, authors_json, keywords_json,
                year, venue_id, forum, number, pdf_url, pdf_path,
                decision, rating, raw_json, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, CURRENT_TIMESTAMP
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

    def search_semantic(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：默认优先在 chunk 表中搜索，然后聚合为 paper 结果返回（兼容旧接口）。
        """
        if self.db is None:
            raise RuntimeError("KnowledgeBase not initialized. Call initialize_db() first.")

        print(f"Semantic searching for: {query}")
        query_vector = get_embedding(query, model=self.embedding_model)
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
