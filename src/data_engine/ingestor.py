from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.data_engine.fetcher import ConferenceDataFetcher
from src.data_engine.parser import PDFParser
from src.data_engine.storage import KnowledgeBase


class OpenReviewIngestor:
    """
    OpenReview -> 本地知识库 的一条龙入库管线：
    1) 拉取论文元数据（含评审/决策）
    2) 下载 PDF（可选）
    3) 解析 PDF 得到全文文本（可选）
    4) 写入 SQLite 元数据 + LanceDB 向量索引
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        fetcher: Optional[ConferenceDataFetcher] = None,
        parser: Optional[PDFParser] = None,
    ) -> None:
        self.kb = kb
        self.fetcher = fetcher or ConferenceDataFetcher()
        self.parser = parser or PDFParser()

    def ingest_venue(
        self,
        *,
        venue_id: str,
        limit: Optional[int] = None,
        accepted_only: bool = False,
        presentation_in: Optional[List[str]] = None,
        download_pdfs: bool = True,
        parse_pdfs: bool = True,
        max_pdf_pages: Optional[int] = 12,
        max_downloads: Optional[int] = None,
        on_progress: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        papers = self.fetcher.fetch_papers(venue_id, limit=limit, accepted_only=accepted_only)

        # 双重兜底：某些会议 decision 字段可能缺失/格式不同，这里再做一次过滤保证语义
        if accepted_only:
            kept = []
            for p in papers:
                d = str((p or {}).get("decision") or "").lower()
                if "accept" in d:
                    kept.append(p)
            papers = kept

        # 进一步过滤展示类型（oral/spotlight/poster/unknown）
        if accepted_only and isinstance(presentation_in, list) and presentation_in:
            allowed = set([str(x).strip().lower() for x in presentation_in if str(x).strip()])
            if allowed:
                kept = []
                for p in papers:
                    pres = str((p or {}).get("presentation") or "").strip().lower()
                    if pres in allowed:
                        kept.append(p)
                papers = kept

        if download_pdfs:
            self.fetcher.download_pdfs(papers, max_downloads=max_downloads or limit, on_progress=on_progress)

        if parse_pdfs:
            parse_targets = [p for p in papers if p.get("pdf_path") and os.path.exists(p.get("pdf_path"))]
            total = len(parse_targets)
            done = 0
            for p in parse_targets:
                done += 1
                if callable(on_progress):
                    try:
                        on_progress(
                            {
                                "stage": "parse_pdf",
                                "current": done,
                                "total": total,
                                "paper_id": p.get("id"),
                                "title": p.get("title"),
                                "pdf_path": p.get("pdf_path"),
                            }
                        )
                    except Exception:
                        pass

                pdf_path = p.get("pdf_path")
                if pdf_path and os.path.exists(pdf_path):
                    p["content"] = self.parser.parse_pdf(pdf_path, max_pages=max_pdf_pages)

        self.kb.ingest_data(papers)
        return papers


