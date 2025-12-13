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
        download_pdfs: bool = True,
        parse_pdfs: bool = True,
        max_pdf_pages: Optional[int] = 12,
        max_downloads: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        papers = self.fetcher.fetch_papers(venue_id, limit=limit)

        if download_pdfs:
            self.fetcher.download_pdfs(papers, max_downloads=max_downloads or limit)

        if parse_pdfs:
            for p in papers:
                pdf_path = p.get("pdf_path")
                if pdf_path and os.path.exists(pdf_path):
                    p["content"] = self.parser.parse_pdf(pdf_path, max_pages=max_pdf_pages)

        self.kb.ingest_data(papers)
        return papers


