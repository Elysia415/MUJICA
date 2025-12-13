from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data_engine.storage import KnowledgeBase


class ResearcherAgent:
    def __init__(self, kb: KnowledgeBase, llm_client, model: str = "gpt-4o"):
        self.kb = kb
        self.llm = llm_client
        self.model = model

    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        if df is None or df.empty or not filters:
            return df

        out = df.copy()
        min_rating = filters.get("min_rating", None)
        if isinstance(min_rating, (int, float)):
            out = out[out["rating"].notna() & (out["rating"] >= float(min_rating))]

        year_in = filters.get("year_in", None)
        if isinstance(year_in, list) and year_in:
            out = out[out["year"].isin(year_in)]

        decision_in = filters.get("decision_in", None)
        if isinstance(decision_in, list) and decision_in:
            # decision 可能为空
            out = out[out["decision"].fillna("").isin(decision_in)]

        return out

    def execute_research(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        执行研究：结构化过滤 + 语义检索（chunk）+ 生成可追溯研究笔记（含证据片段）。
        """
        print("Starting research phase...")
        research_notes: List[Dict[str, Any]] = []

        metadata_df = self.kb.search_structured()
        global_filters = plan.get("global_filters") or {}
        default_top_papers = int(plan.get("estimated_papers") or 10)
        default_top_papers = max(5, min(default_top_papers, 20))

        for section in plan.get("sections", []):
            section_name = section.get("name") or "Section"
            query = section.get("search_query") or ""
            print(f"Researching section: {section_name} (Query: {query})")

            section_filters = dict(global_filters)
            section_filters.update(section.get("filters") or {})

            top_k_papers = int(section.get("top_k_papers") or default_top_papers)
            top_k_papers = max(3, min(top_k_papers, 20))

            top_k_chunks = int(section.get("top_k_chunks") or max(top_k_papers * 4, 40))
            top_k_chunks = max(20, min(top_k_chunks, 120))

            # 1) structured filtering -> 候选 paper_id 集合
            allowed_paper_ids: Optional[set[str]] = None
            if isinstance(metadata_df, pd.DataFrame) and not metadata_df.empty and section_filters:
                filtered = self._apply_filters(metadata_df, section_filters)
                allowed_paper_ids = set(filtered["id"].tolist())

            # 2) chunk-level retrieval
            chunk_hits = self.kb.search_chunks(query, limit=top_k_chunks)
            if allowed_paper_ids is not None:
                chunk_hits = [h for h in chunk_hits if h.get("paper_id") in allowed_paper_ids]

            if not chunk_hits:
                research_notes.append(
                    {
                        "section": section_name,
                        "query": query,
                        "content": "未检索到足够证据（可能尚未入库/未生成向量/过滤条件过严）。",
                        "sources": [],
                        "evidence": [],
                        "filters": section_filters,
                    }
                )
                continue

            # 3) 选 Top-N papers，并为每篇 paper 取前若干 chunks 作为证据
            by_paper: Dict[str, List[Dict[str, Any]]] = {}
            for h in chunk_hits:
                pid = h.get("paper_id")
                if not pid:
                    continue
                by_paper.setdefault(pid, []).append(h)

            # 按最优距离排序 papers
            paper_ranked = sorted(
                by_paper.items(),
                key=lambda kv: min([x.get("_distance", 1e9) for x in kv[1]]),
            )[:top_k_papers]

            evidence: List[Dict[str, Any]] = []
            source_ids: List[str] = []
            chunks_per_paper = 2
            for pid, hits in paper_ranked:
                source_ids.append(pid)
                hits_sorted = sorted(hits, key=lambda x: x.get("_distance", 1e9))[:chunks_per_paper]
                for hh in hits_sorted:
                    evidence.append(
                        {
                            "paper_id": pid,
                            "title": hh.get("title", ""),
                            "chunk_id": hh.get("chunk_id"),
                            "source": hh.get("source"),
                            "chunk_index": hh.get("chunk_index"),
                            "text": (hh.get("text") or "")[:1500],
                            "rating": hh.get("rating"),
                            "decision": hh.get("decision"),
                            "_distance": hh.get("_distance"),
                        }
                    )

            # 4) LLM：基于证据生成“中间态笔记”
            #    - 让模型输出 JSON，便于后续 writer/verifier 使用
            evidence_text = ""
            for e in evidence:
                evidence_text += (
                    f"\n[Paper ID: {e['paper_id']} | Chunk: {e['chunk_id']} | Source: {e['source']}]\n"
                    f"Title: {e['title']}\n"
                    f"Snippet: {e['text']}\n"
                )

            system_prompt = (
                "你是 MUJICA 的 Researcher（中文输出）。"
                "只能基于给定的 Evidence Snippets 写研究笔记；不允许编造。"
                "如果证据不足，请明确说明未知/证据缺失。"
            )
            user_prompt = f"""
任务：为报告章节「{section_name}」撰写研究笔记（中间态）。
章节检索词：{query}
结构化过滤条件：{json.dumps(section_filters, ensure_ascii=False)}

Evidence Snippets（可引用 chunk_id 以便溯源）：
{evidence_text}

请返回 JSON：
{{
  "summary": "<本章节的核心发现，200-400字>",
  "key_points": [
    {{"point": "<要点>", "citations": [{{"paper_id": "...", "chunk_id": "..."}}]}}
  ]
}}
"""

            summary_content = ""
            key_points: List[Dict[str, Any]] = []
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                parsed = json.loads(response.choices[0].message.content or "{}")
                summary_content = parsed.get("summary", "") or ""
                key_points = parsed.get("key_points", []) or []
            except Exception as e:
                print(f"Error summarising: {e}")
                summary_content = "研究笔记生成失败（LLM 调用/JSON 解析异常）。"

            research_notes.append(
                {
                    "section": section_name,
                    "query": query,
                    "content": summary_content,
                    "key_points": key_points,
                    "sources": source_ids,
                    "evidence": evidence,
                    "filters": section_filters,
                }
            )

        print(f"Completed research for {len(research_notes)} sections.")
        return research_notes
