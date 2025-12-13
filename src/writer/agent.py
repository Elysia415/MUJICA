from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


class WriterAgent:
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.llm = llm_client
        self.model = model

    def _build_allowed_citations(self, evidence: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        allowed: List[Tuple[str, str]] = []
        for e in evidence or []:
            pid = e.get("paper_id")
            cid = e.get("chunk_id")
            if pid and cid:
                allowed.append((str(pid), str(cid)))
        # 去重保持顺序
        seen = set()
        out = []
        for pid, cid in allowed:
            key = (pid, cid)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def write_report(self, plan: Dict[str, Any], research_notes: List[Dict[str, Any]]) -> str:
        """
        循证写作：严格基于 research_notes.evidence 生成 Markdown 报告。
        - 句级引用，引用到 chunk： [Paper ID: <paper_id> | Chunk: <chunk_id>]
        """
        print("Writing final report...")

        title = plan.get("title", "Research Report")

        sections_payload = []
        for note in research_notes:
            evidence = note.get("evidence") or []
            allowed = self._build_allowed_citations(evidence)
            sections_payload.append(
                {
                    "section": note.get("section"),
                    "query": note.get("query"),
                    "summary": note.get("content"),
                    "key_points": note.get("key_points", []),
                    "evidence": [
                        {
                            "paper_id": e.get("paper_id"),
                            "title": e.get("title"),
                            "chunk_id": e.get("chunk_id"),
                            "source": e.get("source"),
                            "text": e.get("text"),
                        }
                        for e in evidence
                    ],
                    "allowed_citations": [{"paper_id": pid, "chunk_id": cid} for pid, cid in allowed],
                }
            )

        system_prompt = """
你是 MUJICA 的 Writer（循证写作，中文输出）。
你的目标：基于提供的研究笔记与 Evidence Snippets，写一篇客观、学术风格的 Markdown 报告。

严格规则（必须遵守）：
1) 只能使用输入中的信息；不允许补充常识性细节或编造。
2) 每一个“事实性/结论性”句子末尾必须附上至少 1 个引用，格式必须为：
   [Paper ID: <paper_id> | Chunk: <chunk_id>]
3) 只能引用 allowed_citations 中出现的 (paper_id, chunk_id) 组合；禁止引用其他 ID。
4) 若证据不足以支持某个结论，必须写成“证据不足/尚无直接证据支持……”，并且该句不要强行引用。
5) 报告结构：标题 + 章节（按输入顺序）+ 最后给出 References（列出出现过的 Paper ID）。
"""

        user_prompt = f"""
Report Title: {title}

Sections (JSON):
{json.dumps(sections_payload, ensure_ascii=False)}

请生成完整 Markdown 报告。
"""

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error writing report: {e}")
            return "Error generating report."
