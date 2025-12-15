from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Tuple


class WriterAgent:
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.llm = llm_client
        self.model = model

    _REF_RE = re.compile(r"\[(R\d+)\]")
    _REF_SECTION_RE = re.compile(r"(?im)^\s*#{1,6}\s+(references|参考文献)\s*$")
    _SENT_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+")

    def _strip_references_section(self, text: str) -> str:
        """
        防御性处理：如果模型自己输出了 References/参考文献，小节会与我们自动生成的 References 重复。
        """
        s = (text or "").replace("\r\n", "\n")
        m = self._REF_SECTION_RE.search(s)
        if not m:
            return s.strip()
        return s[: m.start()].rstrip()

    def _build_ref_catalog(self, research_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        为所有 evidence chunk 分配稳定的 Ref ID（R1/R2/...）。

        目标：
        - 报告正文只出现 [R#]（对人友好、避免 paper_id 乱码/冗长）
        - References 小节由我们自动生成（标题+来源+片段+snippet）
        - Verifier 用 ref_map 把 [R#] 映射回 chunk_id 再做核查
        """
        ref_items: List[Dict[str, Any]] = []
        chunk_to_ref: Dict[str, str] = {}

        for note in research_notes or []:
            for e in (note.get("evidence") or []):
                cid = str(e.get("chunk_id") or "").strip()
                if not cid:
                    continue
                if cid in chunk_to_ref:
                    continue

                ref_id = f"R{len(ref_items) + 1}"
                chunk_to_ref[cid] = ref_id

                ref_items.append(
                    {
                        "ref": ref_id,
                        "paper_id": e.get("paper_id"),
                        "title": e.get("title") or "",
                        "chunk_id": cid,
                        "source": e.get("source") or "",
                        "chunk_index": e.get("chunk_index"),
                        "text": e.get("text") or "",
                    }
                )

        ref_map = {it["ref"]: it["chunk_id"] for it in ref_items}
        return {"ref_items": ref_items, "chunk_to_ref": chunk_to_ref, "ref_map": ref_map}

    def _render_references(self, report_text: str, ref_items: List[Dict[str, Any]]) -> str:
        used: List[str] = []
        seen = set()
        for rid in self._REF_RE.findall(report_text or ""):
            if rid in seen:
                continue
            seen.add(rid)
            used.append(rid)

        if not used:
            return ""

        by_ref = {it.get("ref"): it for it in (ref_items or []) if it.get("ref")}

        def _source_label(src: str) -> str:
            s = (src or "").strip()
            if s.startswith("review_"):
                try:
                    # review_0 => 评审 #1（对用户更直观）
                    idx = int(s.split("_", 1)[1])
                    return f"评审 #{idx + 1}"
                except Exception:
                    return "评审"
            return {
                "meta": "元信息",
                "title_abstract": "标题+摘要",
                "tldr": "TL;DR",
                "full_text": "正文",
                "decision": "最终决策说明",
                "rebuttal": "作者 Rebuttal/Response",
            }.get(s, s or "unknown")

        lines = ["", "## References", ""]
        for rid in used:
            it = by_ref.get(rid) or {}
            title = (it.get("title") or "").strip() or "（无标题）"
            src = _source_label(str(it.get("source") or ""))
            cidx = it.get("chunk_index")
            try:
                cidx_disp = int(cidx) if cidx is not None else None
            except Exception:
                cidx_disp = None

            snippet = (it.get("text") or "").strip()
            snippet = re.sub(r"\s+", " ", snippet)
            if len(snippet) > 240:
                snippet = snippet[:240].rstrip() + "…"

            loc = f"{src}" + (f" · 片段 {cidx_disp}" if cidx_disp is not None else "")
            # 不在报告中暴露 paper_id/chunk_id（用户嫌太“工程”）；需要溯源可在右侧 Evidence 面板查看
            lines.append(f"- **[{rid}]《{title}》**：{loc} — {snippet}")

        return "\n".join(lines).rstrip() + "\n"

    def write_report(
        self,
        plan: Dict[str, Any],
        research_notes: List[Dict[str, Any]],
        *,
        on_progress: Any = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        循证写作：严格基于 research_notes.evidence 生成 Markdown 报告。
        - 句级引用：使用 Ref ID（对人友好）：[R1] [R2] ...
        - 参考文献（References）由系统自动补全：标题 + 来源类型 + 片段 + snippet
        """
        print("Writing final report...")

        title = plan.get("title", "Research Report")

        def _emit(stage: str, **payload):  # noqa: ANN001
            if callable(on_progress):
                try:
                    on_progress({"stage": stage, **payload})
                except Exception:
                    pass

        t_all = time.time()
        _emit("write_start", title=title)
        ref_ctx = self._build_ref_catalog(research_notes)
        ref_items = ref_ctx.get("ref_items") or []
        chunk_to_ref = ref_ctx.get("chunk_to_ref") or {}
        _emit("write_refs_built", refs_total=len(ref_items))

        sections_payload = []
        evidence_total = 0
        for note in research_notes:
            evidence = note.get("evidence") or []
            evidence_total += len(evidence)

            # 将 key_points 的 citations（paper_id/chunk_id）映射为 refs（R#）
            key_points_out = []
            for kp in (note.get("key_points") or []):
                if not isinstance(kp, dict):
                    continue
                refs = []
                for cit in (kp.get("citations") or []):
                    if not isinstance(cit, dict):
                        continue
                    cid = str(cit.get("chunk_id") or "").strip()
                    rid = chunk_to_ref.get(cid)
                    if rid and rid not in refs:
                        refs.append(rid)
                key_points_out.append({"point": kp.get("point"), "refs": refs})

            allowed_refs = []
            for e in evidence:
                cid = str(e.get("chunk_id") or "").strip()
                rid = chunk_to_ref.get(cid)
                if rid and rid not in allowed_refs:
                    allowed_refs.append(rid)
            sections_payload.append(
                {
                    "section": note.get("section"),
                    "query": note.get("query"),
                    "summary": note.get("content"),
                    "key_points": key_points_out,
                    "evidence": [
                        {
                            "ref": chunk_to_ref.get(str(e.get("chunk_id") or "").strip()),
                            "paper_id": e.get("paper_id"),
                            "title": e.get("title"),
                            "year": e.get("year"),
                            "rating": e.get("rating"),
                            "decision": e.get("decision"),
                            "presentation": e.get("presentation"),
                            "chunk_id": e.get("chunk_id"),
                            "source": e.get("source"),
                            "chunk_index": e.get("chunk_index"),
                            "text": e.get("text"),
                        }
                        for e in evidence
                    ],
                    "allowed_refs": allowed_refs,
                }
            )
        _emit(
            "write_payload_built",
            sections=len(sections_payload),
            evidence_snippets=evidence_total,
            allowed_refs_total=len(ref_items),
        )

        system_prompt = """
你是 MUJICA 的 Writer（循证写作，中文输出）。
你的目标：基于提供的研究笔记与 Evidence Snippets，写一篇客观、学术风格的 Markdown 报告。

严格规则（必须遵守）：
1) 只能使用输入中的信息；不允许补充常识性细节或编造。
2) 每一个“事实性/结论性”句子末尾必须附上至少 1 个引用，引用格式必须为：
   [R<number>] 例如：[R1] 或 [R12][R3]
3) 只能引用该章节 allowed_refs 中出现的 Ref ID；禁止引用 paper_id/chunk_id 等内部标识。
4) 报告中提到论文时，优先使用《论文标题》而不是 Paper ID。
5) 若证据不足以支持某个结论，必须写成“证据不足/尚无直接证据支持……”，并且该句不要强行引用。
6) 不要输出 References/参考文献 小节（系统会自动生成）。

写作质量要求（尽量做到更“完整/可读/有洞察”，但仍必须循证）：
7) 每个章节至少包含：
   - 1 段「本章节结论概述」（建议 150-300 字）
   - 1 个「评审优点/Strengths 模式」列表（至少 4 条；优先引用 source=review_* / decision / rebuttal）
   - 1 个「评审不足/Weaknesses/Concerns 模式」列表（至少 4 条；优先引用 source=review_* / decision / rebuttal）
   - 1 个「代表性证据例子」小节：给出 2-3 条“原话级”短引用/转述（每条都要引用 [R#]）
8) 如果用户问题与“高分/低分、对比、评审关注点”相关：
   - 请把重点放在评审文本里的优势/不足特征（而不是论文技术细节本身）。
   - 明确指出高分组与低分组在“优点/不足/关注点”上的差异，并用证据支持。
9) 报告表达：优先结构化（小标题/要点列表/对比表格）。如能基于证据做出“主题归纳”（例如可复现性、实验设置、理论严谨性、写作清晰度等），请归纳成更抽象的类别，并用多条证据支撑。
"""

        user_prompt = f"""
Report Title: {title}

Sections (JSON):
{json.dumps(sections_payload, ensure_ascii=False)}

请生成完整 Markdown 报告。
"""

        try:
            prompt_chars = len(system_prompt) + len(user_prompt)
            _emit("write_llm_call", model=self.model, prompt_chars=prompt_chars, refs_total=len(ref_items))
            t_llm = time.time()

            # max_tokens：允许输出更长的报告；可用 MUJICA_WRITER_MAX_TOKENS 控制
            llm_kwargs: Dict[str, Any] = {}
            try:
                max_tokens = int(os.getenv("MUJICA_WRITER_MAX_TOKENS", "4096") or 4096)
            except Exception:
                max_tokens = 4096
            max_tokens = max(256, min(max_tokens, 16_384))
            llm_kwargs["max_tokens"] = max_tokens

            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **llm_kwargs,
            )
            dt_llm = time.time() - t_llm

            # token usage（部分 OpenAI-compatible provider 可能不返回 usage）
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
            total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None

            body = response.choices[0].message.content or ""
            body = self._strip_references_section(body)
            refs_md = self._render_references(body, ref_items)
            final = (body.rstrip() + "\n" + refs_md).strip() + "\n"

            # ---------- 高级统计（用于 UI 日志与排查） ----------
            used_refs = []
            seen = set()
            for rid in self._REF_RE.findall(body or ""):
                if rid in seen:
                    continue
                seen.add(rid)
                used_refs.append(rid)

            # 证据来源分布（meta/full_text/review/decision/rebuttal 等）
            src_counter = Counter()
            for it in ref_items:
                src_counter[str(it.get("source") or "unknown")] += 1

            # 句子级引用覆盖率（粗略指标，用于 debug）
            sents = []
            for line in (body or "").splitlines():
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    continue
                # 简单跳过明显的引用清单/分隔线
                if s.lower() in {"references", "参考文献"}:
                    continue
                sents.extend([x.strip() for x in self._SENT_SPLIT_RE.split(s) if x.strip()])

            total_sents = len(sents)
            cited_sents = sum(1 for s in sents if self._REF_RE.search(s))
            coverage = (cited_sents / total_sents) if total_sents > 0 else 0.0

            writer_stats = {
                "title": title,
                "sections": int(len(sections_payload)),
                "evidence_snippets": int(evidence_total),
                "refs_total": int(len(ref_items)),
                "refs_used": int(len(used_refs)),
                "refs_unused": int(max(0, len(ref_items) - len(used_refs))),
                "sources_top": dict(src_counter.most_common(10)),
                "prompt_chars": int(prompt_chars),
                "body_chars": int(len(body)),
                "final_chars": int(len(final)),
                "dt_llm_sec": float(dt_llm),
                "dt_total_sec": float(time.time() - t_all),
                "sentences_total_est": int(total_sents),
                "sentences_cited_est": int(cited_sents),
                "citation_coverage_est": float(coverage),
                "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
                "completion_tokens": int(completion_tokens) if isinstance(completion_tokens, int) else None,
                "total_tokens": int(total_tokens) if isinstance(total_tokens, int) else None,
            }
            ref_ctx["writer_stats"] = writer_stats

            _emit(
                "write_done",
                dt_llm_sec=writer_stats["dt_llm_sec"],
                refs_used=writer_stats["refs_used"],
                refs_total=writer_stats["refs_total"],
                coverage=writer_stats["citation_coverage_est"],
                body_chars=writer_stats["body_chars"],
                total_tokens=writer_stats["total_tokens"],
            )
            return final, ref_ctx
        except Exception as e:
            print(f"Error writing report: {e}")
            try:
                ref_ctx["writer_stats"] = {
                    "title": title,
                    "error": str(e),
                    "dt_total_sec": float(time.time() - t_all),
                }
            except Exception:
                pass
            _emit("write_error", error=str(e))
            return "Error generating report.", ref_ctx
