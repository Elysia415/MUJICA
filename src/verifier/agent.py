from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


class VerifierAgent:
    """
    事实核查（NLI/Entailment）：逐句对齐引用到的 evidence chunk。

    期望引用格式：
      [Paper ID: <paper_id> | Chunk: <chunk_id>]
    """

    _CIT_RE = re.compile(r"\[Paper ID:\s*([^\]|]+?)\s*\|\s*Chunk:\s*([^\]]+?)\]")
    _PAPER_ONLY_RE = re.compile(r"\[Paper ID:\s*([^\]]+?)\]")

    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.llm = llm_client
        self.model = model

    def _extract_claims(self, report_text: str) -> List[Dict[str, Any]]:
        text = (report_text or "").replace("\r\n", "\n")
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        claims: List[Dict[str, Any]] = []
        for line in lines:
            # 进一步按句号/问号/感叹号切分，降低“一个段落多个句子”带来的误配
            parts = re.split(r"(?<=[。！？.!?])\s+", line)
            for part in parts:
                cits = self._CIT_RE.findall(part)
                if not cits:
                    continue

                claim_text = self._CIT_RE.sub("", part).strip()
                claim_text = re.sub(r"\s+", " ", claim_text).strip()
                if not claim_text:
                    continue

                citations = [{"paper_id": p.strip(), "chunk_id": c.strip()} for p, c in cits]
                claims.append({"claim": claim_text, "raw": part, "citations": citations})

        return claims

    def verify_report(self, report_text: str, source_data: Dict[str, Any]) -> Dict[str, Any]:
        print("Verifying report integrity...")

        # evidence index: chunk_id -> text
        chunk_map = source_data.get("chunks") if isinstance(source_data, dict) else None
        if not isinstance(chunk_map, dict):
            chunk_map = {}

        claims = self._extract_claims(report_text)
        if not claims:
            # fallback：兼容旧格式，仅检查是否存在 [Paper ID: ...]
            citations = self._PAPER_ONLY_RE.findall(report_text or "")
            if not citations:
                return {
                    "is_valid": False,
                    "score": 0.0,
                    "notes": "未发现任何引用。报告必须将结论锚定到来源。",
                }
            return {
                "is_valid": True,
                "score": 0.5,
                "notes": "仅检测到 paper-level 引用（未提供 chunk 级证据），跳过逐句 NLI 核查。",
                "stats": {"unique_paper_citations": len(set([c.strip() for c in citations]))},
            }

        # 1) 结构性检查：引用的 chunk 是否存在
        missing_chunks = []
        if chunk_map:
            for c in claims:
                for cit in c["citations"]:
                    cid = cit["chunk_id"]
                    if cid not in chunk_map:
                        missing_chunks.append(cid)

        if missing_chunks:
            uniq = sorted(set(missing_chunks))
            return {
                "is_valid": False,
                "score": 0.0,
                "notes": f"发现引用了未知的 chunk（数量={len(uniq)}），无法溯源核查。",
                "missing_chunks": uniq[:50],
            }

        # 2) 若无 LLM：只做结构性核查
        if self.llm is None:
            return {
                "is_valid": True,
                "score": 0.6,
                "notes": "LLM 不可用：仅完成引用结构/可溯源性检查（未做 entailment）。",
                "stats": {"claims_checked": len(claims), "mode": "structural_only"},
            }

        # 3) entailment 核查（抽样/限额）
        max_claims = int(source_data.get("max_claims", 12)) if isinstance(source_data, dict) else 12
        max_claims = max(3, min(max_claims, 30))

        evaluations: List[Dict[str, Any]] = []
        contradicts = 0
        supports = 0
        scores: List[float] = []

        system_prompt = (
            "你是严格的事实核查/NLI 模型。"
            "给定 Claim 与 Evidence（来自论文片段），判断 Evidence 是否支持 Claim。"
            "只输出 JSON，不要输出其他文本。"
        )

        for item in claims[:max_claims]:
            claim = item["claim"]
            cited_chunks = item["citations"]

            ev_texts = []
            for cit in cited_chunks:
                cid = cit["chunk_id"]
                t = chunk_map.get(cid, "")
                if t:
                    ev_texts.append(f"[Chunk: {cid}]\n{t}")

            evidence_block = "\n\n".join(ev_texts)[:6000]

            user_prompt = f"""
Claim:
{claim}

Evidence:
{evidence_block}

请返回 JSON：
{{
  "label": "entailed|contradicted|unknown",
  "score": 0.0,
  "reason": "一句话解释"
}}
"""
            try:
                resp = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                parsed = json.loads(resp.choices[0].message.content or "{}")
            except Exception as e:
                parsed = {"label": "unknown", "score": 0.0, "reason": f"verification_error: {e}"}

            label = str(parsed.get("label", "unknown")).lower().strip()
            score = float(parsed.get("score", 0.0) or 0.0)
            reason = str(parsed.get("reason", ""))

            if label == "contradicted":
                contradicts += 1
            elif label == "entailed":
                supports += 1
            scores.append(max(0.0, min(1.0, score)))

            evaluations.append(
                {
                    "claim": claim,
                    "citations": cited_chunks,
                    "label": label,
                    "score": score,
                    "reason": reason,
                }
            )

        overall = sum(scores) / len(scores) if scores else 0.0
        is_valid = (overall >= 0.7) and (contradicts == 0)

        notes = f"checked={len(evaluations)}, supports={supports}, contradicts={contradicts}, score={overall:.2f}"
        return {
            "is_valid": is_valid,
            "score": overall,
            "notes": notes,
            "evaluations": evaluations,
            "stats": {"claims_total": len(claims), "claims_checked": len(evaluations)},
        }
