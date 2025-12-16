from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from src.utils.cancel import MujicaCancelled, check_cancel
from src.utils.json_utils import extract_json_object


def _env_truthy(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


class PlannerAgent:
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.llm = llm_client
        self.model = model

    def generate_plan(
        self,
        user_query: str,
        db_stats: Dict,
        *,
        cancel_event: Optional[Any] = None,
    ) -> Dict:
        """
        Generates a research plan/outline based on user query and DB stats using LLM.
        """
        print(f"Planning research for: {user_query} using {self.model}")
        check_cancel(cancel_event, stage="planner_start")
        
        system_prompt = """
你是 MUJICA 的 Planner（中文输出）。你的任务是：根据用户主题与数据库统计信息，生成一个研究计划（JSON）。

强约束（必须遵守）：
1) 只输出一个 JSON object，不要输出任何额外文字/解释/Markdown/代码块。
2) JSON 必须包含字段：
   - title: string
   - sections: array（3~5 个）
   - estimated_papers: number（5~20）
3) 每个 section 必须包含：
   - name: string
   - search_query: string（用于语义检索/关键词检索）
   - 可选 filters（min_rating / decision_in / presentation_in / year_in / min_year / max_year / author_contains / keyword_contains / title_contains / venue_contains）
   - 可选 top_k_papers / top_k_chunks

JSON 示例结构（仅结构示意，不要照抄内容）：
{
  "title": "…",
  "global_filters": {
    "min_rating": 6.0,
    "decision_in": ["Accept"],
    "year_in": [2024]
  },
  "sections": [
    {
      "name": "…",
      "search_query": "…",
      "filters": {"min_rating": 6.0},
      "top_k_papers": 10,
      "top_k_chunks": 40
    }
  ],
  "estimated_papers": 10
}
""".strip()
        
        user_prompt = f"""
用户主题: "{user_query}"
数据库统计: {db_stats}
""".strip()
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # 优先尝试 JSON mode（部分模型/网关不支持，会报 code=20024）
            # 可通过 MUJICA_DISABLE_JSON_MODE=1 强制关闭（例如 GLM 等）
            if not _env_truthy("MUJICA_DISABLE_JSON_MODE"):
                try:
                    check_cancel(cancel_event, stage="planner_llm_json_before")
                    response = self.llm.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                    check_cancel(cancel_event, stage="planner_llm_json_after")
                    content = response.choices[0].message.content or ""
                    plan = json.loads(content)
                    if isinstance(plan, dict) and plan.get("sections"):
                        return plan
                except Exception as e:
                    print(f"Planner json_mode failed: {e} (fallback to plain JSON)")

            # fallback：不用 response_format，让模型按提示输出 JSON，再做提取/解析
            check_cancel(cancel_event, stage="planner_llm_plain_before")
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            check_cancel(cancel_event, stage="planner_llm_plain_after")
            content = response.choices[0].message.content or ""
            plan = extract_json_object(content)
            if isinstance(plan, dict) and plan.get("sections"):
                return plan

            raise ValueError("Planner returned invalid plan JSON.")
        except MujicaCancelled:
            raise
        except Exception as e:
            print(f"Error generating plan: {e}")
            # Fallback mock plan
            return {
                "title": "Error in Planning",
                "sections": [{"name": "Error", "search_query": "error"}],
                "estimated_papers": 0,
                "_error": str(e),
            }

    def refine_plan(self, original_plan: Dict, user_feedback: str) -> Dict:
        """
        Updates the plan based on user feedback.
        """
        # For simplicity in this iteration, we just return the original or could add logic here.
        print("Refining plan (Mock implementation)...")
        return original_plan
