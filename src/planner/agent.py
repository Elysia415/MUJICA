from typing import List, Dict
import json
from src.utils.llm import chat

class PlannerAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def generate_plan(self, user_query: str, db_stats: Dict) -> Dict:
        """
        Generates a research plan/outline based on user query and DB stats.
        使用 LLM 生成智能的、结构化的研究大纲
        
        Args:
            user_query: 用户的研究问题
            db_stats: 数据库统计信息（如论文数量）
        
        Returns:
            包含研究大纲的字典
        """
        print(f"Planning research for: {user_query}")
        print(f"Database contains {db_stats.get('count', 0)} papers")
        
        # 构建 prompt
        prompt = f"""你是一个专业的学术研究规划专家。请为以下研究问题设计一个详细的研究大纲。

研究问题：{user_query}

数据库信息：
- 可用论文数量: {db_stats.get('count', 0)}
- 数据来源: NeurIPS 2024 论文集

请生成一个结构化的研究大纲，包含以下内容：
1. 研究主题的核心维度（3-5个主要方面）
2. 每个维度需要重点关注的问题
3. 预计需要深入阅读的论文数量
4. 建议的搜索关键词

请以 JSON 格式输出，格式如下：
{{
  "title": "报告标题",
  "sections": [
    {{
      "name": "章节名称",
      "focus": "重点关注的问题",
      "keywords": ["关键词1", "关键词2"]
    }}
  ],
  "estimated_papers": 预计论文数量,
  "search_strategy": "检索策略建议"
}}

只输出 JSON，不要其他解释。
"""
        
        try:
            response = chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                client=self.llm
            )
            
            # 尝试解析 JSON
            # 清理可能的 markdown 代码块标记
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            plan = json.loads(response)
            print(f"✓ Generated plan with {len(plan.get('sections', []))} sections")
            return plan
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response as JSON: {e}")
            print(f"Response was: {response[:200]}...")
            # 返回简单的默认计划
            return self._generate_fallback_plan(user_query)
        except Exception as e:
            print(f"Error generating plan: {e}")
            return self._generate_fallback_plan(user_query)
    
    def _generate_fallback_plan(self, user_query: str) -> Dict:
        """生成一个简单的后备计划"""
        return {
            "title": f"Research Report: {user_query}",
            "sections": [
                {
                    "name": "Introduction",
                    "focus": "Background and motivation",
                    "keywords": ["introduction", "background"]
                },
                {
                    "name": "Key Methodologies",
                    "focus": "Technical approaches and methods",
                    "keywords": ["method", "approach", "technique"]
                },
                {
                    "name": "Experimental Results",
                    "focus": "Findings and evaluations",
                    "keywords": ["results", "evaluation", "experiment"]
                },
                {
                    "name": "Conclusion",
                    "focus": "Summary and future directions",
                    "keywords": ["conclusion", "future work"]
                }
            ],
            "estimated_papers": 15,
            "search_strategy": "Semantic search with keyword filtering"
        }

    def refine_plan(self, original_plan: Dict, user_feedback: str) -> Dict:
        """
        Updates the plan based on user feedback.
        
        Args:
            original_plan: 原始计划
            user_feedback: 用户反馈
        
        Returns:
            更新后的计划
        """
        print("Refining plan based on user feedback...")
        
        prompt = f"""请根据用户反馈更新研究计划。

原始计划：
{json.dumps(original_plan, indent=2, ensure_ascii=False)}

用户反馈：
{user_feedback}

请输出更新后的计划（JSON 格式，结构与原计划相同）。
"""
        
        try:
            response = chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                client=self.llm
            )
            
            # 清理和解析
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            refined_plan = json.loads(response)
            print("✓ Plan refined successfully")
            return refined_plan
            
        except Exception as e:
            print(f"Error refining plan: {e}")
            return original_plan
