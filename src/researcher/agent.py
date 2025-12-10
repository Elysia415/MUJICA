from typing import List, Dict
from src.data_engine.storage import KnowledgeBase
from src.utils.llm import chat
import json

class ResearcherAgent:
    def __init__(self, kb: KnowledgeBase, llm_client):
        self.kb = kb
        self.llm = llm_client

    def execute_research(self, plan: Dict) -> List[Dict]:
        """
        Executes the search plan, retrieving papers and generating notes.
        为每个章节执行检索、阅读和总结
        
        Args:
            plan: 研究计划（由 Planner 生成）
        
        Returns:
            研究笔记列表，每个笔记对应一个章节
        """
        print("Starting research phase...")
        research_notes = []
        
        sections = plan.get("sections", [])
        
        for idx, section in enumerate(sections, 1):
            # 支持两种格式：简单字符串或详细字典
            if isinstance(section, str):
                section_name = section
                keywords = []
                focus = ""
            else:
                section_name = section.get("name", f"Section {idx}")
                keywords = section.get("keywords", [])
                focus = section.get("focus", "")
            
            print(f"\n[{idx}/{len(sections)}] Researching section: {section_name}")
            
            # 1. 生成搜索查询
            search_queries = self._generate_search_queries(section_name, keywords, focus)
            print(f"  Generated {len(search_queries)} search queries")
            
            # 2. 执行检索
            retrieved_papers = self._search_papers(search_queries)
            print(f"  Retrieved {len(retrieved_papers)} papers")
            
            # 3. 阅读和总结
            summary = self._summarize_papers(section_name, focus, retrieved_papers)
            
            note = {
                "section": section_name,
                "focus": focus,
                "content": summary,
                "sources": [p.get("id", 0) for p in retrieved_papers],
                "key_papers": retrieved_papers[:5]  # 保存前5篇关键论文
            }
            research_notes.append(note)
            print(f"  ✓ Completed section: {section_name}")
            
        print(f"\n✓ Research phase complete: {len(research_notes)} sections processed")
        return research_notes
    
    def _generate_search_queries(self, section_name: str, keywords: List[str], focus: str) -> List[str]:
        """
        为一个章节生成搜索查询
        
        Args:
            section_name: 章节名称
            keywords: 关键词列表
            focus: 重点关注的问题
        
        Returns:
            搜索查询列表
        """
        # 如果有关键词，直接使用
        if keywords:
            return keywords[:3]  # 限制最多3个查询
        
        # 否则基于章节名称和focus生成
        prompt = f"""为以下研究章节生成2-3个精确的搜索关键词（英文）。

章节名称：{section_name}
重点问题：{focus}

要求：
1. 关键词应该具体、技术性强
2. 适合在学术论文数据库中搜索
3. 每个关键词2-5个单词

请以 JSON 数组格式输出：["keyword1", "keyword2", "keyword3"]
只输出 JSON 数组，不要其他内容。
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
            
            queries = json.loads(response)
            return queries[:3]
            
        except Exception as e:
            print(f"  Warning: Failed to generate queries: {e}")
            # 返回简单的后备查询
            return [section_name]
    
    def _search_papers(self, queries: List[str], limit_per_query: int = 5) -> List[Dict]:
        """
        使用多个查询检索论文
        
        Args:
            queries: 搜索查询列表
            limit_per_query: 每个查询返回的最大结果数
        
        Returns:
            论文列表
        """
        all_papers = []
        seen_ids = set()
        
        for query in queries:
            # 执行语义搜索
            papers = self.kb.search_semantic(query, limit=limit_per_query)
            
            # 去重
            for paper in papers:
                paper_id = paper.get("id", hash(str(paper)))
                if paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    all_papers.append(paper)
        
        return all_papers
    
    def _summarize_papers(self, section_name: str, focus: str, papers: List[Dict]) -> str:
        """
        总结一组论文的核心发现
        
        Args:
            section_name: 章节名称
            focus: 重点关注的问题
            papers: 论文列表
        
        Returns:
            总结文本
        """
        if not papers:
            return f"No papers found for section: {section_name}"
        
        # 构建论文摘要
        papers_text = "\n\n".join([
            f"Paper {i+1}: {p.get('title', 'Untitled')}\n"
            f"Abstract: {p.get('abstract', 'No abstract available')[:300]}..."
            for i, p in enumerate(papers[:10])  # 最多总结10篇
        ])
        
        prompt = f"""请总结以下论文在「{section_name}」方面的核心发现。

重点关注：{focus}

论文信息：
{papers_text}

要求：
1. 提炼共性和差异
2. 突出关键技术和方法
3. 保持客观学术风格
4. 200-300字左右

请直接输出总结内容。
"""
        
        try:
            summary = chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                client=self.llm
            )
            return summary.strip()
            
        except Exception as e:
            print(f"  Warning: Failed to generate summary: {e}")
            # 返回简单的后备总结
            return f"Found {len(papers)} papers related to {section_name}. "\
                   f"Key papers include: {', '.join([p.get('title', 'Unknown')[:50] for p in papers[:3]])}"
