from typing import List, Dict
from src.data_engine.storage import KnowledgeBase

class ResearcherAgent:
    def __init__(self, kb: KnowledgeBase, llm_client):
        self.kb = kb
        self.llm = llm_client

    def execute_research(self, plan: Dict) -> List[Dict]:
        """
        Executes the search plan, retrieving papers and generating notes.
        """
        print("Starting research phase...")
        research_notes = []
        
        for section in plan.get("sections", []):
            print(f"Researching section: {section}")
            # 1. Generate search queries
            # 2. Search KB
            # 3. Read & Summarize
            
            note = {
                "section": section,
                "content": f"Summary of findings for {section}...",
                "sources": [123, 456] # Paper IDs
            }
            research_notes.append(note)
            
        return research_notes
