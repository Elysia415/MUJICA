from typing import List, Dict
from src.data_engine.storage import KnowledgeBase

class ResearcherAgent:
    def __init__(self, kb: KnowledgeBase, llm_client, model: str = "gpt-4o"):
        self.kb = kb
        self.llm = llm_client
        self.model = model

    def execute_research(self, plan: Dict) -> List[Dict]:
        """
        Executes the search plan, retrieving papers and generating notes.
        """
        print("Starting research phase...")
        research_notes = []
        
        for section in plan.get("sections", []):
            section_name = section.get("name")
            query = section.get("search_query")
            print(f"Researching section: {section_name} (Query: {query})")
            
            # 1. Search KB
            papers = self.kb.search_semantic(query, limit=5)
            
            if not papers:
                print(f"No papers found for {section_name}")
                continue
                
            # 2. Prepare Context for LLM
            context_text = ""
            source_ids = []
            for p in papers:
                # Use distance/score later if needed.
                context_text += f"\n[Paper ID: {p['id']}]\nTitle: {p['title']}\nAbstract: {p['abstract']}\n"
                source_ids.append(p['id'])
            
            # 3. Read & Summarize using LLM
            prompt = f"""
            You are a Researcher. Summarize the following papers specifically for the report section: "{section_name}".
            Focus on key methods, results, and insights. 
            Do NOT mention "Paper ID" in the flowing text, but ensure you capture the essence.
            
            Papers:
            {context_text}
            """
            
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful research assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                summary_content = response.choices[0].message.content
            except Exception as e:
                print(f"Error summarising: {e}")
                summary_content = "Could not generate summary."

            note = {
                "section": section_name,
                "content": summary_content,
                "sources": source_ids
            }
            research_notes.append(note)
            
        print(f"Completed research for {len(research_notes)} sections.")
        return research_notes
