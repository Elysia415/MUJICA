from typing import List, Dict
import json
from src.utils.llm import get_llm_client

class PlannerAgent:
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.llm = llm_client
        self.model = model

    def generate_plan(self, user_query: str, db_stats: Dict) -> Dict:
        """
        Generates a research plan/outline based on user query and DB stats using LLM.
        """
        print(f"Planning research for: {user_query} using {self.model}")
        
        system_prompt = """
        You are the Planner Agent for MUJICA (NeurIPS 2024 Deep Insight).
        Your goal is to accept a user research topic and a summary of available data, 
        and output a structured research plan in JSON format.
        
        The plan should include:
        1. A concise, professional Title.
        2. A list of 3-5 standard academic Sections (Introduction, Methodologies, etc.).
        3. A refined Search Query for each section to find relevant papers.
        4. Estimated number of papers to review (keep it between 5-20).
        5. (Optional) structured filters to narrow down the corpus (rating/decision/year).
        6. (Optional) retrieval budget per section (top_k_papers, top_k_chunks).
        
        Output JSON format:
        {
            "title": "Report Title",
            "global_filters": {
                "min_rating": 6.0,
                "decision_in": ["Accept"],
                "year_in": [2024]
            },
            "sections": [
                {
                    "name": "Section Name",
                    "search_query": "Key terms for this section",
                    "filters": {
                        "min_rating": 6.0
                    },
                    "top_k_papers": 20,
                    "top_k_chunks": 40
                }
            ],
            "estimated_papers": 10
        }
        """
        
        user_prompt = f"""
        User Query: "{user_query}"
        Database Stats: {db_stats}
        """
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            plan = json.loads(content)
            return plan
        except Exception as e:
            print(f"Error generating plan: {e}")
            # Fallback mock plan
            return {
                "title": "Error in Planning",
                "sections": [{"name": "Error", "search_query": "error"}],
                "estimated_papers": 0
            }

    def refine_plan(self, original_plan: Dict, user_feedback: str) -> Dict:
        """
        Updates the plan based on user feedback.
        """
        # For simplicity in this iteration, we just return the original or could add logic here.
        print("Refining plan (Mock implementation)...")
        return original_plan
