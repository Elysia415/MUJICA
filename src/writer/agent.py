from typing import List, Dict

class WriterAgent:
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.llm = llm_client
        self.model = model

    def write_report(self, plan: Dict, research_notes: List[Dict]) -> str:
        """
        Synthesizes notes into a final markdown report.
        """
        print("Writing final report...")
        
        # Prepare context from research notes
        notes_text = ""
        for note in research_notes:
            notes_text += f"\n--- Section: {note['section']} ---\n"
            notes_text += f"{note['content']}\n"
            notes_text += f"Available Source IDs: {note['sources']}\n"

        system_prompt = """
        You are the Writer Agent for MUJICA. 
        Your task is to write a comprehensive, academic-style report based on the provided research notes.
        
        Rules:
        1. Follow the structure provided in the notes.
        2. You must STRICTLY base your claims on the provided notes.
        3. Citation Format: You MUST include citations like [Paper ID: <id>] at the end of relevant sentences.
        4. If a note says "Available Source IDs: ['p1']", you can only cite p1.
        5. Do not invent information.
        """
        
        user_prompt = f"""
        Report Title: {plan.get('title', 'Research Report')}
        
        Research Notes:
        {notes_text}
        
        Please generate the full Markdown report.
        """
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error writing report: {e}")
            return "Error generating report."
