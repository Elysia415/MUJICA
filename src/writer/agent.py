from typing import List, Dict

class WriterAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def write_report(self, plan: Dict, research_notes: List[Dict]) -> str:
        """
        Synthesizes notes into a final markdown report.
        """
        print("Writing final report...")
        report_content = "# " + plan.get("title", "Report") + "\n\n"
        
        for note in research_notes:
            report_content += f"## {note['section']}\n\n"
            report_content += note["content"] + "\n\n"
            report_content += f"**Sources:** {note['sources']}\n\n"
            
        return report_content
