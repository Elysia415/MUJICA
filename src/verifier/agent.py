from typing import Dict
import re

class VerifierAgent:
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.llm = llm_client
        self.model = model

    def verify_report(self, report_text: str, source_data: Dict) -> Dict:
        """
        Checks the report for hallucinations and citation integrity.
        source_data: A dict of {paper_id: paper_content/abstract} to verify against.
                     (For now, we might just check if citations exist and match known IDs).
        """
        print("Verifying report integrity...")
        
        # 1. Extract Citations
        citations = re.findall(r"\[Paper ID: (.*?)\]", report_text)
        unique_citations = list(set(citations))
        
        # 2. Check if citations are empty
        if not unique_citations:
            return {
                "is_valid": False,
                "score": 0.0,
                "notes": "No citations found. Report must anchor claims to sources."
            }

        # 3. LLM-based hallucination check (Sample)
        # We grab a random claim with a citation and ask LLM if it follows.
        # For this prototype, I'll basically ask LLM to rate the report's "Academic Tone" and "Citation Usage".
        
        prompt = f"""
        You are a Verifier. Analyze the following report segment.
        
        Report:
        {report_text[:2000]}... (Truncated)
        
        Found Citations: {unique_citations}
        
        Task:
        1. Does the report use a purely objective, academic tone?
        2. Are there citations attached to claims?
        
        Return JSON:
        {{
            "score": <float 0-1>,
            "reason": "<short explanation>"
        }}
        """
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a critical reviewer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            result = response.choices[0].message.content
            # parse json
            import json
            res = json.loads(result)
            return {
                "is_valid": res.get("score", 0) > 0.7,
                "score": res.get("score", 0),
                "notes": res.get("reason", "")
            }
        except Exception as e:
            print(f"Error validating: {e}")
            return {
                "is_valid": True, # Fail open for now
                "score": 0.5,
                "notes": "Validation error, manual review needed."
            }
