import os
import shutil
from src.utils.llm import get_llm_client
from src.data_engine.storage import KnowledgeBase
from src.data_engine.loader import DataLoader
from src.planner.agent import PlannerAgent
from src.researcher.agent import ResearcherAgent
from src.writer.agent import WriterAgent
from src.verifier.agent import VerifierAgent

def test_full_chain():
    print("=== Testing Full MUJICA Chain ===")
    
    # Setup
    db_path = "data/lancedb_test_full"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    # 1. Initialize Components
    llm = get_llm_client()
    if not llm:
        print("Skipping test: No OpenAI API Key.")
        return

    kb = KnowledgeBase(db_path=db_path)
    kb.initialize_db()
    
    # 2. Ingest Sample Data
    print("\n--- Ingesting Data ---")
    sample_papers = [
        {
            "id": "p_rlhf",
            "title": "Alignment via RLHF",
            "abstract": "We improve LLM alignment using Reinforcement Learning from Human Feedback.",
            "content": "RLHF is great.",
            "authors": ["Alice"],
            "year": 2024,
            "rating": 9.0
        },
        {
            "id": "p_dpo",
            "title": "Direct Preference Optimization",
            "abstract": "We show DPO is stable and effective for alignment.",
            "content": "DPO is simpler than PPO.",
            "authors": ["Bob"],
            "year": 2024,
            "rating": 9.5
        }
    ]
    kb.ingest_data(sample_papers)
    
    # 3. Planner
    print("\n--- Planning ---")
    planner = PlannerAgent(llm)
    query = "Compare RLHF and DPO for alignment"
    plan = planner.generate_plan(query, {"count": 2})
    print(f"Plan: {plan}")
    
    # 4. Researcher
    print("\n--- Researching ---")
    researcher = ResearcherAgent(kb, llm)
    notes = researcher.execute_research(plan)
    # print(f"Notes: {notes}")
    
    # 5. Writer
    print("\n--- Writing ---")
    writer = WriterAgent(llm)
    report = writer.write_report(plan, notes)
    print("Report generated (first 200 chars):")
    print(report[:200])
    
    # 6. Verifier
    print("\n--- Verifying ---")
    verifier = VerifierAgent(llm)
    verification = verifier.verify_report(report, {})
    print(f"Verification: {verification}")
    
    # Validate
    if verification['is_valid'] or verification['score'] > 0.5:
        print("\nSUCCESS: Full chain completed and report verified (or strictly checked).")
    else:
        print("\nWARNING: Chain completed but verification failed.")

    # Cleanup
    # shutil.rmtree(db_path)

if __name__ == "__main__":
    test_full_chain()
