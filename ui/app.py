import streamlit as st
import sys
import os

# Add root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.llm import get_llm_client
from src.data_engine.storage import KnowledgeBase
from src.planner.agent import PlannerAgent
from src.researcher.agent import ResearcherAgent
from src.writer.agent import WriterAgent
from src.verifier.agent import VerifierAgent

def main():
    st.set_page_config(page_title="MUJICA: NeurIPS 2024 Agent", page_icon="📝", layout="wide")
    
    st.title("🤖 MUJICA: NeurIPS 2024 Deep Insight AI")
    st.markdown("### Multi-stage User-Judged Integration & Corroboration Architecture")

    # Sidebar inputs
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
        st.info("Knowledge Base Status: 🟢 Connected (Mocked)")

    user_query = st.chat_input("Enter your research topic (e.g., 'Compare DPO vs PPO in LLM Alignment')")

    if user_query:
        st.session_state["messages"] = st.session_state.get("messages", [])
        st.session_state["messages"].append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.write(user_query)

        # Initialize Agents
        llm = get_llm_client()
        if not llm:
            st.error("Please provide OpenAI API Key.")
            return

        kb = KnowledgeBase()
        kb.initialize_db()
        
        planner = PlannerAgent(llm)
        researcher = ResearcherAgent(kb, llm)
        writer = WriterAgent(llm)
        verifier = VerifierAgent(llm)

        # 1. PLANNER
        with st.status("Phase 1: Planning...", expanded=True) as status:
            st.write("Analyzing query intent...")
            db_stats = {"count": 158} # Mock stats
            plan = planner.generate_plan(user_query, db_stats)
            st.json(plan)
            status.update(label="Phase 1: Planning Complete!", state="complete", expanded=False)

        # 2. RESEARCHER
        with st.status("Phase 2: Researching...", expanded=True) as status:
            st.write(f"Searching for {plan.get('estimated_papers')} relevant papers...")
            notes = researcher.execute_research(plan)
            for note in notes:
                st.text(f"Processed section: {note['section']}")
            status.update(label="Phase 2: Research Complete!", state="complete", expanded=False)

        # 3. WRITER
        with st.status("Phase 3: Writing Report...", expanded=True) as status:
            final_report = writer.write_report(plan, notes)
            status.update(label="Phase 3: Writing Complete!", state="complete", expanded=False)

        # 4. VERIFIER
        with st.status("Phase 4: Verifying...", expanded=True) as status:
            verification = verifier.verify_report(final_report, {})
            if verification['is_valid']:
                st.success(f"Verification Passed! Score: {verification['score']}")
            else:
                st.warning("Issues found during verification.")
            status.update(label="Phase 4: Verification Complete!", state="complete", expanded=False)

        # Display Final Report
        st.divider()
        st.markdown(final_report)
        st.markdown("#### Citations")
        st.info("Click references to see Paper Cards (Mocked)")

if __name__ == "__main__":
    main()
