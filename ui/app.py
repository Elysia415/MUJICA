import streamlit as st
import sys
import os
import time

# Add root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.llm import get_llm_client
from src.data_engine.storage import KnowledgeBase
from src.data_engine.loader import DataLoader
from src.data_engine.fetcher import ConferenceDataFetcher
from src.planner.agent import PlannerAgent
from src.researcher.agent import ResearcherAgent
from src.writer.agent import WriterAgent
from src.verifier.agent import VerifierAgent

# --- Page Config ---
st.set_page_config(
    page_title="MUJICA: NeurIPS 2024 Deep Insight", 
    page_icon="🌌", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Custom CSS ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("ui/style.css")

# --- Initial State ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "research_notes" not in st.session_state:
    st.session_state["research_notes"] = []
if "final_report" not in st.session_state:
    st.session_state["final_report"] = ""

# --- Sidebar ---
with st.sidebar:
    st.title("🌌 MUJICA")
    st.caption("Multi-stage User-Judged Integration")
    
    st.divider()
    
    mode = st.radio("System Mode", ["Research Agent", "Data Dashboard"])
    
    st.divider()
    
    st.subheader("Model Configuration")
    
    # 1. Access Control Logic
    # Define a secret code for system access (In production, load this from secret)
    SYSTEM_ACCESS_CODE = os.getenv("MUJICA_ACCESS_CODE", "mujica2024") 
    
    auth_code = st.text_input("Access Code (Optional)", type="password", help="Enter code to use System API Key")
    
    use_system_key = False
    if auth_code == SYSTEM_ACCESS_CODE:
        use_system_key = True
        st.success("Authentication: Authorized ✅")
    elif auth_code:
        st.error("Authentication: Invalid Code ❌")
        
    # 2. User Credentials (Fallback)
    user_api_key = st.text_input("API Key", type="password", disabled=use_system_key, help="Required if no Access Code provided")
    user_base_url = st.text_input("Base URL (Optional)", placeholder="e.g. https://api.deepseek.com/v1")
    model_name = st.text_input("Model Name", value="gpt-4o", help="e.g. gpt-3.5-turbo, deepseek-chat")
    
    st.caption(f"System Status: {'Using System Key' if use_system_key else 'Using User Key'}")

# --- MAIN: Data Dashboard ---
if mode == "Data Dashboard":
    st.header("💾 Knowledge Base Management")
    
    tab1, tab2 = st.tabs(["Ingest Local Data", "OpenReview Crawler"])
    
    with tab1:
        st.subheader("Load Local Samples")
        if st.button("Load Test Dataset (Samples)"):
            with st.spinner("Ingesting sample data..."):
                kb = KnowledgeBase()
                kb.initialize_db()
                loader = DataLoader("data/raw/test_samples.json")
                if not os.path.exists("data/raw/test_samples.json"):
                    sample_papers = [
                        {"id": "p1", "title": "Self-Rewarding Language Models", "abstract": "We propose...", "rating": 9.0},
                        {"id": "p2", "title": "Direct Preference Optimization", "abstract": "DPO is stable...", "rating": 9.5}
                    ]
                    loader.save_local_data(sample_papers)
                
                data = loader.load_local_data()
                kb.ingest_data(data)
                st.success(f"Ingested {len(data)} papers into LanceDB!")

    with tab2:
        st.subheader("Crawl OpenReview (Live)")
        venue_id = st.text_input("Venue ID", "NeurIPS.cc/2024/Conference")
        limit = st.slider("Max Papers", 10, 100, 20)
        
        if st.button("Fetch & Ingest"):
            fetcher = ConferenceDataFetcher()
            with st.status("Crawling OpenReview...") as status:
                st.write("Fetching metadata...")
                papers = fetcher.fetch_papers(venue_id, limit=limit)
                st.write(f"Found {len(papers)} papers.")
                
                # Ingest
                st.write("Ingesting to Knowledge Base...")
                kb = KnowledgeBase()
                kb.initialize_db()
                kb.ingest_data(papers)
                
                status.update(label="Crawl Complete!", state="complete")
            st.success(f"Successfully added {len(papers)} new papers.")

# --- MAIN: Research Agent ---
elif mode == "Research Agent":
    st.header("🧠 Deep Insight Agent")
    
    # Split Layout
    col_chat, col_context = st.columns([0.65, 0.35], gap="large")
    
    with col_chat:
        user_query = st.chat_input("Ask a research question")
        
        # Display History
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if user_query:
            # User Msg
            st.session_state["messages"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Workflow
            
            # --- AUTH RESOLUTION LOGIC ---
            active_api_key = None
            active_base_url = None
            
            if use_system_key:
                active_api_key = os.getenv("OPENAI_API_KEY")
                # Also verify if system has a base URL override
                active_base_url = os.getenv("OPENAI_BASE_URL", None)
            else:
                active_api_key = user_api_key if user_api_key.strip() else None
                active_base_url = user_base_url if user_base_url.strip() else None
            
            # Init LLM
            llm = get_llm_client(api_key=active_api_key, base_url=active_base_url)
            
            if not llm:
                st.error("Authentication Failed. Please provide a valid Access Code or your own API Key.")
            else:
                kb = KnowledgeBase()
                kb.initialize_db()
                
                planner = PlannerAgent(llm, model=model_name)
                researcher = ResearcherAgent(kb, llm, model=model_name)
                writer = WriterAgent(llm, model=model_name)
                verifier = VerifierAgent(llm, model=model_name)
                
                # 1. Plan
                with st.status("Thinking...", expanded=True) as status:
                    st.write("Generating Research Plan...")
                    plan = planner.generate_plan(user_query, {})
                    st.json(plan, expanded=False)
                    
                    st.write("Conducting Research...")
                    notes = researcher.execute_research(plan)
                    st.session_state["research_notes"] = notes 
                    
                    st.write("Writing Report...")
                    report = writer.write_report(plan, notes)
                    
                    st.write("Verifying...")
                    verification = verifier.verify_report(report, {})
                    
                    status.update(label="Insight Generated", state="complete")
                
                # Output
                st.session_state["final_report"] = report
                st.session_state["messages"].append({"role": "assistant", "content": report})
                with st.chat_message("assistant"):
                    st.markdown(report)
                    if verification['is_valid']:
                        st.caption(f"✅ Verified (Score: {verification.get('score', 0)})")
                    else:
                        st.caption(f"⚠️ Verification Issues: {verification.get('notes')}")

    with col_context:
        st.subheader("📚 Knowledge Context")
        
        if st.session_state["research_notes"]:
            st.info("Sources referenced in this session:")
            
            all_sources = {}
            for note in st.session_state["research_notes"]:
                for pid in note.get('sources', []):
                    all_sources[pid] = {
                        "title": f"Paper {pid}", 
                        "abstract": "Abstract content unavailable in this view."
                    }
            
            for pid, info in all_sources.items():
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-title">📄 {info['title']}</div>
                    <div class="source-abstract">{info['abstract'][:100]}...</div>
                    <small style="color: grey">ID: {pid}</small>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"View PDF ({pid})", key=pid):
                    st.toast(f"Opening PDF for {pid}")
        else:
            st.markdown("*Research active... sources will appear here.*")
