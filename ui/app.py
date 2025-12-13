from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

# 确保项目根目录在 sys.path，方便 `import src.*`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import load_env
from src.utils.llm import get_llm_client
from src.data_engine.storage import KnowledgeBase
from src.data_engine.loader import DataLoader
from src.data_engine.fetcher import ConferenceDataFetcher
from src.data_engine.ingestor import OpenReviewIngestor
from src.planner.agent import PlannerAgent
from src.researcher.agent import ResearcherAgent
from src.writer.agent import WriterAgent
from src.verifier.agent import VerifierAgent


def _ensure_streamlit_context() -> bool:
    """
    在非 `streamlit run` 场景下避免 session_state 报错（例如误用 `python ui/app.py`）。
    """
    try:
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        # Streamlit 内部 API 变化时，尽量保持可运行（最坏情况与原来一致）
        return True


def _local_css(css_path: Path) -> None:
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _render_data_dashboard() -> None:
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
                        {"id": "p2", "title": "Direct Preference Optimization", "abstract": "DPO is stable...", "rating": 9.5},
                    ]
                    loader.save_local_data(sample_papers)

                data = loader.load_local_data()
                kb.ingest_data(data)
                st.success(f"Ingested {len(data)} papers into LanceDB!")

    with tab2:
        st.subheader("Crawl OpenReview (Live)")
        venue_id = st.text_input("Venue ID", "NeurIPS.cc/2024/Conference")
        limit = st.slider("Max Papers", 10, 100, 20)
        download_pdfs = st.checkbox("Download PDFs", value=True)
        parse_pdfs = st.checkbox("Parse PDFs to Full Text", value=True, disabled=not download_pdfs)
        max_pages = st.slider("Max PDF pages to parse", 1, 50, 12, disabled=not parse_pdfs)

        if st.button("Fetch & Ingest"):
            kb = KnowledgeBase()
            kb.initialize_db()
            ingestor = OpenReviewIngestor(kb, fetcher=ConferenceDataFetcher(output_dir="data/raw"))

            with st.status("Crawling OpenReview...", expanded=True) as status:
                st.write("Fetching / Downloading / Parsing / Indexing ...")
                papers = ingestor.ingest_venue(
                    venue_id=venue_id,
                    limit=limit,
                    download_pdfs=download_pdfs,
                    parse_pdfs=parse_pdfs,
                    max_pdf_pages=max_pages if parse_pdfs else None,
                    max_downloads=limit if download_pdfs else None,
                )
                status.update(label="Crawl Complete!", state="complete")

            st.success(f"Successfully ingested {len(papers)} papers.")


def _render_research_agent(*, use_system_key: bool, user_api_key: str, user_base_url: str, model_name: str) -> None:
    st.header("🧠 Deep Insight Agent")

    col_chat, col_context = st.columns([0.65, 0.35], gap="large")

    with col_chat:
        # 初始化/连接知识库（不依赖 LLM）
        kb = KnowledgeBase()
        kb.initialize_db()

        # 展示历史对话（只放用户问题/简短状态，不把整篇报告塞进聊天气泡）
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Ask a research question")

        # 新问题：生成 plan（待用户批准）
        if user_query:
            st.session_state["messages"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # 清空上一次结果
            st.session_state["research_notes"] = []
            st.session_state["final_report"] = ""
            st.session_state["verification_result"] = None
            st.session_state["plan_approved"] = False
            st.session_state["pending_plan"] = None
            st.session_state["plan_editor_text"] = ""

            # 解析认证信息 -> 初始化 LLM
            active_api_key = os.getenv("OPENAI_API_KEY") if use_system_key else (user_api_key.strip() or None)
            active_base_url = os.getenv("OPENAI_BASE_URL", None) if use_system_key else (user_base_url.strip() or None)
            llm = get_llm_client(api_key=active_api_key, base_url=active_base_url)
            if not llm:
                st.error("Authentication Failed. Please provide a valid Access Code or your own API Key.")
            else:
                # DB stats（给 planner 用）
                df = kb.search_structured()
                stats = {"count": int(len(df))}
                if hasattr(df, "empty") and not df.empty:
                    try:
                        stats["avg_rating"] = float(df["rating"].dropna().mean()) if "rating" in df.columns else None
                    except Exception:
                        stats["avg_rating"] = None
                    try:
                        if "decision" in df.columns:
                            stats["decision_counts"] = (
                                df["decision"].fillna("UNKNOWN").value_counts().head(10).to_dict()
                            )
                    except Exception:
                        pass

                planner = PlannerAgent(llm, model=model_name)
                with st.status("Planning...", expanded=True) as status:
                    st.write("Generating Research Plan...")
                    plan = planner.generate_plan(user_query, stats)
                    st.session_state["pending_plan"] = plan
                    st.session_state["plan_editor_text"] = json.dumps(plan, ensure_ascii=False, indent=2)
                    status.update(label="Plan Generated (Waiting for Approval)", state="complete")

        # 计划审核/编辑/批准
        if st.session_state.get("pending_plan") and not st.session_state.get("plan_approved"):
            st.subheader("Step 1 · Review & Approve Plan")
            plan_text = st.text_area(
                "Plan JSON (editable)",
                key="plan_editor_text",
                height=320,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Apply Edits"):
                    try:
                        st.session_state["pending_plan"] = json.loads(plan_text)
                        st.success("Plan updated.")
                    except Exception as e:
                        st.error(f"Plan JSON parse error: {e}")

            with col_b:
                if st.button("Approve & Run"):
                    # 再次解析认证信息 -> 初始化 LLM
                    active_api_key = os.getenv("OPENAI_API_KEY") if use_system_key else (user_api_key.strip() or None)
                    active_base_url = os.getenv("OPENAI_BASE_URL", None) if use_system_key else (user_base_url.strip() or None)
                    llm = get_llm_client(api_key=active_api_key, base_url=active_base_url)
                    if not llm:
                        st.error("Authentication Failed. Please provide a valid Access Code or your own API Key.")
                    else:
                        try:
                            plan = json.loads(plan_text)
                        except Exception as e:
                            st.error(f"Plan JSON parse error: {e}")
                            plan = None

                        if plan:
                            st.session_state["plan_approved"] = True

                            researcher = ResearcherAgent(kb, llm, model=model_name)
                            writer = WriterAgent(llm, model=model_name)
                            verifier = VerifierAgent(llm, model=model_name)

                            with st.status("Running...", expanded=True) as status:
                                st.write("Conducting Research...")
                                notes = researcher.execute_research(plan)
                                st.session_state["research_notes"] = notes

                                st.write("Writing Report...")
                                report = writer.write_report(plan, notes)
                                st.session_state["final_report"] = report

                                st.write("Verifying (Claim-level NLI)...")
                                chunk_map = {}
                                for n in notes:
                                    for e in (n.get("evidence") or []):
                                        cid = e.get("chunk_id")
                                        txt = e.get("text")
                                        if cid and txt and cid not in chunk_map:
                                            chunk_map[cid] = txt

                                verification = verifier.verify_report(report, {"chunks": chunk_map})
                                st.session_state["verification_result"] = verification

                                status.update(label="Completed", state="complete")

                            # 给聊天区一个简短回执（不贴整篇报告）
                            v = st.session_state.get("verification_result") or {}
                            st.session_state["messages"].append(
                                {
                                    "role": "assistant",
                                    "content": f"报告已生成。核查：valid={v.get('is_valid')}, score={v.get('score')}.（详见右侧溯源/核查面板）",
                                }
                            )

        # 输出最终报告（左栏）
        if st.session_state.get("final_report"):
            st.divider()
            st.subheader("Final Report")
            st.markdown(st.session_state["final_report"])

            v = st.session_state.get("verification_result")
            if isinstance(v, dict) and v:
                st.caption(f"Verification: valid={v.get('is_valid')} · score={v.get('score')} · {v.get('notes')}")

    with col_context:
        st.subheader("🔎 Traceability")

        tab_evi, tab_ver = st.tabs(["Evidence", "Verification"])

        with tab_evi:
            notes = st.session_state.get("research_notes") or []
            if not notes:
                st.markdown("*No evidence yet. Ingest data and run a query.*")
            else:
                for note in notes:
                    section_name = note.get("section", "Section")
                    with st.expander(f"📌 {section_name}", expanded=False):
                        if note.get("filters"):
                            st.caption(f"Filters: {json.dumps(note.get('filters'), ensure_ascii=False)}")

                        # 展示 key points（带 citations）
                        if note.get("key_points"):
                            st.markdown("**Key Points**")
                            st.json(note.get("key_points"), expanded=False)

                        evidence = note.get("evidence") or []
                        if not evidence:
                            st.markdown("*No evidence snippets for this section.*")
                        else:
                            for e in evidence:
                                pid = e.get("paper_id")
                                title = e.get("title", "")
                                cid = e.get("chunk_id")
                                src = e.get("source")
                                st.markdown(f"**{title}**  \n`paper_id={pid}` · `chunk_id={cid}` · `source={src}`")
                                st.code((e.get("text") or "")[:1200])

        with tab_ver:
            v = st.session_state.get("verification_result")
            if not isinstance(v, dict) or not v:
                st.markdown("*No verification yet.*")
            else:
                st.caption(f"valid={v.get('is_valid')} · score={v.get('score')} · {v.get('notes')}")
                evals = v.get("evaluations") or []
                if evals:
                    try:
                        import pandas as pd

                        st.dataframe(pd.DataFrame(evals), use_container_width=True)
                    except Exception:
                        st.json(evals, expanded=False)
                else:
                    st.json(v, expanded=False)


def main() -> None:
    load_env()

    if not _ensure_streamlit_context():
        print("这是一个 Streamlit 应用，请使用：streamlit run ui/app.py")
        return

    st.set_page_config(
        page_title="MUJICA: NeurIPS 2024 Deep Insight",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _local_css(Path(__file__).with_name("style.css"))

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("research_notes", [])
    st.session_state.setdefault("final_report", "")
    st.session_state.setdefault("pending_plan", None)
    st.session_state.setdefault("plan_editor_text", "")
    st.session_state.setdefault("plan_approved", False)
    st.session_state.setdefault("verification_result", None)

    with st.sidebar:
        st.title("🌌 MUJICA")
        st.caption("Multi-stage User-Judged Integration")

        st.divider()
        mode = st.radio("System Mode", ["Research Agent", "Data Dashboard"])

        st.divider()
        st.subheader("Model Configuration")

        SYSTEM_ACCESS_CODE = os.getenv("MUJICA_ACCESS_CODE", "mujica2024")
        auth_code = st.text_input(
            "Access Code (Optional)",
            type="password",
            help="Enter code to use System API Key",
        )

        use_system_key = False
        if auth_code == SYSTEM_ACCESS_CODE:
            use_system_key = True
            st.success("Authentication: Authorized ✅")
        elif auth_code:
            st.error("Authentication: Invalid Code ❌")

        user_api_key = st.text_input(
            "API Key",
            type="password",
            disabled=use_system_key,
            help="Required if no Access Code provided",
        )
        user_base_url = st.text_input("Base URL (Optional)", placeholder="e.g. https://api.deepseek.com/v1")
        model_name = st.text_input(
            "Model Name",
            value=os.getenv("MUJICA_DEFAULT_MODEL", "gpt-4o"),
            help="e.g. gpt-3.5-turbo, deepseek-chat",
        )

        st.caption(f"System Status: {'Using System Key' if use_system_key else 'Using User Key'}")

    if mode == "Data Dashboard":
        _render_data_dashboard()
    else:
        _render_research_agent(
            use_system_key=use_system_key,
            user_api_key=user_api_key,
            user_base_url=user_base_url,
            model_name=model_name,
        )


if __name__ == "__main__":
    main()
