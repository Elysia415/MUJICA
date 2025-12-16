from __future__ import annotations

import inspect
import json
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# 确保项目根目录在 sys.path，方便 `import src.*`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import load_env
from src.utils.llm import get_llm_client, get_embedding
from src.data_engine.storage import KnowledgeBase
from src.data_engine.loader import DataLoader
from src.data_engine.fetcher import ConferenceDataFetcher
from src.data_engine.ingestor import OpenReviewIngestor
from src.planner.agent import PlannerAgent
from src.researcher.agent import ResearcherAgent
from src.writer.agent import WriterAgent
from src.verifier.agent import VerifierAgent
from src.utils.cancel import MujicaCancelled
from src.utils.chat_history import (
    delete_conversation,
    list_conversations,
    load_conversation,
    new_conversation_id,
    rename_conversation,
    save_conversation,
)


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


def _rerun() -> None:
    """
    Streamlit rerun 兼容：
    - 新版：st.rerun()
    - 旧版（如 1.26）：st.experimental_rerun()
    """
    if hasattr(st, "rerun"):
        st.rerun()
        return
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
        return
    # 极端兜底：不 rerun（避免直接崩）
    return


def _width_kwargs(fn, *, stretch: bool = True) -> dict:
    """
    Streamlit 参数兼容：
    - 新版（>=1.52）：推荐使用 width='stretch'/'content'
    - 旧版：使用 use_container_width=True/False
    """
    try:
        params = inspect.signature(fn).parameters
        if "width" in params:
            return {"width": "stretch" if stretch else "content"}
        if "use_container_width" in params:
            return {"use_container_width": bool(stretch)}
    except Exception:
        pass
    # 兜底：旧版大概率支持 use_container_width
    return {"use_container_width": bool(stretch)}


def _get_query_params() -> dict:
    # Streamlit 版本兼容（1.26: experimental_get_query_params；新版: st.query_params）
    try:
        if hasattr(st, "query_params"):
            qp = st.query_params  # type: ignore[attr-defined]
            out = {}
            for k in qp.keys():
                try:
                    out[k] = qp.get_all(k)  # type: ignore[attr-defined]
                except Exception:
                    v = qp.get(k)  # type: ignore[attr-defined]
                    out[k] = v if isinstance(v, list) else [v] if v is not None else []
            return out
        if hasattr(st, "experimental_get_query_params"):
            return st.experimental_get_query_params()  # type: ignore[attr-defined]
    except Exception:
        pass
    return {}


def _set_query_params(**kwargs) -> None:
    # kwargs: key -> str
    try:
        if hasattr(st, "query_params"):
            qp = st.query_params  # type: ignore[attr-defined]
            qp.clear()  # type: ignore[attr-defined]
            for k, v in kwargs.items():
                if v is None:
                    continue
                qp[str(k)] = str(v)  # type: ignore[attr-defined]
            return
        if hasattr(st, "experimental_set_query_params"):
            st.experimental_set_query_params(**{k: v for k, v in kwargs.items() if v is not None})  # type: ignore[attr-defined]
    except Exception:
        return


def _reset_workspace_state(*, cancel_running_job: bool = True) -> None:
    # 可选：离开时尝试停止后台任务
    if cancel_running_job:
        job = st.session_state.get("research_job")
        try:
            if isinstance(job, _ResearchJob) and job.status == "running":
                job.cancel_event.set()
        except Exception:
            pass
        pj = st.session_state.get("plan_job")
        try:
            if isinstance(pj, _PlanJob) and pj.status == "running":
                pj.cancel_event.set()
        except Exception:
            pass

    st.session_state["messages"] = []
    st.session_state["research_notes"] = []
    st.session_state["final_report"] = ""
    st.session_state["report_ref_ctx"] = None
    st.session_state["writer_stats"] = None
    st.session_state["pending_plan"] = None
    st.session_state["plan_editor_text"] = ""
    st.session_state["plan_approved"] = False
    st.session_state["verification_result"] = None
    # 当前对话标题（用于历史保存；避免把旧标题写入新对话）
    st.session_state["conversation_title"] = ""
    st.session_state.pop("pending_user_query", None)
    st.session_state.pop("plan_run_requested", None)


def _history_snapshot() -> Dict[str, Any]:
    """
    对话历史快照（脱敏！绝不保存 API Key/Access Code）。
    """
    return {
        "created_ts": float(st.session_state.get("history_created_ts") or time.time()),
        # 若用户手动重命名，这里必须带上 title，否则自动保存会被“首条用户消息”重置标题
        "title": str(st.session_state.get("conversation_title") or "").strip() or None,
        "messages": list(st.session_state.get("messages") or []),
        "pending_plan": st.session_state.get("pending_plan"),
        "plan_editor_text": str(st.session_state.get("plan_editor_text") or ""),
        "plan_approved": bool(st.session_state.get("plan_approved")),
        "research_notes": st.session_state.get("research_notes") or [],
        "final_report": str(st.session_state.get("final_report") or ""),
        "verification_result": st.session_state.get("verification_result"),
        "writer_stats": st.session_state.get("writer_stats"),
        "report_ref_ctx": st.session_state.get("report_ref_ctx"),
        "system_mode": str(st.session_state.get("system_mode") or "research"),
        "ui_theme": str(st.session_state.get("ui_theme") or "light"),
    }


# ---------------------------
# 后台研究任务（支持停止）
# ---------------------------


@dataclass
class _ResearchJob:
    job_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    status: str = "running"  # running|done|cancelled|error
    stage: str = "init"
    message: str = ""
    progress: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_trace: Optional[str] = None
    started_ts: float = field(default_factory=lambda: time.time())
    finished_ts: Optional[float] = None
    thread: Optional[threading.Thread] = None


def _job_update(job: Any, **kwargs: Any) -> None:
    with job.lock:
        for k, v in kwargs.items():
            setattr(job, k, v)
        # 轻量记录最后一次变更时间（用于 UI 展示）
        job.progress["_ts"] = time.time()


def _job_emit_progress(job: Any, *, kind: str, payload: Dict[str, Any]) -> None:
    """
    线程安全地写入进度信息（注意：不要在 worker 线程里调用任何 st.*）。
    """
    with job.lock:
        job.progress[kind] = payload
        job.progress["_ts"] = time.time()


@dataclass
class _PlanJob:
    job_id: str
    query: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    status: str = "running"  # running|done|cancelled|error
    stage: str = "init"
    message: str = ""
    progress: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_trace: Optional[str] = None
    started_ts: float = field(default_factory=lambda: time.time())
    finished_ts: Optional[float] = None
    thread: Optional[threading.Thread] = None


def _run_plan_job(
    job: _PlanJob,
    *,
    user_query: str,
    stats: Dict[str, Any],
    chat_api_key: Optional[str],
    chat_base_url: Optional[str],
    model_name: str,
) -> None:
    """
    后台线程入口：Plan（生成研究计划）。
    注意：该函数运行在后台线程中，严禁调用 Streamlit API（st.*）。
    """
    try:
        _job_update(job, status="running", stage="init", message="初始化规划（连接模型）...")
        llm = get_llm_client(api_key=chat_api_key, base_url=chat_base_url, allow_env_fallback=False)
        if llm is None:
            raise RuntimeError("Authentication Failed: missing/invalid API key.")

        planner = PlannerAgent(llm, model=model_name)
        _job_update(job, stage="planning", message="生成研究计划（Plan）...")
        plan = planner.generate_plan(user_query, stats, cancel_event=job.cancel_event)
        _job_update(job, result={"plan": plan}, status="done", stage="done", message="规划完成 ✅", finished_ts=time.time())
    except MujicaCancelled as e:
        _job_update(job, status="cancelled", stage="cancelled", message="已停止规划", error=str(e), finished_ts=time.time())
    except Exception as e:
        _job_update(
            job,
            status="error",
            stage="error",
            message="规划失败 ❌",
            error=str(e),
            error_trace=traceback.format_exc(),
            finished_ts=time.time(),
        )


def _run_research_job(
    job: _ResearchJob,
    *,
    plan: Dict[str, Any],
    chat_api_key: Optional[str],
    chat_base_url: Optional[str],
    model_name: str,
    embedding_model: str,
    embedding_api_key: Optional[str],
    embedding_base_url: Optional[str],
) -> None:
    """
    后台线程入口：Research -> Write -> Verify。

    约束：该函数运行在后台线程中，严禁调用 Streamlit API（st.*）。
    """
    try:
        _job_update(job, status="running", stage="init", message="初始化（连接知识库/模型）...")

        # 每个 job 自己创建 KB/连接，避免与 UI 线程共享连接对象
        kb = KnowledgeBase(
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
        )
        kb.initialize_db()

        llm = get_llm_client(
            api_key=chat_api_key,
            base_url=chat_base_url,
            allow_env_fallback=False,  # 门禁：禁止偷读 env
        )
        if llm is None:
            raise RuntimeError("Authentication Failed: missing/invalid API key.")

        researcher = ResearcherAgent(kb, llm, model=model_name)
        writer = WriterAgent(llm, model=model_name)
        verifier = VerifierAgent(llm, model=model_name)

        # ---------- Research ----------
        _job_update(job, stage="research", message="检索证据（Research）...")

        def _on_research_progress(payload: Dict[str, Any]) -> None:
            if not isinstance(payload, dict):
                return
            _job_emit_progress(job, kind="research", payload=payload)
            # 让 UI 能看到更友好的当前阶段描述
            stg = payload.get("stage")
            if stg == "research_section":
                sec = payload.get("section") or ""
                q = payload.get("query") or ""
                _job_update(job, stage="research", message=f"检索中：{sec}（{q}）")
            elif stg == "research_section_done":
                sec = payload.get("section") or ""
                _job_update(job, stage="research", message=f"已完成章节：{sec}")

        notes = researcher.execute_research(plan, on_progress=_on_research_progress, cancel_event=job.cancel_event)
        _job_update(job, result={**job.result, "research_notes": notes})

        # ---------- Write ----------
        _job_update(job, stage="write", message="循证写作（Write）...")

        def _on_write_progress(payload: Dict[str, Any]) -> None:
            if not isinstance(payload, dict):
                return
            _job_emit_progress(job, kind="write", payload=payload)
            stg = payload.get("stage")
            if stg == "write_refs_built":
                _job_update(job, stage="write", message=f"写作准备：refs={payload.get('refs_total')}")
            elif stg == "write_payload_built":
                _job_update(
                    job,
                    stage="write",
                    message=(
                        f"写作准备：sections={payload.get('sections')} · evidence={payload.get('evidence_snippets')} · refs={payload.get('allowed_refs_total')}"
                    ),
                )
            elif stg == "write_llm_call":
                _job_update(job, stage="write", message=f"LLM 生成中：model={payload.get('model')}")
            elif stg == "write_done":
                _job_update(job, stage="write", message="写作完成。")
            elif stg == "write_error":
                _job_update(job, stage="write", message=f"写作失败：{payload.get('error')}")

        report, ref_ctx = writer.write_report(
            plan,
            notes,
            on_progress=_on_write_progress,
            cancel_event=job.cancel_event,
        )

        writer_stats = None
        try:
            writer_stats = (ref_ctx or {}).get("writer_stats")
        except Exception:
            writer_stats = None

        _job_update(
            job,
            result={
                **job.result,
                "final_report": report,
                "report_ref_ctx": ref_ctx,
                "writer_stats": writer_stats,
            },
        )

        # ---------- Verify ----------
        _job_update(job, stage="verify", message="逐句核查（Verify）...")

        chunk_map: Dict[str, str] = {}
        for n in notes:
            for e in (n.get("evidence") or []):
                cid = e.get("chunk_id")
                txt = e.get("text")
                if cid and txt and cid not in chunk_map:
                    chunk_map[cid] = txt

        ref_map: Dict[str, Any] = {}
        try:
            ref_map = (ref_ctx or {}).get("ref_map") or {}
        except Exception:
            ref_map = {}

        verification = verifier.verify_report(
            report,
            {"chunks": chunk_map, "ref_map": ref_map},
            cancel_event=job.cancel_event,
        )
        _job_update(job, result={**job.result, "verification_result": verification})

        _job_update(job, status="done", stage="done", message="完成 ✅", finished_ts=time.time())
    except MujicaCancelled as e:
        _job_update(job, status="cancelled", stage="cancelled", message="已停止（取消成功）", error=str(e), finished_ts=time.time())
    except Exception as e:
        _job_update(
            job,
            status="error",
            stage="error",
            message="运行失败 ❌",
            error=str(e),
            error_trace=traceback.format_exc(),
            finished_ts=time.time(),
        )


# ---------------------------
# 后台入库任务（支持停止 + UI 不中断）
# ---------------------------

@dataclass
class _IngestJob:
    """数据入库后台任务（下载/解析/Embedding）"""
    job_id: str
    venue_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    status: str = "running"  # running|done|cancelled|error
    stage: str = "init"
    message: str = ""
    progress: Dict[str, Any] = field(default_factory=dict)
    result: List[Dict[str, Any]] = field(default_factory=list)  # papers list
    error: Optional[str] = None
    error_trace: Optional[str] = None
    started_ts: float = field(default_factory=lambda: time.time())
    finished_ts: Optional[float] = None
    thread: Optional[threading.Thread] = None


def _run_ingest_job(
    job: _IngestJob,
    *,
    ingestor: Any,
    venue_id: str,
    limit: Optional[int],
    accepted_only: bool,
    presentation_in: Optional[List[str]],
    skip_existing: bool,
    download_pdfs: bool,
    parse_pdfs: bool,
    max_pdf_pages: Optional[int],
    max_downloads: Optional[int],
) -> None:
    """
    后台线程入口：数据入库（Fetch -> Download -> Parse -> Embed）。
    注意：该函数运行在后台线程中，严禁调用 Streamlit API（st.*）。
    """
    try:
        _job_update(job, stage="ingest", message="正在入库...")

        def _on_progress(payload: Dict[str, Any]) -> None:
            # 不调用 st.* 只更新 job.progress
            if job.cancel_event.is_set():
                # 抛出异常中止 ingestor（如果支持的话）
                raise MujicaCancelled("用户取消")
            if isinstance(payload, dict):
                stage = payload.get("stage", "unknown")
                _job_emit_progress(job, kind=stage, payload=payload)
                # 也更新 message 方便 UI 显示
                cur = payload.get("current", 0)
                tot = payload.get("total", 0)
                if stage == "fetch_papers":
                    _job_update(job, message=f"抓取元数据 {cur}/{tot}")
                elif stage == "download_pdf":
                    _job_update(job, message=f"下载 PDF {cur}/{tot}")
                elif stage == "parse_pdf":
                    _job_update(job, message=f"解析 PDF {cur}/{tot}")
                elif stage in {"embed_papers", "embed_chunks"}:
                    _job_update(job, message=f"Embedding {cur}/{tot}")

        papers = ingestor.ingest_venue(
            venue_id=venue_id,
            limit=limit,
            accepted_only=accepted_only,
            presentation_in=presentation_in,
            skip_existing=skip_existing,
            download_pdfs=download_pdfs,
            parse_pdfs=parse_pdfs,
            max_pdf_pages=max_pdf_pages,
            max_downloads=max_downloads,
            on_progress=_on_progress,
        )

        _job_update(
            job,
            status="done",
            stage="done",
            message=f"入库完成 ✅ 共 {len(papers)} 篇论文",
            result=papers,
            finished_ts=time.time(),
        )

    except MujicaCancelled:
        _job_update(
            job,
            status="cancelled",
            stage="cancelled",
            message="已取消",
            finished_ts=time.time(),
        )
    except Exception as e:
        _job_update(
            job,
            status="error",
            stage="error",
            message="入库失败 ❌",
            error=str(e),
            error_trace=traceback.format_exc(),
            finished_ts=time.time(),
        )

def _apply_theme_vars(theme: str) -> None:
    """
    通过 CSS 变量实现主题切换。
    注意：变量必须在页面渲染早期注入，且每次 rerun 都注入一次，避免旧主题残留。
    """
    theme = (theme or "").strip().lower()
    if theme in {"dark", "深色"}:
        vars_css = """
            /* Ave Mujica Theme (Dark/Gothic/Elegant) */
            /* Background: Deep Abyss Black */
            --bg: #050505;
            
            /* Background Glows: Blood Red & Phantom Purple */
            --bg-glow-1: rgba(139, 0, 50, 0.35); 
            --bg-glow-2: rgba(75, 0, 130, 0.25); 

            /* Panels: Dark tempered glass (more transparent) */
            --panel: rgba(18, 18, 24, 0.70);
            --panel-2: rgba(26, 26, 32, 0.65);

            /* Text: Silver/White for contrast against dark bg */
            --text: #eaeaea;
            --muted: #999999;

            /* Borders: Bright Antique Gold (popping against glass) */
            --border: rgba(197, 160, 89, 0.6); 
            
            /* Shadows: Heavy and dark for depth */
            --shadow: 0 20px 50px rgba(0, 0, 0, 0.85);

            --sidebar-bg: rgba(10, 10, 12, 0.85);
            
            /* Inputs: More transparent glass */
            --input-bg: rgba(0, 0, 0, 0.35);
            --code-bg: rgba(0, 0, 0, 0.4);

            /* Accents: Crimson Red & Gold */
            --accent: #8a002b;      /* Deep Crimson */
            --accent-2: #c5a059;    /* Antique Gold */
            
            --accent-hover: #a30033;
            --accent-2-hover: #d4af37;
            
            --accent-shadow: rgba(139, 0, 50, 0.5);
            --accent-shadow-hover: rgba(212, 175, 55, 0.3);
            
            --accent-focus: rgba(197, 160, 89, 0.6);
            --accent-focus-shadow: rgba(139, 0, 50, 0.2);

            /* Button Specifics (Dark Mode) */
            --btn-primary-bg: linear-gradient(145deg, #8a002b 0%, #4a0016 100%);
            --btn-primary-text: #ffffff;
            --btn-primary-border: rgba(197, 160, 89, 0.6);

            --btn-secondary-bg: rgba(255, 255, 255, 0.05);
            --btn-secondary-text: #eaeaea;
            --btn-secondary-border: rgba(197, 160, 89, 0.4);

            /* Hover Variables (Dark Mode) */
            --btn-hover-bg: linear-gradient(145deg, #a30033 0%, #5e001f 100%);
            --btn-hover-filter: brightness(1.1);
            --btn-hover-transform: translateY(-2px);
            --btn-hover-shadow: 0 0 20px rgba(139, 0, 50, 0.6);
            --btn-hover-border: rgba(212, 175, 55, 0.8);
            --btn-hover-color: #ffffff;

            --btn-sec-hover-bg: rgba(255, 255, 255, 0.1);
            --btn-sec-hover-border: rgba(212, 175, 55, 0.8);
            --btn-sec-hover-color: #ffffff;
        """
    else:
        # 默认：简明模式（仿截图风格 - 干净、纸张感、暖白）
        vars_css = """
            --bg: #ffffff;
            --bg-glow-1: transparent;
            --bg-glow-2: transparent;
            
            --panel: #ffffff;
            --panel-2: #fcfcfc;     /* Almost white */
            --text: #202124;        /* Google Sans Black / Deep Grey */
            --muted: #5f6368;       /* Secondary Text */
            --border: #dadce0;      /* Subtle border */

            /* Concise Accents (Lighter Silver) */
            --accent: #bdbdbd;      /* Lighter Grey */
            --accent-2: #757575;    /* Material Grey 600 */
            
            --accent-hover: #9e9e9e;
            --accent-2-hover: #616161; 
            
            --accent-shadow: rgba(0, 0, 0, 0.02);
            --accent-shadow-hover: rgba(0, 0, 0, 0.05);
            
            --accent-focus: #f5f5f5;
            --accent-focus-shadow: rgba(0, 0, 0, 0.02);
            
            --shadow: none;
            
            --sidebar-bg: #f8f9fa;
            
            --input-bg: #ffffff;
            --code-bg: #f1f3f4;

            /* Button Specifics (Lighter Gray) */
            --btn-primary-bg: #cccccc;
            --btn-primary-text: #ffffff;
            --btn-primary-border: #cccccc;

            --btn-secondary-bg: #ffffff;
            --btn-secondary-text: #999999;
            --btn-secondary-border: #eeeeee;

            /* Hover Variables (Light Mode) */
            --btn-hover-bg: #e0e0e0;
            --btn-hover-filter: brightness(1.08);
            --btn-hover-transform: translateY(-1px);
            --btn-hover-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
            --btn-hover-border: #d8d8d8;
            --btn-hover-color: #ffffff;

            --btn-sec-hover-bg: #f8f8f8;
            --btn-sec-hover-border: #e0e0e0;
            --btn-sec-hover-color: #666666;
        """

    st.markdown(f"<style>:root{{{vars_css}}}</style>", unsafe_allow_html=True)



def _ingest_test_dataset(kb: KnowledgeBase, path: str = "data/raw/test_samples.json") -> int:
    """
    一键导入样例数据，方便本地快速跑通工作流。
    返回导入的 paper 数量。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    loader = DataLoader(path)

    # 1) 如果文件不存在，先创建假数据并保存
    if not os.path.exists(path):
        sample_papers = [
            {"id": "p1", "title": "Self-Rewarding Language Models", "abstract": "We propose...", "rating": 9.0},
            {"id": "p2", "title": "Direct Preference Optimization", "abstract": "DPO is stable...", "rating": 9.5},
        ]
        loader.save_local_data(sample_papers)

    # 2) 无论上面是否创建了新文件，这里都要读取数据并入库
    data = loader.load_local_data()
    kb.ingest_data(data)
    return int(len(data))


def _set_system_mode(mode: str) -> None:
    """
    导航切换（用于 widget 回调）。
    注意：不要在 widget 实例化之后直接修改同 key 的 session_state；
    用回调让 Streamlit 在 rerun 初期完成赋值，避免 StreamlitAPIException。
    """
    st.session_state["system_mode"] = mode


def _plan_to_markdown(plan: dict) -> str:
    """
    将 JSON 计划渲染成更易读的自然语言版（Markdown）。
    """
    if not isinstance(plan, dict):
        return "（计划为空）"

    title = str(plan.get("title") or "").strip() or "（未命名标题）"
    sections = plan.get("sections") or []
    if not isinstance(sections, list):
        sections = []

    lines = [f"**报告标题**：{title}", ""]

    est = plan.get("estimated_papers", None)
    if isinstance(est, int) and est > 0:
        lines.append(f"**预计使用论文数**：{est}")
        lines.append("")

    if not sections:
        lines.append("（无章节）")
        return "\n".join(lines)

    def _fmt_filters(f: dict) -> str:
        if not isinstance(f, dict) or not f:
            return "无"
        parts = []
        if f.get("year_in"):
            parts.append(f"年份={f.get('year_in')}")
        if f.get("min_year") is not None or f.get("max_year") is not None:
            parts.append(f"年份范围={f.get('min_year')}~{f.get('max_year')}")
        if f.get("venue_contains"):
            parts.append(f"Venue 包含「{f.get('venue_contains')}」")
        if f.get("title_contains"):
            parts.append(f"标题包含「{f.get('title_contains')}」")
        if f.get("author_contains"):
            parts.append(f"作者包含「{f.get('author_contains')}」")
        if f.get("keyword_contains"):
            parts.append(f"关键词包含「{f.get('keyword_contains')}」")
        if f.get("decision_in"):
            parts.append(f"Decision ∈ {f.get('decision_in')}")
        if f.get("presentation_in"):
            parts.append(f"展示类型 ∈ {f.get('presentation_in')}")
        if f.get("min_rating") is not None:
            parts.append(f"最低评分 ≥ {f.get('min_rating')}")
        return "；".join([str(x) for x in parts if x]) or "无"

    for i, s in enumerate(sections):
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or "").strip() or f"第 {i+1} 节"
        q = str(s.get("search_query") or "").strip()
        topk = s.get("top_k_papers", None)
        f = s.get("filters") or {}

        lines.append(f"#### {i+1}. {name}")
        if q:
            lines.append(f"- **检索 query**：`{q}`")
        lines.append(f"- **筛选**：{_fmt_filters(f)}")
        if topk is not None:
            lines.append(f"- **top_k_papers**：{topk}")
        lines.append("")

    return "\n".join(lines).strip()


def _ensure_plan_section_uids(n: int) -> None:
    """
    为可读版编辑器提供稳定的 section key（避免增删章节导致 widget key 混乱）。
    """
    try:
        import uuid
    except Exception:
        uuid = None  # type: ignore[assignment]

    uids = st.session_state.get("plan_section_uids")
    if not isinstance(uids, list):
        uids = []
    # 增补
    while len(uids) < int(n):
        uids.append((uuid.uuid4().hex if uuid else f"sec_{len(uids)}"))  # type: ignore[attr-defined]
    # 截断
    if len(uids) > int(n):
        uids = uids[: int(n)]
    st.session_state["plan_section_uids"] = uids


def _build_plan_from_readable_widgets(*, fallback_plan: dict) -> dict:
    """
    从可读版表单的 widget state 组装出标准 JSON plan。
    注意：这里不要依赖局部变量，便于 on_click 回调调用。
    """
    plan = dict(fallback_plan) if isinstance(fallback_plan, dict) else {}
    plan.pop("_error", None)

    title = (st.session_state.get("plan_edit_title") or plan.get("title") or "").strip()
    plan["title"] = title or "研究计划"

    uids = st.session_state.get("plan_section_uids") or []
    if not isinstance(uids, list):
        uids = []

    new_sections = []
    sum_topk = 0
    for uid in uids:
        uid = str(uid)
        name = (st.session_state.get(f"plan_sec_name_{uid}") or "").strip()
        query = (st.session_state.get(f"plan_sec_query_{uid}") or "").strip()
        try:
            topk = int(st.session_state.get(f"plan_sec_topk_{uid}") or 5)
        except Exception:
            topk = 5
        topk = max(1, min(topk, 50))
        sum_topk += topk

        filters: dict = {}

        years_raw = st.session_state.get(f"plan_sec_year_in_{uid}") or []
        years = []
        if isinstance(years_raw, list):
            for y in years_raw:
                try:
                    years.append(int(y))
                except Exception:
                    pass
        if years:
            filters["year_in"] = sorted(list({int(y) for y in years}))

        for k, key_name in [
            ("venue_contains", f"plan_sec_venue_contains_{uid}"),
            ("title_contains", f"plan_sec_title_contains_{uid}"),
            ("author_contains", f"plan_sec_author_contains_{uid}"),
            ("keyword_contains", f"plan_sec_keyword_contains_{uid}"),
        ]:
            v = (st.session_state.get(key_name) or "").strip()
            if v:
                filters[k] = v

        decision_raw = st.session_state.get(f"plan_sec_decision_in_{uid}") or []
        if isinstance(decision_raw, list):
            decision_in = [str(x) for x in decision_raw if str(x).strip()]
            if decision_in:
                filters["decision_in"] = decision_in

        pres_raw = st.session_state.get(f"plan_sec_presentation_in_{uid}") or []
        if isinstance(pres_raw, list):
            pres = [str(x).strip().lower() for x in pres_raw if str(x).strip()]
            if pres:
                filters["presentation_in"] = pres

        min_rating_raw = (st.session_state.get(f"plan_sec_min_rating_{uid}") or "").strip()
        if min_rating_raw:
            try:
                filters["min_rating"] = float(min_rating_raw)
            except Exception:
                pass

        new_sections.append(
            {
                "name": name or "未命名章节",
                "search_query": query or (name or ""),
                "filters": filters,
                "top_k_papers": topk,
            }
        )

    plan["sections"] = new_sections

    # estimated_papers：用户不填就按 top_k 求和
    try:
        est = st.session_state.get("plan_edit_estimated_papers")
        if est is None or str(est).strip() == "":
            plan["estimated_papers"] = int(sum_topk)
        else:
            plan["estimated_papers"] = int(est)
    except Exception:
        plan["estimated_papers"] = int(sum_topk)

    return plan


def _plan_add_section() -> None:
    plan = st.session_state.get("pending_plan")
    if not isinstance(plan, dict):
        return
    secs = plan.get("sections") or []
    if not isinstance(secs, list):
        secs = []
    secs.append({"name": "新章节", "search_query": "", "filters": {}, "top_k_papers": 5})
    plan["sections"] = secs
    st.session_state["pending_plan"] = plan
    _ensure_plan_section_uids(len(secs))
    st.session_state["plan_editor_text"] = json.dumps(plan, ensure_ascii=False, indent=2)
    st.session_state["plan_flash"] = "已添加一个新章节。"


def _plan_delete_section(uid: str) -> None:
    plan = st.session_state.get("pending_plan")
    if not isinstance(plan, dict):
        return
    secs = plan.get("sections") or []
    if not isinstance(secs, list):
        secs = []
    uids = st.session_state.get("plan_section_uids") or []
    if not isinstance(uids, list):
        uids = []

    uid = str(uid)
    if uid in uids:
        idx = uids.index(uid)
        if 0 <= idx < len(secs):
            secs.pop(idx)
        uids.pop(idx)

    plan["sections"] = secs
    st.session_state["pending_plan"] = plan
    st.session_state["plan_section_uids"] = uids
    st.session_state["plan_editor_text"] = json.dumps(plan, ensure_ascii=False, indent=2)
    st.session_state["plan_flash"] = "已删除该章节。"


def _plan_apply_readable() -> None:
    plan = st.session_state.get("pending_plan")
    if not isinstance(plan, dict):
        return
    new_plan = _build_plan_from_readable_widgets(fallback_plan=plan)
    st.session_state["pending_plan"] = new_plan
    st.session_state["plan_editor_text"] = json.dumps(new_plan, ensure_ascii=False, indent=2)
    st.session_state["plan_flash"] = "计划已更新（已同步到 JSON）。"


def _plan_apply_json() -> None:
    txt = st.session_state.get("plan_editor_text") or ""
    try:
        plan = json.loads(txt)
        if not isinstance(plan, dict):
            raise ValueError("plan 不是 JSON object")
        secs = plan.get("sections") or []
        if not isinstance(secs, list):
            plan["sections"] = []
        st.session_state["pending_plan"] = plan
        # 让 uids 重新匹配
        st.session_state["plan_section_uids"] = []
        _ensure_plan_section_uids(len(plan.get("sections") or []))
        st.session_state["plan_editor_text"] = json.dumps(plan, ensure_ascii=False, indent=2)
        st.session_state["plan_flash"] = "已从 JSON 覆盖可读版。"
    except Exception as e:
        st.session_state["plan_flash_error"] = f"计划 JSON 解析失败：{e}"


def _plan_run_from_readable() -> None:
    plan = st.session_state.get("pending_plan")
    if not isinstance(plan, dict):
        return
    new_plan = _build_plan_from_readable_widgets(fallback_plan=plan)
    st.session_state["pending_plan"] = new_plan
    st.session_state["plan_editor_text"] = json.dumps(new_plan, ensure_ascii=False, indent=2)
    st.session_state["plan_run_requested"] = "readable"


def _plan_run_from_json() -> None:
    txt = st.session_state.get("plan_editor_text") or ""
    try:
        plan = json.loads(txt)
        if not isinstance(plan, dict):
            raise ValueError("plan 不是 JSON object")
        st.session_state["pending_plan"] = plan
        st.session_state["plan_run_requested"] = "json"
    except Exception as e:
        st.session_state["plan_flash_error"] = f"计划 JSON 解析失败：{e}"


def _render_data_dashboard(
    *,
    embedding_model: str,
    embedding_api_key: Optional[str],
    embedding_base_url: Optional[str],
    use_fake_embeddings: bool,
) -> None:
    st.header("知识库管理")

    # --- 当前知识库概览（解决“Ctrl+C 后看不到数据”的困惑）---
    kb = KnowledgeBase(
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
    )
    kb.initialize_db()

    db_path = getattr(kb, "db_path", "data/lancedb")
    meta_path = getattr(kb, "metadata_path", "data/lancedb/metadata.sqlite")

    # SQLite / LanceDB 统计
    df = kb.search_structured()
    papers_count = int(len(df)) if hasattr(df, "__len__") else 0

    reviews_count = 0
    try:
        if kb._meta_conn is not None:
            row = kb._meta_conn.execute("SELECT COUNT(*) AS c FROM reviews").fetchone()
            reviews_count = int(row["c"]) if row and "c" in row.keys() else int(row[0])  # type: ignore[index]
    except Exception:
        reviews_count = 0

    papers_vec_count = 0
    chunks_vec_count = 0
    try:
        if kb.db is not None:
            names = set(kb.db.table_names())
            if "papers" in names:
                papers_vec_count = int(kb.db.open_table("papers").count_rows())
            if "chunks" in names:
                chunks_vec_count = int(kb.db.open_table("chunks").count_rows())
    except Exception:
        pass

    pdf_count = 0
    try:
        pdf_dir = Path("data/raw/pdfs")
        if pdf_dir.exists():
            pdf_count = len(list(pdf_dir.glob("*.pdf")))
    except Exception:
        pdf_count = 0

    st.caption(f"当前知识库目录：`{db_path}` · 元数据库：`{meta_path}`（Ctrl+C 不会清空数据）")

    # 删除/导入等操作后的“闪现提示”
    flash = st.session_state.pop("kb_flash", None)
    if isinstance(flash, str) and flash.strip():
        st.success(flash.strip())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("论文（SQLite）", papers_count)
    c2.metric("评审（SQLite）", reviews_count)
    c3.metric("向量 Papers（LanceDB）", papers_vec_count)
    c4.metric("向量 Chunks（LanceDB）", chunks_vec_count)
    st.caption(f"PDF 文件数（data/raw/pdfs）：{pdf_count}")

    c_fix, c_fix_hint = st.columns([1, 3])
    with c_fix:
        if st.button("🛠 修复 pdf_path（扫描本地 PDF）", key="kb_repair_pdf_path"):
            with st.spinner("正在扫描并回填 pdf_path..."):
                res = kb.repair_pdf_paths(pdf_dir="data/raw/pdfs")
            if isinstance(res, dict) and res.get("ok"):
                st.session_state["kb_flash"] = f"已回填 pdf_path：{res.get('updated')}（扫描候选 {res.get('scanned')}）"
                _rerun()
            else:
                st.error(f"修复失败：{res}")
    with c_fix_hint:
        st.caption(
            "当你在“不开启下载 PDF”的情况下重新抓取元数据，旧版本可能会把 SQLite 里的 pdf_path 覆盖为空。"
            "这个按钮会扫描 `data/raw/pdfs/<paper_id>.pdf` 并回填到 SQLite。"
        )

    with st.expander("查看已入库论文（预览/搜索）", expanded=False):
        if papers_count <= 0:
            st.info("当前知识库为空。请先导入样例或抓取 OpenReview。", icon="ℹ️")
        else:
            q = st.text_input("按标题关键词过滤", value="", placeholder="例如：DPO / alignment / agent")
            view = df.copy()
            try:
                if q.strip() and "title" in view.columns:
                    view = view[view["title"].fillna("").str.contains(q.strip(), case=False, regex=False)]
            except Exception:
                pass

            # 只展示高频字段
            cols = [
                c
                for c in ["id", "title", "year", "rating", "decision", "presentation", "pdf_path", "updated_at"]
                if c in view.columns
            ]

            # --- 表格内选择（勾选） + 删除/详情一体化 ---
            try:
                # 切换筛选时清空选择，避免“看不见但被选中”的误删
                _last_q = str(st.session_state.get("_kb_last_filter_q") or "")
                if str(q or "") != _last_q:
                    st.session_state["_kb_last_filter_q"] = str(q or "")
                    st.session_state["_kb_selected_ids"] = []
            except Exception:
                pass

            selected_ids: list[str] = []
            try:
                shown = view[cols].head(500).copy()
                shown.insert(0, "选择", False)
                # 预填已有选择
                prev = st.session_state.get("_kb_selected_ids") or []
                prev_set = set([str(x) for x in prev if str(x).strip()])
                if "id" in shown.columns and prev_set:
                    shown["选择"] = shown["id"].astype(str).isin(prev_set)

                edited = st.data_editor(
                    shown,
                    key="kb_table_editor",
                    disabled=[c for c in shown.columns if c != "选择"],
                    **_width_kwargs(st.data_editor, stretch=True),
                )
                try:
                    selected_ids = (
                        edited[edited["选择"] == True]["id"]  # noqa: E712
                        .astype(str)
                        .tolist()
                    )
                except Exception:
                    selected_ids = []
                st.session_state["_kb_selected_ids"] = selected_ids
            except Exception:
                # 兜底：不支持 data_editor 时退化为只读表
                st.dataframe(view[cols].head(500), **_width_kwargs(st.dataframe, stretch=True))
                selected_ids = []

            c_sel1, c_sel2, c_sel3 = st.columns([1, 1, 3])
            with c_sel1:
                st.caption(f"已选中：{len(selected_ids)}")
            with c_sel2:
                if st.button("清空选择", key="kb_clear_selection", disabled=(not selected_ids)):
                    st.session_state["_kb_selected_ids"] = []
                    _rerun()
            with c_sel3:
                st.caption("提示：勾选 1 篇会自动显示详情；勾选多篇可直接批量删除。")

            st.divider()
            st.markdown("**删除/操作**")
            st.warning("删除不可撤销：会删除论文元数据/评审/向量索引（以及可选本地 PDF）。")

            batch_delete_pdf = st.checkbox(
                "同时删除本地 PDF 文件（如果存在）",
                value=False,
                key="kb_del_pdf_selected",
            )
            batch_confirm = st.checkbox(
                f"我已确认要删除选中的 {len(selected_ids)} 篇论文",
                value=False,
                key="kb_del_confirm_selected",
            )
            if st.button(
                f"删除选中（{len(selected_ids)}）",
                type="primary",
                disabled=(not batch_confirm) or (not selected_ids),
                key="kb_del_btn_selected",
                **_width_kwargs(st.button, stretch=True),
            ):
                with st.spinner("正在批量删除..."):
                    res = kb.delete_papers(selected_ids, delete_pdf=batch_delete_pdf)
                if isinstance(res, dict) and res.get("ok"):
                    msg = f"已删除 {res.get('deleted_sql_papers')} 篇论文（reviews={res.get('deleted_sql_reviews')}）"
                    if batch_delete_pdf:
                        msg += f" · 删除本地 PDF：{res.get('deleted_pdf')}/{res.get('requested')}"
                    st.session_state["kb_flash"] = msg
                    _rerun()
                else:
                    st.error(f"批量删除失败：{(res or {}).get('error') if isinstance(res, dict) else res}")

            st.divider()
            st.markdown("**单篇详情**")
            pid = selected_ids[0] if len(selected_ids) == 1 else None
            if not pid:
                if len(selected_ids) > 1:
                    st.info("当前选中多篇：请只勾选 1 篇以查看单篇详情。", icon="ℹ️")
                else:
                    st.info("请在上表勾选 1 篇论文以查看详情。", icon="ℹ️")
            else:
                paper = kb.get_paper(pid) or {}
                reviews = kb.get_reviews(pid) or []
                st.markdown(f"**{paper.get('title','')}**")
                st.caption(
                    f"paper_id={pid} · year={paper.get('year')} · rating={paper.get('rating')} · "
                    f"decision={paper.get('decision')} · presentation={paper.get('presentation')}"
                )
                if paper.get("abstract"):
                    st.markdown("**Abstract**")
                    st.write(paper.get("abstract"))

                decision_text = str(paper.get("decision_text") or "").strip()
                if decision_text:
                    # 默认收起，避免长文本占屏
                    preview = " ".join(decision_text.split())[:120]
                    title_line = "Decision（最终决策说明）"
                    if preview:
                        title_line += f" · {preview}" + ("…" if len(preview) >= 120 else "")
                    with st.expander(title_line, expanded=False):
                        st.write(decision_text)

                rebuttal_text = str(paper.get("rebuttal_text") or "").strip()
                if rebuttal_text:
                    preview = " ".join(rebuttal_text.split())[:120]
                    title_line = "Rebuttal / Author Response"
                    if preview:
                        title_line += f" · {preview}" + ("…" if len(preview) >= 120 else "")
                    with st.expander(title_line, expanded=False):
                        st.write(rebuttal_text)

                if reviews:
                    st.markdown("**Reviews（前 3 条）**")
                    for i, r in enumerate(reviews[:3]):
                        ridx = i + 1
                        rating_raw = (r.get("rating_raw") if isinstance(r, dict) else None) or ""
                        conf_raw = (r.get("confidence_raw") if isinstance(r, dict) else None) or ""
                        title_line = f"Review #{ridx}"
                        if rating_raw or conf_raw:
                            title_line += f" · rating={rating_raw} · confidence={conf_raw}"
                        with st.expander(title_line, expanded=False):
                            if isinstance(r, dict) and (r.get("text") or "").strip():
                                st.write((r.get("text") or "").strip())
                            else:
                                st.caption("（未解析到评审正文 text：可能是旧数据或该会议字段名不同/权限不足）")
                                st.json(r, expanded=False)

                st.divider()
                st.markdown("**删除该论文（单篇）**")
                delete_pdf = st.checkbox("同时删除本地 PDF 文件（如果存在）", value=False, key=f"del_pdf_{pid}")
                confirm = st.checkbox("我已确认要删除这篇论文", value=False, key=f"del_confirm_{pid}")
                if st.button("删除该论文", type="primary", disabled=not confirm, key=f"del_btn_{pid}"):
                    with st.spinner("正在删除..."):
                        res = kb.delete_paper(pid, delete_pdf=delete_pdf)
                    if isinstance(res, dict) and res.get("ok"):
                        msg = f"已删除 paper_id={pid}（reviews={res.get('deleted_sql_reviews')}）"
                        if delete_pdf:
                            if res.get("deleted_pdf"):
                                msg += " · 本地 PDF 已删除"
                            elif res.get("pdf_path"):
                                msg += " · 本地 PDF 未删除（可能不存在/无权限）"
                        st.session_state["kb_flash"] = msg
                        _rerun()
                    else:
                        st.error(f"删除失败：{(res or {}).get('error') if isinstance(res, dict) else res}")

    tab1, tab2 = st.tabs(["导入本地样例", "抓取 OpenReview"])

    with tab1:
        st.subheader("快速导入样例（用于本地跑通流程）")
        if st.button("一键导入样例数据", type="primary"):
            with st.spinner("正在导入样例数据..."):
                kb = KnowledgeBase(
                    embedding_model=embedding_model,
                    embedding_api_key=embedding_api_key,
                    embedding_base_url=embedding_base_url,
                )
                kb.initialize_db()
                n = _ingest_test_dataset(kb)
                st.success(f"已导入 {n} 篇样例论文（SQLite + LanceDB）。")

    with tab2:
        st.subheader("OpenReview 实时抓取入库")
        st.info(
            "建议按顺序填写：① 会议/年份（Venue ID）→ ② 抓取范围（是否只要 Accept）→ ③ 是否下载/解析 PDF → 开始入库。",
            icon="ℹ️",
        )

        # ---------------------------
        # 1) 选择会议
        # ---------------------------
        st.markdown("#### 1) 选择会议（OpenReview Venue ID）")
        st.session_state.setdefault("or_auto_sync_venue", True)
        st.session_state.setdefault("_or_last_auto_venue", "")
        st.session_state.setdefault("or_venue_id", "NeurIPS.cc/2024/Conference")

        pick_mode = st.radio(
            "Venue ID 输入方式",
            options=["热门会议快捷选择（主会）", "自定义 Venue ID（高级）"],
            horizontal=True,
        )

        venue_id = ""
        if pick_mode.startswith("热门会议"):
            conf_map = {
                "NeurIPS": "NeurIPS.cc",
                "ICLR": "ICLR.cc",
                "ICML": "ICML.cc",
                "CoRL": "CoRL.cc",
                "COLM": "COLM.cc",
            }
            c1, c2, c3 = st.columns([0.40, 0.20, 0.40], gap="large")
            with c1:
                conf = st.selectbox("会议（主会）", options=list(conf_map.keys()), index=0)
            with c2:
                year = st.selectbox("年份", options=list(range(2019, 2026)), index=5)  # default 2024
            with c3:
                track_choice = st.selectbox(
                    "Track（主会通常为 Conference）",
                    options=["Conference", "Workshop", "自定义"],
                    index=0,
                    help="OpenReview 的 Venue ID 最后一段；主会一般是 Conference。",
                )
                track = track_choice
                if track_choice == "自定义":
                    track = st.text_input("自定义 Track", value="Conference")

            auto_sync = st.checkbox(
                "自动生成并同步 Venue ID",
                value=bool(st.session_state.get("or_auto_sync_venue")),
                help="关闭后你可以手动修改 Venue ID，不会被会议/年份变化覆盖。",
                key="or_auto_sync_venue",
            )

            auto_venue = f"{conf_map[conf]}/{year}/{track}".strip()
            if auto_sync and auto_venue and st.session_state.get("_or_last_auto_venue") != auto_venue:
                st.session_state["_or_last_auto_venue"] = auto_venue
                st.session_state["or_venue_id"] = auto_venue

            venue_id = st.text_input(
                "Venue ID（最终会使用这个）",
                value=str(st.session_state.get("or_venue_id") or auto_venue),
                key="or_venue_id",
                help="例：NeurIPS.cc/2024/Conference",
            )
        else:
            venue_id = st.text_input(
                "会议 Venue ID（OpenReview）",
                value=str(st.session_state.get("or_venue_id") or "NeurIPS.cc/2024/Conference"),
                key="or_venue_id",
                help="格式通常：<Conf>.cc/<Year>/<Track>，例如 NeurIPS.cc/2024/Conference",
            )

        venue_id = (venue_id or "").strip()
        if venue_id:
            st.caption(f"将使用 OpenReview invitation：`{venue_id}/-/Submission`")
        else:
            st.warning("请先填写 Venue ID。", icon="⚠️")

        # ---------------------------
        # 2) 抓取范围与筛选
        # ---------------------------
        st.markdown("#### 2) 抓取范围与筛选")
        scope = st.radio(
            "抓取范围",
            options=["全部（含 Reject/Pending）", "仅 Accept（含 oral/spotlight/poster）"],
            horizontal=True,
        )
        accepted_only = scope.startswith("仅 Accept")
        # 新增：一键抓取所有 AC 论文
        fetch_all_ac = st.checkbox(
            "抓取该会议全部 Accept 论文（不限数量）",
            value=False,
            help="开启后：将忽略上方“抓取范围”和下方“数量上限”，自动抓取该会议的所有接收论文（可能包含数千篇，耗时较长）。",
        )

        if fetch_all_ac:
            accepted_only = True
            limit = None
            st.info("已开启全量抓取：将获取该会议所有 Accepted 论文。", icon="🚀")
        else:
            limit = st.slider(
                "抓取数量上限",
                10,
                300,
                50,
                help="当开启“仅 Accept”时，这个上限指 accepted 论文数量；系统会扫描更多 submission 直到凑够或扫完。"
                "当开启“追加抓取（只抓新论文）”时，这个上限指“新增论文数量”。",
            )

        skip_existing = st.checkbox(
            "追加抓取（只抓新论文，跳过已入库 paper_id）",
            value=False,
            help="开启后：如果你库里已经有 300 篇，再抓 300 会尽量再新增 300 篇（会扫描更多 submission）。"
            "关闭则表示“刷新/补全已有论文元数据”。",
        )

        presentation_in = None
        if accepted_only:
            st.caption("提示：展示类型来自 decision 文本解析；未标明类型的 accept 会记为 unknown。")
            presentation_in = st.multiselect(
                "Accept 展示类型（可选）",
                options=["oral", "spotlight", "poster", "unknown"],
                default=["oral", "spotlight", "poster", "unknown"],
                help="只在开启“仅 Accept”时生效。",
            )

        # ---------------------------
        # 3) PDF 下载与解析
        # ---------------------------
        st.markdown("#### 3) PDF 下载与解析（可选）")
        download_pdfs = st.checkbox(
            "下载 PDF（保存到 data/raw/pdfs）",
            value=True,
            help="不下载也能做元数据分析；下载后才能解析全文。",
        )
        parse_pdfs = st.checkbox(
            "解析 PDF 全文（较慢，但检索效果更好）",
            value=True,
            disabled=not download_pdfs,
            help="解析结果会写入向量库 chunks 表（用于证据检索与引用）。",
        )
        max_pages = st.slider(
            "解析 PDF 最大页数",
            1,
            50,
            12,
            disabled=not parse_pdfs,
            help="越大越慢；建议先用 8-12 页跑通流程，再逐步加大。",
        )

        # ---------------------------
        # Advanced knobs（可选）
        # ---------------------------
        with st.expander("高级（速度/稳定性，可选）", expanded=False):
            st.markdown("**OpenReview 认证（可选）**")
            st.caption("部分会议的评审/回复需要登录后才能通过 API 获取；不填也能抓论文元数据与 PDF。")
            or_user = st.text_input(
                "OpenReview Username（可选）",
                value=str(os.getenv("OPENREVIEW_USERNAME", "") or ""),
                help="也可通过环境变量 OPENREVIEW_USERNAME 设置。",
            )
            or_pass = st.text_input(
                "OpenReview Password（可选）",
                value=str(os.getenv("OPENREVIEW_PASSWORD", "") or ""),
                type="password",
                help="也可通过环境变量 OPENREVIEW_PASSWORD 设置。不会写入磁盘，仅在当前进程生效。",
            )
            force_replace_reviews = st.checkbox(
                "强制刷新 Reviews（允许覆盖为空，MUJICA_REPLACE_EMPTY_REVIEWS）",
                value=(os.getenv("MUJICA_REPLACE_EMPTY_REVIEWS", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"},
                help="用于修复历史数据/误分类：如果你发现 Reviews 里混进了 Rebuttal/Author Response，"
                "勾上一次可以允许本次抓取结果覆盖旧 reviews（即使本次抓不到 reviews，也会清空旧的）。",
            )

            st.divider()
            page_size = st.number_input(
                "OpenReview 分页大小（MUJICA_OPENREVIEW_PAGE_SIZE）",
                min_value=20,
                max_value=1000,
                value=int(os.getenv("MUJICA_OPENREVIEW_PAGE_SIZE", "200") or 200),
                step=20,
                help="越大请求次数越少，但单次返回更大；accepted-only 可能会扫描更多页。",
            )
            pdf_workers = st.number_input(
                "PDF 下载并发线程（MUJICA_PDF_DOWNLOAD_WORKERS）",
                min_value=1,
                max_value=16,
                value=int(os.getenv("MUJICA_PDF_DOWNLOAD_WORKERS", "6") or 6),
                step=1,
            )
            pdf_timeout = st.number_input(
                "PDF 下载超时（秒，MUJICA_PDF_DOWNLOAD_TIMEOUT）",
                min_value=5.0,
                max_value=300.0,
                value=float(os.getenv("MUJICA_PDF_DOWNLOAD_TIMEOUT", "60") or 60),
                step=5.0,
            )
            pdf_retries = st.number_input(
                "PDF 下载重试次数（MUJICA_PDF_DOWNLOAD_RETRIES）",
                min_value=0,
                max_value=5,
                value=int(os.getenv("MUJICA_PDF_DOWNLOAD_RETRIES", "2") or 2),
                step=1,
            )
            pdf_delay = st.number_input(
                "每次请求前延迟（秒，MUJICA_PDF_DOWNLOAD_DELAY）",
                min_value=0.0,
                max_value=5.0,
                value=float(os.getenv("MUJICA_PDF_DOWNLOAD_DELAY", "0.0") or 0.0),
                step=0.1,
            )
            pdf_force_redownload = st.checkbox(
                "强制重下已存在 PDF（覆盖，MUJICA_PDF_FORCE_REDOWNLOAD）",
                value=(os.getenv("MUJICA_PDF_FORCE_REDOWNLOAD", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"},
                help="开启后：即使本地已有同名 PDF，也会重新下载覆盖（用于修复历史损坏/不完整 PDF）。",
            )
            pdf_validate_existing = st.checkbox(
                "校验已存在 PDF（损坏/过小则重下，MUJICA_PDF_VALIDATE_EXISTING）",
                value=(os.getenv("MUJICA_PDF_VALIDATE_EXISTING", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"},
                help="开启后：若本地 PDF 不是有效 PDF（或小于最小大小），会自动触发重下。",
            )
            pdf_min_bytes = st.number_input(
                "最小 PDF 大小（字节，MUJICA_PDF_MIN_BYTES）",
                min_value=0,
                max_value=50_000_000,
                value=int(os.getenv("MUJICA_PDF_MIN_BYTES", "10240") or 10240),
                step=1024,
                help="用于判定“下载到 HTML/错误页/空文件”等异常情况（默认 10KB）。",
            )
            pdf_eof_check = st.checkbox(
                "校验 PDF EOF 标记（MUJICA_PDF_EOF_CHECK）",
                value=(os.getenv("MUJICA_PDF_EOF_CHECK", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"},
                help="更严格的校验：检查文件尾部是否包含 %%EOF（有助于发现截断文件）。",
            )
            st.caption("这些参数只影响当前 Streamlit 进程；重启后会恢复为 .env/环境变量的值。")

        with st.expander("本次会写入哪些内容？", expanded=False):
            st.markdown(
                "- **SQLite**：title/authors/keywords/year/decision/decision_text/rebuttal_text/presentation/rating/reviews/pdf_url/pdf_path\n"
                "- **LanceDB**：paper 向量 + chunks（含 meta chunk；含 decision/rebuttal/review chunks；若勾选解析则含全文 chunks）"
            )

        st.caption("配置预览（你点开始前可以快速确认）：")
        st.json(
            {
                "venue_id": venue_id,
                "limit": limit,
                "accepted_only": accepted_only,
                "presentation_in": presentation_in,
                "download_pdfs": download_pdfs,
                "parse_pdfs": parse_pdfs,
                "max_pdf_pages": max_pages if parse_pdfs else None,
                "force_replace_reviews": force_replace_reviews,
            },
            expanded=False,
        )

        if st.button("开始抓取并入库", type="primary", **_width_kwargs(st.button, stretch=True)):
            if not venue_id:
                st.error("Venue ID 不能为空。请先选择会议/年份或手动填写。")
                st.stop()

            # 将高级参数写入环境变量（fetcher 内部按 env 读取）
            try:
                if str(or_user or "").strip():
                    os.environ["OPENREVIEW_USERNAME"] = str(or_user).strip()
                if str(or_pass or "").strip():
                    os.environ["OPENREVIEW_PASSWORD"] = str(or_pass).strip()
                os.environ["MUJICA_REPLACE_EMPTY_REVIEWS"] = "1" if force_replace_reviews else "0"
                os.environ["MUJICA_OPENREVIEW_PAGE_SIZE"] = str(int(page_size))
                os.environ["MUJICA_PDF_DOWNLOAD_WORKERS"] = str(int(pdf_workers))
                os.environ["MUJICA_PDF_DOWNLOAD_TIMEOUT"] = str(float(pdf_timeout))
                os.environ["MUJICA_PDF_DOWNLOAD_RETRIES"] = str(int(pdf_retries))
                os.environ["MUJICA_PDF_DOWNLOAD_DELAY"] = str(float(pdf_delay))
                os.environ["MUJICA_PDF_FORCE_REDOWNLOAD"] = "1" if pdf_force_redownload else "0"
                os.environ["MUJICA_PDF_VALIDATE_EXISTING"] = "1" if pdf_validate_existing else "0"
                os.environ["MUJICA_PDF_MIN_BYTES"] = str(int(pdf_min_bytes))
                os.environ["MUJICA_PDF_EOF_CHECK"] = "1" if pdf_eof_check else "0"
            except Exception:
                pass

            # 预检：embedding 不可用时直接提示（否则会在终端刷屏且无法语义检索）
            if (not use_fake_embeddings) and (not embedding_api_key):
                st.error('未配置 Embedding 所需的 API Key。请在侧边栏填写 Key，或开启"离线 Embedding"。')
                st.stop()

            if not use_fake_embeddings:
                test_vec = get_embedding(
                    "ping",
                    model=embedding_model,
                    api_key=embedding_api_key,
                    base_url=embedding_base_url,
                )
                if not test_vec:
                    st.error(
                        f"Embedding 初始化失败：模型 `{embedding_model}` 不存在/不支持或鉴权失败。"
                        '请更换 Embedding Model（注意：embedding 模型通常与聊天模型不同），或开启"离线 Embedding"。'
                    )
                    st.stop()

            # 创建 KnowledgeBase 和 Ingestor
            kb = KnowledgeBase(
                embedding_model=embedding_model,
                embedding_api_key=embedding_api_key,
                embedding_base_url=embedding_base_url,
            )
            kb.initialize_db()
            ingestor = OpenReviewIngestor(kb, fetcher=ConferenceDataFetcher(output_dir="data/raw"))

            # 创建后台任务并启动
            job = _IngestJob(
                job_id=f"ingest-{uuid.uuid4().hex[:8]}",
                venue_id=venue_id,
            )
            job.thread = threading.Thread(
                target=_run_ingest_job,
                kwargs={
                    "job": job,
                    "ingestor": ingestor,
                    "venue_id": venue_id,
                    "limit": limit,
                    "accepted_only": accepted_only,
                    "presentation_in": presentation_in,
                    "skip_existing": skip_existing,
                    "download_pdfs": download_pdfs,
                    "parse_pdfs": parse_pdfs,
                    "max_pdf_pages": max_pages if parse_pdfs else None,
                    "max_downloads": limit if download_pdfs else None,
                },
                daemon=True,
            )
            job.thread.start()
            st.session_state["ingest_job"] = job
            _rerun()

        # -------------------------------------------------------
        # 入库任务进度显示（使用 @st.fragment 实现独立刷新）
        # -------------------------------------------------------
        ingest_job: Optional[_IngestJob] = st.session_state.get("ingest_job")
        if ingest_job is not None:
            st.divider()
            
            @st.fragment(run_every="0.8s")
            def _ingest_progress_fragment():
                """独立刷新的进度 Fragment：不受外部 UI 变化影响"""
                job = st.session_state.get("ingest_job")
                if job is None:
                    return
                
                with job.lock:
                    status = job.status
                    message = job.message
                    progress = dict(job.progress)
                    result = list(job.result) if job.result else []
                    error = job.error
                    error_trace = job.error_trace
                
                if status == "running":
                    st.info(f"🔄 {message}")
                    
                    # 显示各阶段进度条
                    col1, col2 = st.columns(2)
                    with col1:
                        fetch_p = progress.get("fetch_papers", {})
                        if fetch_p:
                            cur, tot = fetch_p.get("current", 0), fetch_p.get("total", 0)
                            pct = int(cur * 100 / tot) if tot > 0 else 0
                            st.caption(f"抓取元数据: {cur}/{tot}")
                            st.progress(min(100, pct))
                        
                        parse_p = progress.get("parse_pdf", {})
                        if parse_p:
                            cur, tot = parse_p.get("current", 0), parse_p.get("total", 0)
                            pct = int(cur * 100 / tot) if tot > 0 else 0
                            st.caption(f"解析 PDF: {cur}/{tot}")
                            st.progress(min(100, pct))
                    
                    with col2:
                        dl_p = progress.get("download_pdf", {})
                        if dl_p:
                            cur, tot = dl_p.get("current", 0), dl_p.get("total", 0)
                            pct = int(cur * 100 / tot) if tot > 0 else 0
                            st.caption(f"下载 PDF: {cur}/{tot}")
                            st.progress(min(100, pct))
                        
                        embed_p = progress.get("embed_chunks", {}) or progress.get("embed_papers", {})
                        if embed_p:
                            cur, tot = embed_p.get("current", 0), embed_p.get("total", 0)
                            pct = int(cur * 100 / tot) if tot > 0 else 0
                            st.caption(f"Embedding: {cur}/{tot}")
                            st.progress(min(100, pct))
                    
                    # 停止按钮
                    if st.button("⏹ 停止入库", key="stop_ingest_btn"):
                        job.cancel_event.set()
                        st.warning("正在停止...")
                
                elif status == "done":
                    st.success(f"✅ {message}")
                    # 显示统计
                    try:
                        papers = result
                        decided = sum(1 for p in (papers or []) if (p or {}).get("decision"))
                        rated = sum(1 for p in (papers or []) if (p or {}).get("rating") is not None)
                        reviewed = sum(1 for p in (papers or []) if (p or {}).get("reviews"))
                        st.caption(f"decision={decided} · rating={rated} · reviews={reviewed}")
                    except Exception:
                        pass
                    # 清除 job 以结束 fragment 刷新
                    if st.button("清除", key="clear_ingest_job"):
                        st.session_state.pop("ingest_job", None)
                        _rerun()
                
                elif status == "cancelled":
                    st.warning(f"⚠️ {message}")
                    if st.button("清除", key="clear_ingest_job_cancelled"):
                        st.session_state.pop("ingest_job", None)
                        _rerun()
                
                elif status == "error":
                    st.error(f"❌ {message}")
                    if error_trace:
                        with st.expander("错误详情"):
                            st.code(error_trace)
                    if st.button("清除", key="clear_ingest_job_error"):
                        st.session_state.pop("ingest_job", None)
                        _rerun()
            
            # 调用 Fragment
            _ingest_progress_fragment()


def _render_research_agent(
    *,
    chat_api_key: Optional[str],
    chat_base_url: Optional[str],
    model_name: str,
    embedding_model: str,
    embedding_api_key: Optional[str],
    embedding_base_url: Optional[str],
    use_fake_embeddings: bool,
) -> None:
    # 初始化/连接知识库（不依赖 LLM）
    kb = KnowledgeBase(
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
    )
    kb.initialize_db()

    # 让用户明确知道“数据是否在库里”
    try:
        _df_kb = kb.search_structured()
        kb_papers = int(len(_df_kb))
    except Exception:
        kb_papers = 0

    try:
        chunks_rows = int(kb.db.open_table("chunks").count_rows()) if kb.db is not None else 0
    except Exception:
        chunks_rows = 0

    # 新问题：由首页输入框 / 底部 chat_input 写入 session_state
    user_query = st.session_state.pop("pending_user_query", None)
    has_auth = bool((chat_api_key or "").strip())

    has_messages = bool(st.session_state.get("messages"))
    has_any_result = bool(st.session_state.get("pending_plan") or st.session_state.get("final_report"))
    show_workspace = has_messages or has_any_result or bool(user_query)

    # ---------------------------
    # Landing（参考截图：大留白 + 居中卡片 + 推荐示例）
    # ---------------------------
    if not show_workspace:
        st.markdown(
            """
<div class="mujica-hero">
  <div class="mujica-hero-title">用 MUJICA 生成论文调研报告</div>
  <div class="mujica-hero-subtitle">输入一个主题，系统会自动规划 → 检索证据 → 写作 → 核查（全程可溯源）</div>
</div>
            """.strip(),
            unsafe_allow_html=True,
        )

        st.write("")
        if not has_auth:
            st.warning(
                "运行前需要配置鉴权：请在左侧栏填写 **API Key**，或输入正确的 **Access Code**（用于启用系统 Key）。"
                "否则无法进行「规划/写作/核查」。",
                icon="🔑",
            )
        # 兼容 Streamlit 1.26：st.container 不支持 border 参数
        # 这里用 st.form 做“卡片容器”，再用 CSS 把 form 渲染成卡片。
        with st.form("landing_card", clear_on_submit=False):
            topic = st.text_input(
                "研究问题 / 报告主题",
                placeholder="例如：对比 NeurIPS 2024 高分 vs 低分论文的评审关注点差异",
                key="landing_topic",
                help="用于生成研究计划与报告结构（相当于你想让系统回答的问题）。",
            )
            keywords = st.text_input(
                "辅助关键词（可选）",
                placeholder="例如：DPO, alignment, preference；或：robustness, backdoor, elicitation",
                key="landing_keywords",
                help="可补充你关心的术语/子方向；会与研究问题一起作为检索提示（不是硬过滤）。",
            )

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                do_ingest = st.form_submit_button(
                    "一键导入样例",
                    type="primary",
                    **_width_kwargs(st.form_submit_button, stretch=True),
                )
            with c2:
                go_kb = st.form_submit_button(
                    "打开知识库",
                    **_width_kwargs(st.form_submit_button, stretch=True),
                    on_click=_set_system_mode,
                    args=("data",),
                )
            with c3:
                do_run = st.form_submit_button(
                    "开始生成",
                    type="primary",
                    **_width_kwargs(st.form_submit_button, stretch=True),
                )

        if do_ingest:
            with st.spinner("正在导入样例数据..."):
                n = _ingest_test_dataset(kb)
            st.success(f"已导入 {n} 篇样例论文。")
            _rerun()

        # go_kb：已由 on_click 切换导航；Streamlit 会自动 rerun，无需手动 rerun

        if do_run:
            if not (topic or "").strip():
                st.warning("请先填写「研究问题 / 报告主题」。")
            elif not has_auth:
                st.warning(
                    "未配置鉴权：请先在左侧栏填写 **API Key** 或输入正确 **Access Code**，否则无法开始生成。",
                    icon="🔑",
                )
            else:
                q = topic.strip()
                if (keywords or "").strip():
                    q = f"{q}\n辅助关键词：{keywords.strip()}"
                st.session_state["pending_user_query"] = q
                _rerun()

        st.write("")
        st.subheader("推荐示例")
        samples = [
            ("DPO 研究趋势", "总结 NeurIPS 2024 中 DPO 相关研究趋势，并列出代表性结论与证据。"),
            ("评审观点对比", "对比 NeurIPS 2024 中高分论文与低分论文的评审关注点差异。"),
            ("某方向方法谱系", "梳理 NeurIPS 2024 中 Agent/Tool Use 方向的方法谱系，并给出关键证据。"),
        ]
        cols = st.columns(3)
        for i, (t, q) in enumerate(samples):
            with cols[i]:
                # 同上：用 st.form 做卡片容器
                with st.form(f"sample_card_{i}", clear_on_submit=False):
                    st.markdown(f"**{t}**")
                    st.caption(q[:80] + ("…" if len(q) > 80 else ""))
                    use_it = st.form_submit_button(
                        "使用这个示例",
                        **_width_kwargs(st.form_submit_button, stretch=True),
                    )
                if use_it:
                    if not has_auth:
                        st.warning(
                            "未配置鉴权：请先在左侧栏填写 **API Key** 或输入正确 **Access Code**，否则无法开始生成。",
                            icon="🔑",
                        )
                    else:
                        st.session_state["pending_user_query"] = q
                        _rerun()

        return

    # ---------------------------
    # Workspace（对话 + 证据/核查）
    # ---------------------------
    st.markdown("### 深度洞察助手")
    st.caption(f"知识库：{kb_papers} 篇论文 · chunks={chunks_rows}（建议：先导入数据→再提问）")

    col_chat, col_context = st.columns([0.65, 0.35], gap="large")

    with col_chat:
        # 展示历史对话（只放用户问题/简短状态，不把整篇报告塞进聊天气泡）
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

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

            # 启动后台规划任务（支持停止）
            # 若已有规划任务在跑，先尝试取消（协作式）
            old_pj = st.session_state.get("plan_job")
            try:
                if isinstance(old_pj, _PlanJob) and old_pj.status == "running":
                    old_pj.cancel_event.set()
            except Exception:
                pass

            # Demo 门禁：不允许 get_llm_client 从环境变量偷拿 OPENAI_API_KEY
            llm_probe = get_llm_client(api_key=chat_api_key, base_url=chat_base_url, allow_env_fallback=False)
            if not llm_probe:
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
                            stats["decision_counts"] = df["decision"].fillna("UNKNOWN").value_counts().head(10).to_dict()
                    except Exception:
                        pass

                pj = _PlanJob(job_id=str(uuid.uuid4())[:8], query=str(user_query))
                st.session_state["plan_job"] = pj
                th = threading.Thread(
                    target=_run_plan_job,
                    kwargs={
                        "job": pj,
                        "user_query": str(user_query),
                        "stats": stats,
                        "chat_api_key": chat_api_key,
                        "chat_base_url": chat_base_url,
                        "model_name": model_name,
                    },
                    daemon=True,
                )
                pj.thread = th
                th.start()
                _rerun()
                return

        # 规划任务面板（运行中/已完成/已取消/失败）
        pj = st.session_state.get("plan_job")
        if isinstance(pj, _PlanJob):
            with pj.lock:
                snap = {
                    "job_id": pj.job_id,
                    "status": pj.status,
                    "stage": pj.stage,
                    "message": pj.message,
                    "result": dict(pj.result),
                    "error": pj.error,
                    "error_trace": pj.error_trace,
                }

            if snap["status"] == "running":
                with st.status("正在规划（后台任务）...", expanded=True):
                    st.write(snap.get("message") or "生成研究计划（Plan）...")
                    c_stop, c_refresh, c_hint = st.columns([1, 1, 3])
                    with c_stop:
                        if st.button("⏹ 停止规划", key=f"plan_stop_{snap['job_id']}"):
                            try:
                                pj.cancel_event.set()
                                _job_update(pj, message="正在停止...（等待当前请求返回）")
                            except Exception:
                                pass
                            _rerun()
                    with c_refresh:
                        if st.button("🔄 刷新进度", key=f"plan_refresh_{snap['job_id']}"):
                            _rerun()
                    with c_hint:
                        st.caption("停止为协作式：LLM 单次请求进行中无法强制中断，但会在下一检查点尽快退出。")

            elif snap["status"] == "done":
                plan = (snap.get("result") or {}).get("plan")
                applied = bool((snap.get("result") or {}).get("_applied"))
                if not applied and isinstance(plan, dict):
                    st.session_state["pending_plan"] = plan
                    st.session_state["plan_editor_text"] = json.dumps(plan, ensure_ascii=False, indent=2)
                    if plan.get("_error") or plan.get("title") == "Error in Planning":
                        err = plan.get("_error") or "unknown"
                        st.session_state["plan_flash_error"] = (
                            "Planner 生成计划失败。常见原因：模型不支持 JSON mode（例如部分 GLM 会报 code=20024），"
                            "或 Model Name/Base URL 不匹配。\n\n"
                            f"错误信息：{err}"
                        )
                    else:
                        st.session_state["plan_flash"] = "计划已生成（等待你确认/编辑）。"
                    try:
                        with pj.lock:
                            pj.result["_applied"] = True
                    except Exception:
                        pass
                    _rerun()
                else:
                    st.success("规划完成 ✅")
                    if st.button("清除规划状态", key=f"plan_clear_{snap['job_id']}"):
                        st.session_state.pop("plan_job", None)
                        _rerun()

            elif snap["status"] == "cancelled":
                st.warning("规划已停止（Cancelled）")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("清除规划状态", key=f"plan_clear_{snap['job_id']}"):
                        st.session_state.pop("plan_job", None)
                        _rerun()
                with c2:
                    if st.button("重新规划（同一问题）", key=f"plan_retry_{snap['job_id']}"):
                        st.session_state["pending_user_query"] = str(getattr(pj, "query", "") or "")
                        _rerun()

            elif snap["status"] == "error":
                st.error(f"规划失败：{snap.get('error')}")
                with st.expander("错误详情（Traceback）", expanded=False):
                    st.code(str(snap.get("error_trace") or ""), language="text")
                if st.button("清除规划状态", key=f"plan_clear_{snap['job_id']}"):
                    st.session_state.pop("plan_job", None)
                    _rerun()

        # 计划审核/编辑/批准
        if st.session_state.get("pending_plan") and not st.session_state.get("plan_approved"):
            st.subheader("第 1 步：确认 / 编辑研究计划")

            # flash
            flash = st.session_state.pop("plan_flash", None)
            if isinstance(flash, str) and flash.strip():
                st.success(flash.strip())
            flash_err = st.session_state.pop("plan_flash_error", None)
            if isinstance(flash_err, str) and flash_err.strip():
                st.error(flash_err.strip())

            plan = st.session_state.get("pending_plan")
            if not isinstance(plan, dict):
                st.error("计划格式异常：pending_plan 不是 JSON object。")
            else:
                sections = plan.get("sections") or []
                if not isinstance(sections, list):
                    sections = []
                    plan["sections"] = sections
                _ensure_plan_section_uids(len(sections))

                tab_read, tab_json = st.tabs(["可读版（推荐）", "JSON（高级）"])

                with tab_read:
                    st.caption("你可以在这里像写表单一样改计划；底层仍会同步成 JSON。")
                    with st.expander("计划预览（自然语言）", expanded=True):
                        st.markdown(_plan_to_markdown(plan))

                    st.markdown("#### 编辑（可读版）")
                    st.session_state.setdefault("plan_edit_title", str(plan.get("title") or ""))
                    st.text_input("报告标题", key="plan_edit_title")
                    st.number_input(
                        "预计使用论文数（可选，不填则按各节 top_k 估算）",
                        min_value=0,
                        max_value=500,
                        value=int(plan.get("estimated_papers") or 0),
                        key="plan_edit_estimated_papers",
                    )

                    # 选项：尽量从库里拿候选
                    years_opts = list(range(2018, 2026))
                    decision_opts = ["Accept (oral)", "Accept (spotlight)", "Accept (poster)", "Accept"]
                    try:
                        df_all = kb.search_structured()
                        if hasattr(df_all, "empty") and (not df_all.empty):
                            if "year" in df_all.columns:
                                ys = sorted({int(x) for x in df_all["year"].dropna().tolist() if int(x) > 0})
                                if ys:
                                    years_opts = ys
                            if "decision" in df_all.columns:
                                ds = [str(x) for x in df_all["decision"].dropna().tolist() if str(x).strip()]
                                if ds:
                                    # 保留常见项 + 去重
                                    merged = decision_opts + sorted({x for x in ds})
                                    seen = set()
                                    decision_opts = [x for x in merged if not (x in seen or seen.add(x))]
                    except Exception:
                        pass

                    st.button("➕ 添加章节", on_click=_plan_add_section, **_width_kwargs(st.button, stretch=True))

                    uids = st.session_state.get("plan_section_uids") or []
                    for i, (sec, uid) in enumerate(zip(sections, uids)):
                        if not isinstance(sec, dict):
                            continue
                        uid = str(uid)
                        sec_name = str(sec.get("name") or "").strip() or f"第 {i+1} 节"
                        with st.expander(f"第 {i+1} 节：{sec_name}", expanded=False):
                            st.text_input("章节名称", value=str(sec.get("name") or ""), key=f"plan_sec_name_{uid}")
                            st.text_area(
                                "检索 query（search_query）",
                                value=str(sec.get("search_query") or ""),
                                key=f"plan_sec_query_{uid}",
                                height=90,
                            )
                            st.number_input(
                                "top_k_papers（本节最多选多少篇论文）",
                                min_value=1,
                                max_value=50,
                                value=int(sec.get("top_k_papers") or 5),
                                key=f"plan_sec_topk_{uid}",
                            )

                            f = sec.get("filters") or {}
                            if not isinstance(f, dict):
                                f = {}

                            st.markdown("**筛选条件（filters）**")
                            st.multiselect(
                                "year_in（年份）",
                                options=years_opts,
                                default=[int(x) for x in (f.get("year_in") or []) if isinstance(x, int)],
                                key=f"plan_sec_year_in_{uid}",
                            )
                            st.text_input(
                                "venue_contains（会议信息包含）",
                                value=str(f.get("venue_contains") or ""),
                                key=f"plan_sec_venue_contains_{uid}",
                            )
                            st.text_input(
                                "title_contains（标题包含）",
                                value=str(f.get("title_contains") or ""),
                                key=f"plan_sec_title_contains_{uid}",
                            )
                            st.text_input(
                                "author_contains（作者包含）",
                                value=str(f.get("author_contains") or ""),
                                key=f"plan_sec_author_contains_{uid}",
                            )
                            st.text_input(
                                "keyword_contains（关键词包含）",
                                value=str(f.get("keyword_contains") or ""),
                                key=f"plan_sec_keyword_contains_{uid}",
                            )
                            st.multiselect(
                                "decision_in（录用决策）",
                                options=decision_opts,
                                default=[str(x) for x in (f.get("decision_in") or []) if str(x).strip()],
                                key=f"plan_sec_decision_in_{uid}",
                            )
                            st.multiselect(
                                "presentation_in（展示类型，oral/spotlight/poster/unknown）",
                                options=["oral", "spotlight", "poster", "unknown"],
                                default=[str(x).strip().lower() for x in (f.get("presentation_in") or []) if str(x).strip()],
                                key=f"plan_sec_presentation_in_{uid}",
                            )
                            st.text_input(
                                "min_rating（最低评分，可选）",
                                value=str(f.get("min_rating") or ""),
                                key=f"plan_sec_min_rating_{uid}",
                                help="留空表示不限制；例如 7.5",
                            )

                            st.button(
                                "🗑 删除该章节",
                                key=f"plan_sec_del_btn_{uid}",
                                on_click=_plan_delete_section,
                                args=(uid,),
                            )

                    c1, c2 = st.columns(2)
                    with c1:
                        st.button("应用修改（同步到 JSON）", on_click=_plan_apply_readable)
                    with c2:
                        st.button("确认并运行", type="primary", on_click=_plan_run_from_readable)

                with tab_json:
                    st.caption("高级模式：直接编辑 JSON。编辑后请点「从 JSON 覆盖可读版」或「确认并运行（使用 JSON）」")
                    st.text_area("计划（JSON）", key="plan_editor_text", height=360)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.button("从 JSON 覆盖可读版", on_click=_plan_apply_json)
                    with c2:
                        st.button("确认并运行（使用 JSON）", type="primary", on_click=_plan_run_from_json)

            # 如果用户点了“确认并运行”，在这里统一执行（避免在 on_click 里跑长任务）
            run_req = st.session_state.pop("plan_run_requested", None)
            # 研究任务：改为后台线程执行（支持停止）
            job = st.session_state.get("research_job")

            # 1) 启动任务（点击“确认并运行”触发）
            if run_req:
                if isinstance(job, _ResearchJob) and job.status == "running":
                    st.warning("已有任务正在运行。请先停止或等待完成。")
                else:
                    # 先做一次轻量鉴权检查（避免开线程后立即失败）
                    llm_probe = get_llm_client(api_key=chat_api_key, base_url=chat_base_url, allow_env_fallback=False)
                    if not llm_probe:
                        st.error("Authentication Failed. Please provide a valid Access Code or your own API Key.")
                    else:
                        plan = st.session_state.get("pending_plan")
                        if not isinstance(plan, dict):
                            st.error("计划格式异常：pending_plan 不是 JSON object。")
                        else:
                            st.session_state["plan_approved"] = True

                            # 清空上一次结果（与旧行为一致）
                            st.session_state["research_notes"] = []
                            st.session_state["final_report"] = ""
                            st.session_state["verification_result"] = None
                            st.session_state["report_ref_ctx"] = None
                            st.session_state["writer_stats"] = None

                            job = _ResearchJob(job_id=str(uuid.uuid4())[:8])
                            st.session_state["research_job"] = job

                            # 深拷贝 plan，避免 UI 编辑影响后台线程
                            try:
                                plan_copy = json.loads(json.dumps(plan, ensure_ascii=False))
                            except Exception:
                                plan_copy = dict(plan)

                            th = threading.Thread(
                                target=_run_research_job,
                                kwargs={
                                    "job": job,
                                    "plan": plan_copy,
                                    "chat_api_key": chat_api_key,
                                    "chat_base_url": chat_base_url,
                                    "model_name": model_name,
                                    "embedding_model": embedding_model,
                                    "embedding_api_key": embedding_api_key,
                                    "embedding_base_url": embedding_base_url,
                                },
                                daemon=True,
                            )
                            job.thread = th
                            th.start()
                            _rerun()

            # 任务面板在外层统一渲染（保证 plan_approved=True 后也能看到进度/停止按钮）

        # 输出最终报告（左栏）
        # 运行中任务面板（无论 plan 是否已批准，都显示）
        job = st.session_state.get("research_job")
        if isinstance(job, _ResearchJob):
            with job.lock:
                snap = {
                    "job_id": job.job_id,
                    "status": job.status,
                    "stage": job.stage,
                    "message": job.message,
                    "progress": dict(job.progress),
                    "result": dict(job.result),
                    "error": job.error,
                    "error_trace": job.error_trace,
                    "started_ts": job.started_ts,
                    "finished_ts": job.finished_ts,
                }

            if snap["status"] == "running":
                with st.status("正在执行（后台任务）...", expanded=True):
                    st.write(snap.get("message") or "运行中...")

                    # research 进度
                    rp = snap.get("progress", {}).get("research")
                    if isinstance(rp, dict):
                        cur = int(rp.get("current") or 0)
                        tot = int(rp.get("total") or 0)
                        sec = str(rp.get("section") or "")
                        q = str(rp.get("query") or "")
                        if tot > 0:
                            pct = int(cur * 100 / tot)
                            st.progress(min(100, max(0, pct)))
                            st.caption(f"Research：{cur}/{tot} · {sec} · {q[:60]}")
                        else:
                            # 部分阶段尚未提供 total（或 total=0），先给一个占位进度条
                            st.progress(0)
                            st.caption("Research：准备中…（点「刷新进度」查看更新）")
                    else:
                        st.progress(0)
                        st.caption("Research：准备中…（点「刷新进度」查看更新）")

                    # write 进度（文本型）
                    wp = snap.get("progress", {}).get("write")
                    if isinstance(wp, dict):
                        stg = wp.get("stage")
                        if stg == "write_refs_built":
                            st.caption(f"Write：写作准备 refs={wp.get('refs_total')}")
                        elif stg == "write_payload_built":
                            st.caption(
                                f"Write：sections={wp.get('sections')} · evidence={wp.get('evidence_snippets')} · refs={wp.get('allowed_refs_total')}"
                            )
                        elif stg == "write_llm_call":
                            st.caption(f"Write：LLM 生成中 model={wp.get('model')}")

                    c_stop, c_refresh, c_hint = st.columns([1, 1, 3])
                    with c_stop:
                        if st.button("⏹ 停止本次运行", key=f"job_stop_{snap['job_id']}"):
                            try:
                                job.cancel_event.set()
                                _job_update(job, message="正在停止...（等待当前请求返回）")
                            except Exception:
                                pass
                            _rerun()
                    with c_refresh:
                        if st.button("🔄 刷新进度", key=f"job_refresh_{snap['job_id']}"):
                            _rerun()
                    with c_hint:
                        st.caption(
                            "停止为协作式：LLM 单次请求进行中无法强制中断，但会在下一检查点尽快退出。"
                            "（页面不会自动刷新，点「刷新进度」即可更新）"
                        )

            elif snap["status"] == "done":
                st.success("任务完成 ✅")

                # 将结果回填到 session_state（只做一次，避免重复追加消息）
                applied = bool(snap.get("result", {}).get("_applied"))
                if not applied:
                    res = snap.get("result", {}) or {}
                    st.session_state["research_notes"] = res.get("research_notes") or []
                    st.session_state["final_report"] = str(res.get("final_report") or "")
                    st.session_state["report_ref_ctx"] = res.get("report_ref_ctx")
                    st.session_state["writer_stats"] = res.get("writer_stats")
                    st.session_state["verification_result"] = res.get("verification_result")

                    v = st.session_state.get("verification_result") or {}
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": f"报告已生成。核查：valid={v.get('is_valid')}, score={v.get('score')}.（详见右侧溯源/核查面板）",
                        }
                    )
                    try:
                        with job.lock:
                            job.result["_applied"] = True
                    except Exception:
                        pass

            elif snap["status"] == "cancelled":
                st.warning("任务已停止（Cancelled）")
                if snap.get("error"):
                    st.caption(str(snap.get("error")))

            elif snap["status"] == "error":
                st.error(f"任务失败：{snap.get('error')}")
                with st.expander("错误详情（Traceback）", expanded=False):
                    st.code(str(snap.get("error_trace") or ""), language="text")

            if snap["status"] in {"done", "cancelled", "error"}:
                if snap["status"] in {"cancelled", "error"}:
                    if st.button("返回计划编辑", key=f"job_back_plan_{snap['job_id']}"):
                        st.session_state["plan_approved"] = False
                        _rerun()
                if st.button("清除任务状态", key=f"job_clear_{snap['job_id']}"):
                    st.session_state.pop("research_job", None)
                    _rerun()

        if st.session_state.get("final_report"):
            st.divider()
            st.subheader("最终报告")

            report_md = str(st.session_state.get("final_report") or "")
            c_dl1, c_dl2 = st.columns([1, 3])
            with c_dl1:
                st.download_button(
                    "⬇️ 下载报告（.md）",
                    data=report_md.encode("utf-8"),
                    file_name="mujica_report.md",
                    mime="text/markdown",
                )
            with c_dl2:
                show_raw = st.checkbox("显示 Markdown 源码", value=False, key="show_report_raw")

            if show_raw:
                st.code(report_md, language="markdown")
            else:
                st.markdown(report_md)

            v = st.session_state.get("verification_result")
            if isinstance(v, dict) and v:
                st.caption(f"Verification: valid={v.get('is_valid')} · score={v.get('score')} · {v.get('notes')}")

    with col_context:
        # 浮动窗口：看最终报告时也能随时看到核查/证据（右栏内部滚动）
        float_default = bool(st.session_state.get("final_report"))
        float_panel = st.checkbox(
            "浮动窗口：证据与核查（看报告时保持可见）",
            value=bool(st.session_state.get("float_evidence_panel", float_default)),
            key="float_evidence_panel",
            help="开启后右侧面板会变成粘性窗口，并在内部滚动；适合边看最终报告边对照核查。",
        )

        if float_panel:
            st.markdown('<div class="mujica-float-wrap"><div class="mujica-float-card">', unsafe_allow_html=True)

        st.subheader("证据与核查")

        tab_evi, tab_ver = st.tabs(["Evidence（证据）", "Verification（核查）"])

        with tab_evi:
            notes = st.session_state.get("research_notes") or []
            if not notes:
                # 更具体的空态引导：告诉用户“证据是什么 + 下一步怎么做”
                st.info(
                    "暂无证据片段。证据会在你点击「确认并运行」后生成：来自论文的摘要/正文/评审/决策/作者回复等文本片段，"
                    "并在报告里以引用 [R#] 形式可溯源。",
                    icon="ℹ️",
                )
                try:
                    st.caption(f"当前知识库：papers={kb_papers} · chunks={chunks_rows}")
                except Exception:
                    pass

                c_go1, c_go2 = st.columns(2)
                with c_go1:
                    if st.button("📚 去知识库入库/管理数据", key="evi_go_data"):
                        _set_system_mode("data")
                        _rerun()
                with c_go2:
                    if st.button("🧪 一键导入样例数据", key="evi_ingest_samples"):
                        with st.spinner("正在导入样例数据..."):
                            _ingest_test_dataset(kb)
                        st.session_state["kb_flash"] = "已导入样例数据。现在可以回到首页提问并运行。"
                        _set_system_mode("research")
                        _rerun()

                st.markdown(
                    "**建议步骤**：\n"
                    "1) 在「📚 知识库」页抓取 OpenReview 或导入样例 →\n"
                    "2) 回到首页输入研究问题 →\n"
                    "3) 审核计划后点「确认并运行」，这里就会显示证据。"
                )
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
                            ref_ctx = st.session_state.get("report_ref_ctx") or {}
                            chunk_to_ref = {}
                            try:
                                chunk_to_ref = (ref_ctx or {}).get("chunk_to_ref") or {}
                            except Exception:
                                chunk_to_ref = {}
                            for e in evidence:
                                pid = e.get("paper_id")
                                title = e.get("title", "")
                                cid = e.get("chunk_id")
                                src = e.get("source")
                                rid = chunk_to_ref.get(cid)
                                rid_disp = f"`ref={rid}` · " if rid else ""
                                st.markdown(
                                    f"**{title}**  \n{rid_disp}`paper_id={pid}` · `chunk_id={cid}` · `source={src}`"
                                )
                                st.code((e.get("text") or "")[:1200])

        with tab_ver:
            v = st.session_state.get("verification_result")
            if not isinstance(v, dict) or not v:
                st.info("暂无核查结果。生成报告后会自动触发核查。", icon="ℹ️")
            else:
                evals = v.get("evaluations") or []

                # 汇总信息（更直观）
                try:
                    checked = int((v.get("stats") or {}).get("claims_checked") or 0)
                except Exception:
                    checked = 0
                if not checked and isinstance(evals, list):
                    checked = len(evals)

                supports = 0
                contradicts = 0
                unknowns = 0
                for it in (evals or []):
                    lbl = str((it or {}).get("label") or "unknown").lower().strip()
                    if lbl == "entailed":
                        supports += 1
                    elif lbl == "contradicted":
                        contradicts += 1
                    else:
                        unknowns += 1

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("valid", bool(v.get("is_valid")))
                try:
                    c2.metric("score", f"{float(v.get('score') or 0.0):.2f}")
                except Exception:
                    c2.metric("score", str(v.get("score")))
                c3.metric("checked", int(checked))
                c4.metric("contradicts", int(contradicts))
                st.caption(str(v.get("notes") or "").strip())

                if evals:
                    try:
                        import pandas as pd

                        # 表格美化：只保留关键信息，避免 citations 显示为 [object Object]
                        ref_ctx = st.session_state.get("report_ref_ctx") or {}
                        chunk_to_ref = {}
                        try:
                            chunk_to_ref = (ref_ctx or {}).get("chunk_to_ref") or {}
                        except Exception:
                            chunk_to_ref = {}

                        def _label_zh(lbl: str) -> str:
                            s = (lbl or "").lower().strip()
                            if s == "entailed":
                                return "支持"
                            if s == "contradicted":
                                return "矛盾"
                            return "不确定"

                        def _format_citations(cits: Any) -> str:
                            if not isinstance(cits, list) or not cits:
                                return ""
                            refs = []
                            for c in cits:
                                if not isinstance(c, dict):
                                    continue
                                r = str(c.get("ref") or "").strip()
                                if r:
                                    refs.append(r)
                                    continue
                                cid = str(c.get("chunk_id") or "").strip()
                                rid = chunk_to_ref.get(cid) if isinstance(chunk_to_ref, dict) else None
                                if rid:
                                    refs.append(str(rid))
                            # 去重、限长
                            out = []
                            seen = set()
                            for r in refs:
                                if r in seen:
                                    continue
                                seen.add(r)
                                out.append(r)
                            if out:
                                if len(out) > 5:
                                    return ", ".join(out[:5]) + f" (+{len(out)-5})"
                                return ", ".join(out)
                            # fallback：只显示数量
                            return f"{len(cits)} 条引用"

                        rows: List[Dict[str, Any]] = []
                        for i, it in enumerate(evals):
                            if not isinstance(it, dict):
                                continue
                            claim = str(it.get("claim") or "").strip()
                            claim_short = claim
                            if len(claim_short) > 160:
                                claim_short = claim_short[:160].rstrip() + "…"
                            lbl_raw = str(it.get("label") or "unknown")
                            try:
                                sc = float(it.get("score") or 0.0)
                            except Exception:
                                sc = 0.0
                            cits = it.get("citations") or []
                            rows.append(
                                {
                                    "序号": i + 1,
                                    "结论": _label_zh(lbl_raw),
                                    "分数": round(sc, 2),
                                    "引用": _format_citations(cits),
                                    "要点（claim）": claim_short,
                                }
                            )

                        df = pd.DataFrame(rows)
                        st.dataframe(df, **_width_kwargs(st.dataframe, stretch=True))

                        with st.expander("查看核查明细（原始 JSON）", expanded=False):
                            st.json(evals, expanded=False)
                    except Exception:
                        st.json(evals, expanded=False)
                else:
                    st.json(v, expanded=False)

        if float_panel:
            st.markdown("</div></div>", unsafe_allow_html=True)

    # Chat 输入框必须位于页面根容器（不能在 columns/tabs/sidebar/expander/form 内）
    prompt = st.chat_input("输入你的研究问题（按 Enter 发送）")
    if prompt:
        if not has_auth:
            st.warning(
                "未配置鉴权：请先在左侧栏填写 **API Key** 或输入正确 **Access Code**，否则无法开始生成。",
                icon="🔑",
            )
        else:
            st.session_state["pending_user_query"] = prompt
            _rerun()


def main() -> None:
    load_env()

    if not _ensure_streamlit_context():
        print("这是一个 Streamlit 应用，请使用：streamlit run ui/app.py")
        return

    st.set_page_config(
        page_title="MUJICA Deep Insight",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Session defaults（先设默认，再注入 CSS，避免首帧主题错乱）
    st.session_state.setdefault("ui_theme", "light")
    st.session_state.setdefault("system_mode", "research")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("research_notes", [])
    st.session_state.setdefault("final_report", "")
    st.session_state.setdefault("report_ref_ctx", None)
    st.session_state.setdefault("writer_stats", None)
    st.session_state.setdefault("pending_plan", None)
    st.session_state.setdefault("plan_editor_text", "")
    st.session_state.setdefault("plan_approved", False)
    st.session_state.setdefault("verification_result", None)

    # 对话历史（默认开启；不提供 UI 开关）
    # 如需关闭（例如 HF Spaces 多人 demo 避免互相可见），可设置：MUJICA_DISABLE_CHAT_HISTORY=1
    disable_hist = (os.getenv("MUJICA_DISABLE_CHAT_HISTORY") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
    st.session_state.setdefault("enable_chat_history", not disable_hist)
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("conversation_title", "")
    st.session_state.setdefault("history_loaded", False)
    st.session_state.setdefault("history_created_ts", time.time())
    st.session_state.setdefault("history_last_hash", "")

    # 处理 URL query params：
    # - cid：用于刷新/重开后恢复当前对话
    # - go=home：点击左上角 MUJICA 回到首页（清空工作区）
    qp = _get_query_params()
    go = (qp.get("go") or [None])[0]
    cid = (qp.get("cid") or [None])[0]

    if str(go or "").lower() == "home":
        _reset_workspace_state(cancel_running_job=True)
        # 清理 go 参数，保留 cid（若有）
        _set_query_params(cid=cid or st.session_state.get("conversation_id"))
        _rerun()
        return

    if st.session_state.get("enable_chat_history"):
        # 恢复/初始化 conversation_id
        if cid and not st.session_state.get("conversation_id"):
            st.session_state["conversation_id"] = str(cid)
        if not st.session_state.get("conversation_id"):
            st.session_state["conversation_id"] = new_conversation_id()
            _set_query_params(cid=st.session_state["conversation_id"])
        # 首次加载：如果有历史文件则恢复
        if (not st.session_state.get("history_loaded")) and st.session_state.get("conversation_id"):
            snap = load_conversation(str(st.session_state.get("conversation_id") or ""))
            if isinstance(snap, dict) and snap:
                # 只恢复“工作区相关状态”，不覆盖模型配置/鉴权字段
                try:
                    st.session_state["conversation_title"] = str(snap.get("title") or "").strip()
                except Exception:
                    st.session_state["conversation_title"] = ""
                st.session_state["messages"] = snap.get("messages") or []
                st.session_state["research_notes"] = snap.get("research_notes") or []
                st.session_state["final_report"] = str(snap.get("final_report") or "")
                st.session_state["report_ref_ctx"] = snap.get("report_ref_ctx")
                st.session_state["writer_stats"] = snap.get("writer_stats")
                st.session_state["pending_plan"] = snap.get("pending_plan")
                st.session_state["plan_editor_text"] = str(snap.get("plan_editor_text") or "")
                st.session_state["plan_approved"] = bool(snap.get("plan_approved"))
                st.session_state["verification_result"] = snap.get("verification_result")
                # 轻量恢复 UI 外观/导航
                if snap.get("system_mode") in {"research", "data"}:
                    st.session_state["system_mode"] = snap.get("system_mode")
                if snap.get("ui_theme") in {"light", "dark"}:
                    st.session_state["ui_theme"] = snap.get("ui_theme")
                try:
                    st.session_state["history_created_ts"] = float(snap.get("created_ts") or time.time())
                except Exception:
                    st.session_state["history_created_ts"] = time.time()
            st.session_state["history_loaded"] = True

    _local_css(Path(__file__).with_name("style.css"))
    _apply_theme_vars(st.session_state.get("ui_theme"))

    with st.sidebar:
        # 点击品牌回首页（通过 query param 触发 reset）
        st.markdown('<a class="mujica-brand-link" href="?go=home">MUJICA</a>', unsafe_allow_html=True)
        st.caption("Multi-stage User-Judged Integration")

        st.divider()
        st.subheader("界面")
        st.radio(
            "主题",
            options=["light", "dark"],
            key="ui_theme",
            horizontal=True,
            format_func=lambda x: "简明" if x == "light" else "MUJICA",
        )
        st.radio(
            "导航",
            options=["research", "data"],
            key="system_mode",
            format_func=lambda x: "🏠 首页" if x == "research" else "📚 知识库",
        )

        st.divider()
        st.subheader("运行控制")
        
        @st.fragment(run_every="0.8s")
        def _job_control_fragment():
            """独立刷新的 Fragment：显示 Plan/Research 任务进度"""
            pj = st.session_state.get("plan_job")
            if isinstance(pj, _PlanJob) and pj.status == "running":
                st.caption(f"规划中：{str(getattr(pj, 'query', '') or '')[:60]}")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("⏹ 停止规划", key=f"sb_stop_plan_{pj.job_id}"):
                        try:
                            pj.cancel_event.set()
                            _job_update(pj, message="正在停止规划...（等待当前请求返回）")
                        except Exception:
                            pass
                with c2:
                    pass  # 自动刷新，无需手动刷新按钮
            else:
                st.caption("规划：无")

            rj = st.session_state.get("research_job")
            if isinstance(rj, _ResearchJob) and rj.status == "running":
                # 在侧边栏展示进度（更容易找到"进度条"）
                try:
                    with rj.lock:
                        rj_stage = str(rj.stage or "")
                        rj_msg = str(rj.message or "")
                        rj_prog = dict(rj.progress or {})
                except Exception:
                    rj_stage = str(getattr(rj, "stage", "") or "")
                    rj_msg = str(getattr(rj, "message", "") or "")
                    rj_prog = {}

                st.caption(f"研究运行中：{rj_stage or 'running'}")
                if rj_msg.strip():
                    st.caption(rj_msg.strip())

                rp = rj_prog.get("research")
                if isinstance(rp, dict):
                    cur = int(rp.get("current") or 0)
                    tot = int(rp.get("total") or 0)
                    sec = str(rp.get("section") or "")
                    if tot > 0:
                        pct = int(cur * 100 / tot)
                        st.progress(min(100, max(0, pct)))
                        st.caption(f"Research：{cur}/{tot} · {sec}")
                    else:
                        st.progress(0)
                else:
                    st.progress(0)

                try:
                    ts = float(rj_prog.get("_ts") or 0.0)
                    if ts > 0:
                        st.caption(f"最后更新：{time.time() - ts:.1f}s 前")
                except Exception:
                    pass
                
                if st.button("⏹ 停止运行", key=f"sb_stop_run_{rj.job_id}"):
                    try:
                        rj.cancel_event.set()
                        _job_update(rj, message="正在停止...（等待当前请求返回）")
                    except Exception:
                        pass
            else:
                st.caption("运行：无")
        
        # 调用 Fragment
        _job_control_fragment()

        st.divider()
        st.subheader("对话")
        if not st.session_state.get("enable_chat_history"):
            st.caption("历史对话已关闭（设置 MUJICA_DISABLE_CHAT_HISTORY=1）。")
        else:
            cid_now = str(st.session_state.get("conversation_id") or "")
            items = list_conversations(limit=60)

            # 确保当前 cid 在列表里（新对话尚未写盘时）
            if cid_now and (not any((it or {}).get("cid") == cid_now for it in items)):
                cur_title = str(st.session_state.get("conversation_title") or "").strip() or "（当前对话）"
                items = [{"cid": cid_now, "title": cur_title, "updated_ts": time.time()}] + items

            if st.button("➕ 新聊天", key="history_new_chat", **_width_kwargs(st.button, stretch=True)):
                _reset_workspace_state(cancel_running_job=True)
                st.session_state["conversation_id"] = new_conversation_id()
                st.session_state["conversation_title"] = ""
                st.session_state["history_loaded"] = True  # 新对话无需加载
                st.session_state["history_created_ts"] = time.time()
                st.session_state.pop("history_menu_cid", None)
                st.session_state.pop("history_rename_cid", None)
                st.session_state.pop("history_delete_cid", None)
                _set_query_params(cid=st.session_state["conversation_id"])
                _rerun()

            # ChatGPT 风格：列表 + 省略号菜单（重命名/删除）
            for it in items:
                if not isinstance(it, dict):
                    continue
                cid_it = str(it.get("cid") or "").strip()
                if not cid_it:
                    continue
                title = str(it.get("title") or "未命名对话").strip() or "未命名对话"
                title_disp = title if len(title) <= 28 else (title[:28].rstrip() + "…")
                is_current = cid_it == cid_now

                col_t, col_m = st.columns([0.86, 0.14])
                with col_t:
                    label = f"● {title_disp}" if is_current else title_disp
                    if st.button(label, key=f"hist_open_{cid_it}"):
                        if cid_it != cid_now:
                            _reset_workspace_state(cancel_running_job=True)
                            st.session_state["conversation_id"] = cid_it
                            st.session_state["conversation_title"] = ""
                            st.session_state["history_loaded"] = False  # 触发加载
                            st.session_state.pop("history_menu_cid", None)
                            st.session_state.pop("history_rename_cid", None)
                            st.session_state.pop("history_delete_cid", None)
                            _set_query_params(cid=cid_it)
                            _rerun()
                with col_m:
                    if st.button("⋯", key=f"hist_menu_{cid_it}"):
                        cur = str(st.session_state.get("history_menu_cid") or "")
                        st.session_state["history_menu_cid"] = None if cur == cid_it else cid_it
                        st.session_state.pop("history_rename_cid", None)
                        st.session_state.pop("history_delete_cid", None)
                        _rerun()

                if str(st.session_state.get("history_menu_cid") or "") == cid_it:
                    a1, a2 = st.columns(2)
                    with a1:
                        if st.button("✏️ 重命名", key=f"hist_act_rename_{cid_it}"):
                            st.session_state["history_rename_cid"] = cid_it
                            st.session_state[f"hist_rename_text_{cid_it}"] = title
                            _rerun()
                    with a2:
                        if st.button("🗑 删除", key=f"hist_act_delete_{cid_it}"):
                            st.session_state["history_delete_cid"] = cid_it
                            _rerun()

                    if str(st.session_state.get("history_rename_cid") or "") == cid_it:
                        new_t = st.text_input("新名称", key=f"hist_rename_text_{cid_it}")
                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button("保存", key=f"hist_rename_save_{cid_it}"):
                                res = rename_conversation(cid_it, new_t)
                                if isinstance(res, dict) and res.get("ok"):
                                    if cid_it == cid_now:
                                        st.session_state["conversation_title"] = str(new_t or "").strip()
                                    st.session_state["history_menu_cid"] = None
                                    st.session_state.pop("history_rename_cid", None)
                                    _rerun()
                                else:
                                    st.error(f"重命名失败：{res.get('error') if isinstance(res, dict) else res}")
                        with b2:
                            if st.button("取消", key=f"hist_rename_cancel_{cid_it}"):
                                st.session_state.pop("history_rename_cid", None)
                                _rerun()

                    if str(st.session_state.get("history_delete_cid") or "") == cid_it:
                        confirm = st.checkbox("确认删除该对话", key=f"hist_delete_confirm_{cid_it}")
                        if st.button("确认删除", key=f"hist_delete_do_{cid_it}", disabled=not bool(confirm)):
                            delete_conversation(cid_it)
                            st.session_state["history_menu_cid"] = None
                            st.session_state.pop("history_delete_cid", None)

                            # 删除当前对话：自动新建一个空对话，避免 UI 处于无 cid 状态
                            if cid_it == cid_now:
                                _reset_workspace_state(cancel_running_job=True)
                                st.session_state["conversation_id"] = new_conversation_id()
                                st.session_state["conversation_title"] = ""
                                st.session_state["history_loaded"] = True
                                st.session_state["history_created_ts"] = time.time()
                                _set_query_params(cid=st.session_state["conversation_id"])
                            _rerun()

        st.divider()
        st.subheader("模型配置")

        def _clear_auth_code() -> None:
            st.session_state["auth_code"] = ""

        SYSTEM_ACCESS_CODE = os.getenv("MUJICA_ACCESS_CODE", "mujica2024")
        auth_code = st.text_input(
            "Access Code（可选）",
            type="password",
            key="auth_code",
            help="输入正确的 Access Code 后，将使用系统环境变量中的 OPENAI_API_KEY。",
        )

        use_system_key = False
        if (auth_code or "") == SYSTEM_ACCESS_CODE:
            use_system_key = True
            st.success("Authentication: Authorized ✅（出于安全，Access Code 不回显；刷新后仍可能保持授权）")
            st.button("退出授权 / 更换 Access Code", on_click=_clear_auth_code)
        elif auth_code:
            st.error("Authentication: Invalid Code ❌")

        user_api_key = st.text_input(
            "API Key（必填其一）",
            type="password",
            key="chat_api_key",
            disabled=use_system_key,
            help="未使用 Access Code 时必须填写",
        )
        user_base_url = st.text_input(
            "Base URL（可选）",
            key="chat_base_url",
            placeholder="例如：https://api.deepseek.com/v1",
        )
        model_name = st.text_input(
            "Model Name",
            key="chat_model_name",
            value=os.getenv("MUJICA_DEFAULT_MODEL", "gpt-4o"),
            help="例如：gpt-4o / deepseek-chat / glm-4.6v（需与你的 Base URL 服务匹配）",
        )
        disable_json_mode = st.checkbox(
            "兼容模式：关闭 JSON mode（response_format）",
            value=("glm" in (model_name or "").lower()) or ((os.getenv("MUJICA_DISABLE_JSON_MODE") or "").strip().lower() in {"1","true","yes","y","on"}),
            help="部分 OpenAI-compatible 网关/模型不支持 response_format(JSON mode)，会报 code=20024。开启后将走“提示词输出 JSON + 解析”的方式。",
        )

        st.divider()
        st.subheader("向量检索（Embedding）")
        embedding_model = st.text_input(
            "Embedding Model",
            key="embedding_model",
            value=os.getenv("MUJICA_EMBEDDING_MODEL", "text-embedding-3-small"),
            help="用于向量化/语义检索的模型名（通常与聊天模型不同）。若这里配错，会出现“Model does not exist”。",
        )
        embedding_base_url_input = st.text_input(
            "Embedding Base URL（可选）",
            key="embedding_base_url",
            value=os.getenv("MUJICA_EMBEDDING_BASE_URL", ""),
            placeholder="留空则复用上面的 Base URL；SiliconFlow: https://api.siliconflow.cn/v1",
            disabled=False,
        )
        embedding_api_key_input = st.text_input(
            "Embedding API Key（可选）",
            type="password",
            key="embedding_api_key",
            help="留空则复用上面的 API Key（可用于把 Chat 与 Embedding 拆成不同服务商）",
            disabled=False,
        )
        use_fake_embeddings = st.checkbox(
            "离线 Embedding（不调用接口，仅用于跑通流程）",
            value=((os.getenv("MUJICA_FAKE_EMBEDDINGS") or "").strip().lower() in {"1", "true", "yes", "y", "on"}),
            help="当你的 Base URL 服务不支持 embeddings 或没有可用 embedding 模型时可打开；检索质量会明显下降。",
        )

        st.caption(f"System Status: {'Using System Key' if use_system_key else 'Using User Key'}")

    # 主题选择在 sidebar 中，变更会触发 rerun；这里再次注入，保证主题立即生效
    _apply_theme_vars(st.session_state.get("ui_theme"))

    # 统一计算“当前生效”的 Chat Key/BaseURL
    chat_api_key = os.getenv("OPENAI_API_KEY") if use_system_key else ((user_api_key or "").strip() or None)
    chat_base_url = os.getenv("OPENAI_BASE_URL", None) if use_system_key else ((user_base_url or "").strip() or None)

    # Embedding 可单独配置（优先 UI > .env > 复用 Chat）
    # Demo 门禁：未通过 Access Code 时，不允许使用环境变量中的系统 Embedding Key/BaseURL
    env_embed_key = ((os.getenv("MUJICA_EMBEDDING_API_KEY") or "").strip() or None) if use_system_key else None
    env_embed_base = ((os.getenv("MUJICA_EMBEDDING_BASE_URL") or "").strip() or None) if use_system_key else None
    embedding_api_key = (embedding_api_key_input or "").strip() or env_embed_key or chat_api_key
    embedding_base_url = (embedding_base_url_input or "").strip() or env_embed_base or chat_base_url

    # 把 embedding 配置同步到环境变量（供底层模块与测试场景复用）
    if (embedding_model or "").strip():
        os.environ["MUJICA_EMBEDDING_MODEL"] = embedding_model.strip()
    if use_fake_embeddings:
        os.environ["MUJICA_FAKE_EMBEDDINGS"] = "1"
    else:
        os.environ.pop("MUJICA_FAKE_EMBEDDINGS", None)

    # JSON mode 兼容开关（影响 Planner/Researcher/Verifier）
    if disable_json_mode:
        os.environ["MUJICA_DISABLE_JSON_MODE"] = "1"
    else:
        os.environ.pop("MUJICA_DISABLE_JSON_MODE", None)

    if st.session_state.get("system_mode") == "data":
        _render_data_dashboard(
            embedding_model=(embedding_model or "").strip() or os.getenv("MUJICA_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            use_fake_embeddings=use_fake_embeddings,
        )
    else:
        _render_research_agent(
            chat_api_key=chat_api_key,
            chat_base_url=chat_base_url,
            model_name=model_name,
            embedding_model=(embedding_model or "").strip() or os.getenv("MUJICA_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            use_fake_embeddings=use_fake_embeddings,
        )

    # 自动保存对话历史（仅当有实际内容时保存，避免空对话刷屏）
    try:
        if st.session_state.get("enable_chat_history") and st.session_state.get("conversation_id"):
            snap = _history_snapshot()
            # 只有当存在用户消息时才保存（避免刷新页面产生大量空对话）
            has_content = False
            msgs = snap.get("messages") or []
            for m in msgs:
                if isinstance(m, dict) and m.get("role") in {"user", "assistant"}:
                    has_content = True
                    break
            # 也检查是否有报告/研究笔记等内容
            if not has_content:
                if snap.get("final_report") or snap.get("research_notes") or snap.get("pending_plan"):
                    has_content = True
            
            if has_content:
                s = json.dumps(snap, ensure_ascii=False, sort_keys=True)
                h = str(hash(s))
                if h != str(st.session_state.get("history_last_hash") or ""):
                    save_conversation(str(st.session_state.get("conversation_id") or ""), snap)
                    st.session_state["history_last_hash"] = h
    except Exception:
        pass


if __name__ == "__main__":
    main()
