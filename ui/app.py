from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Optional

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


def _apply_theme_vars(theme: str) -> None:
    """
    通过 CSS 变量实现主题切换。
    注意：变量必须在页面渲染早期注入，且每次 rerun 都注入一次，避免旧主题残留。
    """
    theme = (theme or "").strip().lower()
    if theme in {"dark", "深色"}:
        vars_css = """
            --bg: #0b1020;
            --panel: #0f172a;
            --panel-2: rgba(15, 23, 42, 0.72);
            --text: #e5e7eb;
            --muted: #94a3b8;
            --border: rgba(148, 163, 184, 0.20);
            --accent: #ff5c93;
            --accent-hover: #ff3b82;
            --shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
            --sidebar-bg: rgba(15, 23, 42, 0.90);
            --input-bg: rgba(15, 23, 42, 0.92);
            --code-bg: rgba(2, 6, 23, 0.72);
        """
    else:
        # 默认：浅色粉系（参考截图风格）
        vars_css = """
            --bg: #faf7fb;
            --panel: #ffffff;
            --panel-2: rgba(255, 255, 255, 0.92);
            --text: #111827;
            --muted: #6b7280;
            --border: rgba(17, 24, 39, 0.12);
            --accent: #ff5c93;
            --accent-hover: #ff3b82;
            --shadow: 0 10px 24px rgba(17, 24, 39, 0.08);
            --sidebar-bg: rgba(255, 255, 255, 0.92);
            --input-bg: rgba(255, 255, 255, 0.98);
            --code-bg: rgba(17, 24, 39, 0.04);
        """

    st.markdown(f"<style>:root{{{vars_css}}}</style>", unsafe_allow_html=True)



def _ingest_test_dataset(kb: KnowledgeBase, path: str = "data/raw/test_samples.json") -> int:
    """
    一键导入样例数据，方便本地快速跑通工作流。
    返回导入的 paper 数量。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    loader = DataLoader(path)
    
    # 1. 如果文件不存在，先创建假数据并保存（这部分保持在 if 里面）
    if not os.path.exists(path):
        sample_papers = [
            {"id": "p1", "title": "Self-Rewarding Language Models", "abstract": "We propose...", "rating": 9.0},
            {"id": "p2", "title": "Direct Preference Optimization", "abstract": "DPO is stable...", "rating": 9.5},
        ]
        loader.save_local_data(sample_papers)
    
    # 2. 【关键修改】下面这两行必须向左移动，和 if 对齐
    # 无论上面是否创建了新文件，这里都要读取数据
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
            st.dataframe(view[cols].head(500), **_width_kwargs(st.dataframe, stretch=True))

            st.divider()
            st.markdown("**批量删除（基于当前筛选结果）**")
            st.warning("批量删除不可撤销：会删除选中论文的元数据/评审/向量索引（以及可选本地 PDF）。")
            try:
                ids_for_batch = view["id"].tolist()[:500] if "id" in view.columns else []
                titles_for_batch = (
                    view["title"].fillna("").tolist()[:500] if "title" in view.columns else [""] * len(ids_for_batch)
                )
                opts = []
                for _pid, _title in zip(ids_for_batch, titles_for_batch):
                    _pid = str(_pid)
                    _title = str(_title or "")
                    opts.append(f"{_pid} · {_title[:80]}")

                selected_opts = st.multiselect(
                    "选择要删除的论文（最多 500 条）",
                    options=opts,
                    default=[],
                    help="先用上面的标题关键词过滤缩小范围，再在这里多选要删除的条目。",
                )
                selected_ids = [x.split(" · ", 1)[0] for x in selected_opts if isinstance(x, str)]
            except Exception:
                selected_ids = []

            batch_delete_pdf = st.checkbox(
                "同时删除本地 PDF 文件（如果存在）",
                value=False,
                key="batch_del_pdf",
            )
            batch_confirm = st.checkbox(
                f"我已确认要删除选中的 {len(selected_ids)} 篇论文",
                value=False,
                key="batch_del_confirm",
            )
            if st.button(
                f"批量删除（{len(selected_ids)}）",
                type="primary",
                disabled=(not batch_confirm) or (not selected_ids),
                key="batch_del_btn",
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

            # 单篇详情
            try:
                ids = view["id"].tolist()[:500] if "id" in view.columns else []
                if ids:
                    pid = st.selectbox("查看单篇详情（paper_id）", options=ids)
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
                    if reviews:
                        st.markdown("**Reviews（前 3 条）**")
                        st.json(reviews[:3], expanded=False)

                    st.divider()
                    st.markdown("**删除条目**")
                    st.warning("删除不可撤销：将删除该论文的元数据、评审与向量索引（以及可选本地 PDF）。")
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
            except Exception:
                pass

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
        limit = st.slider(
            "抓取数量上限",
            10,
            300,
            50,
            help="当开启“仅 Accept”时，这个上限指 accepted 论文数量；系统会扫描更多 submission 直到凑够或扫完。",
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
                "- **SQLite**：title/authors/keywords/year/decision/presentation/rating/reviews/pdf_url/pdf_path\n"
                "- **LanceDB**：paper 向量 + chunks（含 meta chunk；若勾选解析则含全文 chunks）"
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
            },
            expanded=False,
        )

        if st.button("开始抓取并入库", type="primary", **_width_kwargs(st.button, stretch=True)):
            if not venue_id:
                st.error("Venue ID 不能为空。请先选择会议/年份或手动填写。")
                st.stop()

            # 将高级参数写入环境变量（fetcher 内部按 env 读取）
            try:
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
                st.error("未配置 Embedding 所需的 API Key。请在侧边栏填写 Key，或开启“离线 Embedding”。")
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
                        "请更换 Embedding Model（注意：embedding 模型通常与聊天模型不同），或开启“离线 Embedding”。"
                    )
                    st.stop()

            kb = KnowledgeBase(
                embedding_model=embedding_model,
                embedding_api_key=embedding_api_key,
                embedding_base_url=embedding_base_url,
            )
            kb.initialize_db()
            ingestor = OpenReviewIngestor(kb, fetcher=ConferenceDataFetcher(output_dir="data/raw"))

            with st.status("正在抓取 OpenReview...", expanded=True) as status:
                st.write("抓取 / 下载 / 解析 / 建索引 ...")
                dl_bar = st.progress(0)
                dl_text = st.empty()
                parse_bar = st.progress(0)
                parse_text = st.empty()

                def _on_progress(payload):
                    if not isinstance(payload, dict):
                        return
                    stage = payload.get("stage")
                    cur = int(payload.get("current") or 0)
                    tot = int(payload.get("total") or 0)
                    if tot <= 0:
                        return

                    pct = int(cur * 100 / tot)
                    if stage == "download_pdf":
                        dl_bar.progress(min(100, max(0, pct)))
                        dl_text.caption(f"下载 PDF：{cur}/{tot}")
                        return

                    if stage == "parse_pdf":
                        parse_bar.progress(min(100, max(0, pct)))
                        title = payload.get("title") or ""
                        parse_text.caption(f"解析 PDF：{cur}/{tot} · {title[:60]}")
                        return

                papers = ingestor.ingest_venue(
                    venue_id=venue_id,
                    limit=limit,
                    accepted_only=accepted_only,
                    presentation_in=presentation_in,
                    download_pdfs=download_pdfs,
                    parse_pdfs=parse_pdfs,
                    max_pdf_pages=max_pages if parse_pdfs else None,
                    max_downloads=limit if download_pdfs else None,
                    on_progress=_on_progress,
                )
                dl_bar.progress(100)
                parse_bar.progress(100)
                status.update(label="入库完成！", state="complete")

            try:
                decided = sum(1 for p in (papers or []) if (p or {}).get("decision"))
                rated = sum(1 for p in (papers or []) if (p or {}).get("rating") is not None)
                st.success(f"成功入库 {len(papers)} 篇论文（decision={decided} · rating={rated}）")
            except Exception:
                st.success(f"成功入库 {len(papers)} 篇论文。")


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
        # 兼容 Streamlit 1.26：st.container 不支持 border 参数
        # 这里用 st.form 做“卡片容器”，再用 CSS 把 form 渲染成卡片。
        with st.form("landing_card", clear_on_submit=False):
            topic = st.text_input("文章主题", placeholder="例如：用综述研究 量子计算", key="landing_topic")
            keywords = st.text_input("检索关键词（可选）", placeholder="例如：DPO, alignment, preference", key="landing_keywords")

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
                st.warning("请先填写「文章主题」。")
            else:
                q = topic.strip()
                if (keywords or "").strip():
                    q = f"{q}\n关键词：{keywords.strip()}"
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

            # 初始化 LLM（用于 Planner）
            llm = get_llm_client(api_key=chat_api_key, base_url=chat_base_url)
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
                with st.status("正在规划...", expanded=True) as status:
                    st.write("生成研究计划（Plan）...")
                    plan = planner.generate_plan(user_query, stats)
                    st.session_state["pending_plan"] = plan
                    st.session_state["plan_editor_text"] = json.dumps(plan, ensure_ascii=False, indent=2)

                    if isinstance(plan, dict) and (plan.get("_error") or plan.get("title") == "Error in Planning"):
                        status.update(label="规划失败（请检查模型/接口能力）", state="error")
                        err = plan.get("_error") or "unknown"
                        st.error(
                            "Planner 生成计划失败。常见原因：模型不支持 JSON mode（例如部分 GLM 会报 code=20024），"
                            "或 Model Name/Base URL 不匹配。\n\n"
                            f"错误信息：{err}"
                        )
                    else:
                        status.update(label="计划已生成（等待你确认/编辑）", state="complete")

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
            if run_req:
                llm = get_llm_client(api_key=chat_api_key, base_url=chat_base_url)
                if not llm:
                    st.error("Authentication Failed. Please provide a valid Access Code or your own API Key.")
                else:
                    plan = st.session_state.get("pending_plan")
                    if not isinstance(plan, dict):
                        st.error("计划格式异常：pending_plan 不是 JSON object。")
                    else:
                        st.session_state["plan_approved"] = True

                        researcher = ResearcherAgent(kb, llm, model=model_name)
                        writer = WriterAgent(llm, model=model_name)
                        verifier = VerifierAgent(llm, model=model_name)

                        with st.status("正在执行...", expanded=True) as status:
                            st.write("检索证据（Research）...")
                            research_bar = st.progress(0)
                            research_text = st.empty()

                            def _on_research_progress(payload):
                                if not isinstance(payload, dict):
                                    return
                                stage = payload.get("stage")
                                cur = int(payload.get("current") or 0)
                                tot = int(payload.get("total") or 0)
                                sec = payload.get("section") or ""
                                if tot > 0:
                                    pct = int(cur * 100 / tot)
                                    research_bar.progress(min(100, max(0, pct)))
                                if stage == "research_section":
                                    q = payload.get("query") or ""
                                    research_text.caption(f"检索中：{cur}/{tot} · {sec} · {str(q)[:60]}")
                                elif stage == "research_section_done":
                                    ev = payload.get("evidence")
                                    sp = payload.get("selected_papers")
                                    dt = payload.get("elapsed")
                                    research_text.caption(f"完成：{cur}/{tot} · {sec} · papers={sp} · evidence={ev} · {dt:.1f}s")

                            notes = researcher.execute_research(plan, on_progress=_on_research_progress)
                            research_bar.progress(100)
                            st.session_state["research_notes"] = notes

                            st.write("循证写作（Write）...")
                            report = writer.write_report(plan, notes)
                            st.session_state["final_report"] = report

                            st.write("逐句核查（Verify）...")
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
            st.subheader("最终报告")
            st.markdown(st.session_state["final_report"])

            v = st.session_state.get("verification_result")
            if isinstance(v, dict) and v:
                st.caption(f"Verification: valid={v.get('is_valid')} · score={v.get('score')} · {v.get('notes')}")

    with col_context:
        st.subheader("证据与核查")

        tab_evi, tab_ver = st.tabs(["Evidence（证据）", "Verification（核查）"])

        with tab_evi:
            notes = st.session_state.get("research_notes") or []
            if not notes:
                st.info("暂无证据。请先导入数据，再在底部输入问题。", icon="ℹ️")
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
                st.info("暂无核查结果。生成报告后会自动触发核查。", icon="ℹ️")
            else:
                st.caption(f"valid={v.get('is_valid')} · score={v.get('score')} · {v.get('notes')}")
                evals = v.get("evaluations") or []
                if evals:
                    try:
                        import pandas as pd

                        st.dataframe(pd.DataFrame(evals), **_width_kwargs(st.dataframe, stretch=True))
                    except Exception:
                        st.json(evals, expanded=False)
                else:
                    st.json(v, expanded=False)

    # Chat 输入框必须位于页面根容器（不能在 columns/tabs/sidebar/expander/form 内）
    prompt = st.chat_input("输入你的研究问题（按 Enter 发送）")
    if prompt:
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
    st.session_state.setdefault("pending_plan", None)
    st.session_state.setdefault("plan_editor_text", "")
    st.session_state.setdefault("plan_approved", False)
    st.session_state.setdefault("verification_result", None)

    _local_css(Path(__file__).with_name("style.css"))
    _apply_theme_vars(st.session_state.get("ui_theme"))

    with st.sidebar:
        st.title("MUJICA")
        st.caption("Multi-stage User-Judged Integration")

        st.divider()
        st.subheader("界面")
        st.radio(
            "主题",
            options=["light", "dark"],
            key="ui_theme",
            horizontal=True,
            format_func=lambda x: "浅色" if x == "light" else "深色",
        )
        st.radio(
            "导航",
            options=["research", "data"],
            key="system_mode",
            format_func=lambda x: "🏠 首页" if x == "research" else "📚 知识库",
        )

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
    env_embed_key = (os.getenv("MUJICA_EMBEDDING_API_KEY") or "").strip() or None
    env_embed_base = (os.getenv("MUJICA_EMBEDDING_BASE_URL") or "").strip() or None
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


if __name__ == "__main__":
    main()
