---
title: MUJICA
emoji: 🌌
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# MUJICA · 论文调研报告生成器（循证可溯源）

MUJICA（Multi-stage User‑Judged Integration & Corroboration Architecture）是一个面向 **OpenReview 会议论文** 的本地知识库 + 多阶段智能体系统：从抓取入库、语义检索、循证写作到逐句核查，全流程可追溯。

你可以把它理解成：**“本地论文/评审知识库 + 研究计划生成 + 证据检索 + 可引用写作 + 核查”** 的一站式工具。

---

## 功能亮点

- **OpenReview 一键入库**
  - 支持按会议/年份（Venue ID）抓取
  - 支持 **仅 Accept** + 展示类型（oral/spotlight/poster/unknown）过滤
  - 抓取并保存：论文元信息、评分（聚合）、**Reviews / Decision note / Rebuttal**（若可获取）
- **PDF 下载/解析可选且可控**
  - 并发下载、可配置延迟/超时/重试
  - 支持校验已有 PDF、强制重下、最小文件大小、EOF 校验
  - 解析默认优先 PyMuPDF，fallback pdfplumber / PyPDF2
- **双存储：SQLite + LanceDB**
  - SQLite：结构化元数据（year/rating/decision/presentation/reviews/decision_text/rebuttal_text/pdf_path…）
  - LanceDB：向量索引（paper 向量 + chunks 向量；包含 meta/review/decision/rebuttal/full_text）
- **四阶段工作流（Planner → Researcher → Writer → Verifier）**
  - **可编辑计划**：可读表单视图 + JSON 高级视图
  - **证据与引用**：报告正文用 `[R#]` 引用，右侧 Evidence 面板可溯源
  - **核查**：Verifier 兼容多种引用格式并逐句核查（可选 JSON mode）
- **知识库管理更顺手**
  - 已入库表格支持勾选：单篇详情 + 批量删除一体化
  - 支持修复 `pdf_path`（扫描 `data/raw/pdfs/<paper_id>.pdf` 回填）
- **更清晰的进度与日志**
  - 入库过程：抓取/下载/解析/向量化/写入向量表/切分 chunks 全部有 UI 进度条
  - 写作过程：refs 数量、输入规模、引用覆盖率等统计更可观测
- **主题样式**
  - 主题切换：**简明（亮色）** / **MUJICA（暗色）**

---

## 快速开始（本地）

### 1) 安装依赖

建议 Python 版本：**3.10+**（越新越好）。

```bash
pip install -r requirements.txt
```

### 2) 配置环境变量（推荐用 `.env`）

在项目根目录创建一个 `.env` 文件（本项目已在 `.gitignore` 忽略该文件，不会被提交），示例：

```env
# Demo 授权（可选）
MUJICA_ACCESS_CODE=mujica2025

# Chat（用于 Planner/Researcher/Writer/Verifier）
OPENAI_API_KEY=your_chat_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
MUJICA_DEFAULT_MODEL=gpt-4o

# Embedding（与 Chat 完全解耦）
MUJICA_EMBEDDING_MODEL=BAAI/bge-m3
MUJICA_EMBEDDING_API_KEY=your_embedding_api_key_here
MUJICA_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
MUJICA_EMBEDDING_BATCH_SIZE=64

# OpenReview（可选，部分会议抓 reviews/decision/rebuttal 需要登录）
OPENREVIEW_USERNAME=
OPENREVIEW_PASSWORD=
```

### 3) 启动 UI

推荐（本地）：  

```bash
streamlit run ui/app.py
```

也可（用于 HuggingFace Spaces 入口，等价）：  

```bash
streamlit run app.py
```

---

## 使用流程（UI）

1. **📚 知识库**：抓取 OpenReview 或导入样例（可选下载/解析 PDF）
2. **🏠 首页**：输入「研究问题 / 报告主题」，必要时补充「辅助关键词」
3. **审核计划**：在可读版里调整过滤条件/章节，或用 JSON 高级编辑
4. **确认并运行**：Research → Write → Verify；右侧 Evidence/Verification 面板可溯源
5. **下载报告**：最终报告支持 Markdown 渲染，并可一键下载 `.md`

---

## 模型与鉴权配置（重要）

MUJICA 将 **聊天模型** 与 **Embedding 模型** 完全解耦（可以不同 provider / 不同 key / 不同 base_url）。

- **聊天模型（Chat）**
  - 用于：Planner/Researcher/Writer/Verifier
  - 环境变量：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`MUJICA_DEFAULT_MODEL`
  - UI 支持 Access Code 模式：输入正确 Access Code 后使用系统环境中的 `OPENAI_API_KEY`

- **Embedding（向量化）**
  - 用于：入库/检索
  - 环境变量：`MUJICA_EMBEDDING_MODEL`、`MUJICA_EMBEDDING_API_KEY`、`MUJICA_EMBEDDING_BASE_URL`
  - 常见坑：**把聊天模型名当 embedding 模型**会报 `Model does not exist (code 20012)`
  - SiliconFlow 示例（以其文档为准）：`BAAI/bge-m3`、`BAAI/bge-large-zh-v1.5` 等

---

## OpenReview 抓取说明

- 默认会抓 submission 元信息；**reviews/decision/rebuttal** 是否可获得取决于会议公开策略与权限。
- 若发现 reviews 为空，建议在 UI 高级选项中填写：
  - `OPENREVIEW_USERNAME` / `OPENREVIEW_PASSWORD`
- 若你修复过分类逻辑或需要覆盖历史误分类 reviews，可勾：
  - **强制刷新 Reviews（MUJICA_REPLACE_EMPTY_REVIEWS）**

---

## 关键环境变量一览（常用）

### 基础
- **`OPENAI_API_KEY`**：聊天模型 API Key（Access Code 模式会用系统这个 key）
- **`OPENAI_BASE_URL`**：聊天模型 Base URL（OpenAI-compatible）
- **`MUJICA_ACCESS_CODE`**：Access Code（用于 demo 授权）
- **`MUJICA_DEFAULT_MODEL`**：默认聊天模型名（UI 可改）

### Embedding（与 Chat 解耦）
- **`MUJICA_EMBEDDING_MODEL`** / **`MUJICA_EMBEDDING_API_KEY`** / **`MUJICA_EMBEDDING_BASE_URL`**
- **`MUJICA_EMBEDDING_BATCH_SIZE`**：embedding 批大小（SiliconFlow 常见上限 64）
- **`MUJICA_EMBEDDING_MIN_INTERVAL`**：embedding 请求最小间隔（缓解 429 TPM）
- **`MUJICA_EMBEDDING_RETRY_MAX`** / `..._BASE_DELAY` / `..._MAX_DELAY`：429 自动退避重试
- **`MUJICA_FAKE_EMBEDDINGS=1`**：离线 embedding（测试/演示用）

### OpenReview / PDF
- **`MUJICA_OPENREVIEW_PAGE_SIZE`**：分页大小
- **`MUJICA_PDF_DOWNLOAD_WORKERS`** / `..._DELAY` / `..._TIMEOUT` / `..._RETRIES`
- **`MUJICA_PDF_FORCE_REDOWNLOAD`** / `MUJICA_PDF_VALIDATE_EXISTING` / `MUJICA_PDF_EOF_CHECK` / `MUJICA_PDF_MIN_BYTES`
- **`MUJICA_PDF_DEBUG=1`**：输出 pdfplumber/pdfminer 噪音日志（排障用）

### Reviews/证据与写作
- **`MUJICA_INGEST_REVIEWS`**：是否将 reviews 向量化（默认 1）
- **`MUJICA_MAX_REVIEWS_PER_PAPER`**：每篇最多入库多少条 review（默认 10）
- **`MUJICA_EVIDENCE_REVIEW_CHUNKS_PER_PAPER`** / `...DECISION...` / `...REBUTTAL...`
- **`MUJICA_WRITER_MAX_TOKENS`**：Writer 最大输出 token（默认 4096）
- **`MUJICA_DISABLE_JSON_MODE=1`**：兼容不支持 `response_format` 的模型（如部分 GLM）

---

## 数据目录说明

- `data/lancedb/`：LanceDB 向量库 + `metadata.sqlite`（SQLite 元数据库）
- `data/raw/pdfs/`：下载的 PDF（文件名为 `<paper_id>.pdf`）
- 如需“清库”，可删除上述目录（注意已在 `.gitignore` 中忽略）。

---

## 常见问题（FAQ）

- **Embedding 报 `Model does not exist (code 20012)`**  
  说明你正在用不支持的 embedding 模型名/网关。请在 UI 中配置正确的 `MUJICA_EMBEDDING_*`，不要用聊天模型名当 embedding。

- **Embedding 报 413 / `batch size > 64`**  
  将 `MUJICA_EMBEDDING_BATCH_SIZE=64`（或更小）。

- **429 TPM 限流**  
  适当增大 `MUJICA_EMBEDDING_MIN_INTERVAL`，或降低抓取规模/解析页数。

- **reviews/decision/rebuttal 抓不到**  
  尝试填写 `OPENREVIEW_USERNAME/PASSWORD`；另外取决于会议是否公开。

---

## 部署到 HuggingFace Spaces（Demo）

- **入口文件**：本仓库根目录的 `app.py` 是 Spaces 入口（也可本地 `streamlit run app.py`）。
- **密钥管理**：把 `OPENAI_API_KEY` / `MUJICA_EMBEDDING_API_KEY` 等放到 Spaces 的 Secrets，不要写进仓库。
- **多人使用注意**：默认情况下所有用户会共享同一份本地数据库与向量库（`data/` 目录）。如果公开 demo 允许“入库/删除/长任务”，可能出现并发写入冲突、数据互相影响与额度被刷爆。
  - 建议：公开 demo 只开放“检索/生成报告”，入库与删除仅管理员使用（或把 Space 设为私有）。
- **数据持久化**：如需重启后保留知识库，请在 Spaces 侧开启持久化存储；否则 `data/` 可能会在重启后丢失。

---

## License

MIT（见 `LICENSE`）
