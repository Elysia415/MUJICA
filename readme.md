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

# MUJICA - 深度洞察与报告生成智能体

**(Multi-stage User-Judged Integration & Corroboration Architecture)**

## 1. 项目核心愿景

构建一个专注于 **各大AI会议论文** 的垂直领域专家 Agent。该系统不依赖通用搜索引擎，而是通过 **OpenReview API** 获取该会议的全量投稿、评审意见及决策逻辑，构建本地化的“全知”知识库。用户可以通过自然语言指令，生成关于 AI 前沿趋势的深度调研报告。

---

## 2. 系统架构 (Architecture)

本项目采用“**云端大脑 + 本地全量知识库**”的混合架构。

### 数据层：全量领域知识库 (The Domain Brain)
- **数据源**：OpenReview API V2
- **数据内容**：
    - **Paper Metadata**：标题、作者、摘要、关键词、TL;DR。
    - **Full Text**：下载 PDF 并解析后的纯文本。
    - **Reviews (亮点)**：审稿人的评分、修改意见、领域主席 (Meta Reviewer) 的决策理由。
- **存储技术**：
    - **向量库**：LanceDB (存储语义索引)。
    - **元数据库**：Pandas DataFrame / SQLite (存储结构化数据)。

### 逻辑层：MUJICA 四阶段工作流

#### 规划与校验
流程始于“规划器”生成研究大纲，并暂停，等待用户的人工批准或编辑，确保任务目标精确。

#### 研究与构建
批准后，“规划器”执行动态研究循环，深入探索并收集证据，将所有原始资料构建为一个结构化的“记忆库”。

#### 循证写作
激活“写入器”，该代理从“记忆库”中进行定向检索，严格基于已核实的证据撰写报告草稿。

#### 核查与交付
最后，事实核查代理启动，逐句对比草稿与“记忆库”中的原始证据，进行交叉验证，并交付附有核查状态的最终报告。

---

## 3. 技术栈 (Tech Stack)

- **数据获取**：`openreview-py`, `requests`
- **PDF 解析**：`PyPDF2` / `pdfplumber`
- **数据存储**：`lancedb` (向量), `pandas` (元数据)。
- **大模型接口**：`openai` SDK (兼容 DeepSeek / Google Gemini / GPT-4)。
- **Web 界面**：`streamlit`
- **环境管理**：`python-dotenv`

## 4. 快速开始 (Quick Start)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置环境
复制 `.env.example` 到 `.env` 并填入 API Key：
```bash
cp .env.example .env
```

### 运行应用
```bash
streamlit run ui/app.py
```
