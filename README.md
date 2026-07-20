# PaperMind — 多 Agent 协作学术研究平台

基于 **RAG（检索增强生成）+ Multi-Agent 协作** 的学术论文智能研究助手。自动爬取 arXiv 论文，构建向量知识库，通过 5 个分工明确的 AI Agent 协同回答复杂的学术问题。

---

## 核心架构

```
┌──────────────────────────────────────────────────────┐
│                    Streamlit UI                      │
│               (端口 8501 · 对话界面)                   │
└──────────────────────┬───────────────────────────────┘
                       │ HTTP
┌──────────────────────▼───────────────────────────────┐
│                  FastAPI 后端                         │
│               (端口 8000 · RESTful API)                │
│                                                      │
│   ┌──────────────────────────────────────────────┐   │
│   │           多 Agent 协作系统                     │   │
│   │                                              │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│   │  │Orchestrator│  │ Reviewer  │  │ Crawler   │   │   │
│   │  │(任务编排) │  │(质量审查) │  │(论文爬取) │   │   │
│   │  └─────┬────┘  └──────────┘  └──────────┘   │   │
│   │        │                                     │   │
│   │  ┌─────▼────┐  ┌──────────┐                  │   │
│   │  │Retriever │  │ Analyst   │                  │   │
│   │  │(文献检索) │  │(深度分析) │                  │   │
│   │  └─────┬────┘  └──────────┘                  │   │
│   └────────┼─────────────────────────────────────┘   │
│            │                                         │
│   ┌────────▼─────────────────────────────────────┐   │
│   │              RAG 管线                          │   │
│   │                                              │   │
│   │  PDF 解析 → 分块(1000字符) → 向量化 → ChromaDB│   │
│   │   (含表格提取、图片说明、双栏识别)              │   │
│   └──────────────────────────────────────────────┘   │
│                                                      │
│   ┌──────────────────────────────────────────────┐   │
│   │          定时爬虫调度 (APScheduler)             │   │
│   │         arXiv API → PDF 下载 → 自动入链        │   │
│   └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## Multi-Agent 协作机制

### Agent 角色定义

| Agent | System Prompt 定位 | 核心职责 |
|---|---|---|
| **Orchestrator** | 任务编排者 | 解析用户意图 → 拆解子任务 → 协调分发 → 汇总综合回答 |
| **Retriever** | 知识检索专家 | LLM 提取关键词 → 多查询融合检索 → 格式化文献结果 |
| **Analyst** | 学术研究分析师 | 阅读检索文献 → 方法对比 → 趋势洞察 → 输出结构化报告 |
| **Reviewer** | 研究质量审查员 | 4 维度评分（事实/逻辑/全面性/实用性）→ 给出改进建议 |
| **Crawler** | 知识库管理员 | LLM 解析领域意图 → 触发 arXiv 爬取 → 新论文自动入链 |

### 协作流程

```
用户提问
  │
  ▼
Orchestrator 解析意图，拆解子任务
  │
  ├─► Retriever: LLM 提取 2-3 个关键词 → 多查询融合检索 → 去重排序
  │
  ▼
Analyst: 基于检索结果深度分析 → 结构化报告（核心发现/方法对比/趋势/引用）
  │
  ▼
Reviewer: 4 维度质量审查 → 评分 + 修改建议
  │
  ▼
Orchestrator: 综合分析报告 + 审查意见 → 输出最终优化答案
```

### Agent 通信机制

所有 Agent 继承自 `BaseAgent`，统一使用 OpenAI 兼容接口调用 LLM：

- **消息历史**：每个 Agent 独立维护最近 10 轮对话上下文
- **Agent 间通信**：`send_message()` / `receive_message()` 通过 `AgentMessage` 数据类传递
- **工具注册**：`BaseAgent.tools` 字典预留了 Function Calling 扩展点
- **温度参数**：统一使用 `0.3`，保证输出的稳定性和一致性

---

## RAG 技术细节

### 文档处理管线

```
arXiv PDF 下载
      │
      ▼
PyMuPDF 解析 ──┬── 纯文本提取（page.get_text("dict")）
               ├── 表格检测与提取（page.find_tables() → Markdown 格式化）
               └── 图片识别与标题匹配（type=1 图片块 + 关键词匹配）
      │
      ▼
多栏布局检测（x 中心点分布分析，间隙 > 40pt 自动判定双栏）
      │
      ▼
版面感知排序（单栏按 y / 双栏通栏→左栏→右栏）
      │
      ▼
RecursiveCharacterTextSplitter 分块
      │
      ▼
sentence-transformers 向量化（BAAI/bge-small-zh-v1.5）
      │
      ▼
ChromaDB 持久化存储（cosine 相似度）
```

### 分块策略

| 参数 | 值 | 说明 |
|---|---|---|
| `chunk_size` | 1000 字符 | 每个文本块最大长度 |
| `chunk_overlap` | 200 字符 | 相邻块重叠部分，防止关键信息被切断 |
| 分隔符优先级 | `\n\n` → `\n` → `。` → `. ` → `；` → `; ` → ` ` → 字符级 | 优先按自然段落/句子边界切分 |

### 检索策略

- **多查询融合**：LLM 从用户问题中提取 2-3 个关键词，每个关键词独立检索 `top_k=3`，合并去重
- **距离→相似度转换**：`similarity = 1 - cosine_distance`
- **上下文限制**：`format_context` 默认截断到 4000 字符，优先保留高相似度结果
- **领域过滤**：支持按研究领域（如 `cs.AI`）限定检索范围

### 文档解析增强

重写了 PDF 解析模块，解决学术论文排版复杂的难题：

| 能力 | 方案 | 效果 |
|---|---|---|
| **表格提取** | PyMuPDF `find_tables()` → Markdown 表 | LLM 可直接理解表格结构 |
| **图片标题** | 检测 type=1 图片块 → 匹配下方文本（Figure/Fig/图 开头） | `[图片说明: Figure 2. Architecture overview...]` |
| **双栏识别** | 统计 x 中心点分布 → 寻找最大间隙 → 自动判定栏分割线 | 双栏论文按正确阅读顺序输出（左栏 → 右栏） |
| **通栏处理** | 宽度 > 页宽 65% 的块（标题/摘要/跨栏图表）单独提取 | 不被打乱到栏中 |

---

## 技术栈

| 层级 | 技术 | 用途 |
|---|---|---|
| **LLM** | DeepSeek Chat (OpenAI 兼容) | 5 个 Agent 的推理引擎 |
| **Embedding** | `BAAI/bge-small-zh-v1.5` (本地) | 文本向量化，384 维，免费 |
| **向量数据库** | ChromaDB (Persistent) | 向量存储与余弦相似度检索 |
| **PDF 解析** | PyMuPDF (fitz) | 文本/表格/图片提取 + 布局分析 |
| **文本分块** | LangChain Text Splitters | 递归字符分块 |
| **后端** | FastAPI + Uvicorn | RESTful API 服务 |
| **前端** | Streamlit | 对话式交互界面 |
| **爬虫** | `arxiv` Python 库 | arXiv API 论文搜索与 PDF 下载 |
| **调度** | APScheduler | 定时自动爬取（默认 24h） |
| **包管理** | uv | 虚拟环境与依赖管理 |
| **Python** | ≥ 3.10 | 运行环境 |

---

## 项目结构

```
PaperMind/
├── src/
│   ├── agents/                 # Multi-Agent 系统
│   │   ├── base_agent.py       # Agent 基类（LLM 调用、消息传递）
│   │   ├── orchestrator.py     # 编排 Agent（任务分解 + 综合）
│   │   ├── retriever_agent.py  # 检索 Agent（关键词提取 + 多查询融合）
│   │   ├── analyst_agent.py    # 分析 Agent（文献综合 + 趋势洞察）
│   │   ├── reviewer_agent.py   # 审查 Agent（4 维度质量评分）
│   │   ├── crawler_agent.py    # 爬取 Agent（领域解析 + 触发爬取）
│   │   └── system.py           # 系统入口（组装所有模块 + 全局单例）
│   │
│   ├── rag/                    # RAG 管线
│   │   ├── document_processor.py  # PDF 解析（文本/表格/图片/双栏检测）
│   │   ├── chunker.py             # 递归文本分块
│   │   ├── embedder.py            # 向量化（本地/远程 API 双模式）
│   │   ├── vector_store.py        # ChromaDB 存储与检索
│   │   ├── retriever.py           # 检索器（多查询融合 + 上下文格式化）
│   │   └── pipeline.py            # 管线编排（解析→分块→向量化→存储）
│   │
│   ├── crawler/                # 论文爬取
│   │   ├── __init__.py         # ArxivCrawler（搜索 + PDF 下载）
│   │   └── scheduler.py        # 定时调度（APScheduler）
│   │
│   ├── api/
│   │   └── main.py             # FastAPI 服务（RESTful 接口）
│   │
│   ├── ui/
│   │   └── app.py              # Streamlit 前端（对话界面）
│   │
│   └── config.py               # 全局配置（环境变量驱动）
│
├── data/                       # 运行时数据（自动生成）
│   ├── papers/                 # 下载的 PDF 论文（按领域分目录）
│   └── chroma_db/              # ChromaDB 持久化向量库
│
├── .env                        # 环境变量配置（API Key 等）
├── .env.example                # 配置模板
├── pyproject.toml              # 项目元数据
├── requirements.txt            # 依赖清单
├── start.bat                   # 一键启动脚本
└── README.md
```

---

## 快速开始

### 1. 环境要求

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) 包管理器

### 2. 安装 uv 并部署虚拟环境

```powershell
# 安装 uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 重新打开终端后，创建虚拟环境
cd PaperMind
uv venv --python 3.10

# 激活环境
.venv\Scripts\Activate.ps1

# 安装依赖
uv pip install -r requirements.txt
```

### 3. 配置 API Key

编辑 `.env` 文件，填入你的 DeepSeek API Key：

```env
LLM_API_KEY=sk-your-deepseek-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

也支持任何 OpenAI 兼容的 API（如 OpenAI 官方、硅基流动等），修改 `LLM_BASE_URL` 和 `LLM_MODEL` 即可。

### 4. 启动

打开两个终端，分别启动后端和前端：

```powershell
# 终端 1：后端 API（端口 8000）
.venv\Scripts\Activate.ps1
python -m src.api.main

# 终端 2：前端界面（端口 8501）
.venv\Scripts\Activate.ps1
streamlit run src/ui/app.py --server.port 8501
```

浏览器打开 **http://localhost:8501** 即可使用。

### 5. 首次使用

1. 左侧边栏点击 **"更新知识库"** — 自动爬取 7 个预设领域的 arXiv 最新论文并向量化入链
2. 或选择单独领域爬取
3. 在聊天框输入学术问题，Agent 团队开始协作回答

---

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/query` | 用户提问（JSON: `{"question": "...", "enable_review": true}`） |
| `POST` | `/crawl` | 触发论文爬取（JSON: `{"domain": "人工智能"}` 或空=全部） |
| `POST` | `/ingest` | 将已下载的 PDF 向量化入链 |
| `GET` | `/stats` | 获取知识库统计（向量数/论文数/爬虫状态） |
| `POST` | `/scheduler/start` | 启动定时爬虫 |
| `POST` | `/scheduler/stop` | 停止定时爬虫 |
| `GET` | `/docs` | Swagger API 文档 |

---

## 配置项说明

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `LLM_API_KEY` | — | DeepSeek / OpenAI 兼容 API Key |
| `LLM_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `LLM_MODEL` | `deepseek-chat` | 模型名称 |
| `EMBEDDING_PROVIDER` | `local` | `local` 本地免费 / `api` 远程接口 |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地模型（384 维，~100MB） |
| `ARXIV_CRAWL_INTERVAL_HOURS` | `24` | 定时爬取间隔 |
| `ARXIV_MAX_RESULTS_PER_FETCH` | `100` | 每个领域每次最多爬取数 |
| `CHUNK_SIZE` | `1000` | 文本分块大小（字符） |
| `CHUNK_OVERLAP` | `200` | 块间重叠字符数 |
| `API_PORT` | `8000` | 后端端口 |
| `UI_PORT` | `8501` | 前端端口 |

**预设研究领域**（可修改 `config.py` 中的 `RESEARCH_DOMAINS`）：

| 领域 | arXiv 分类 |
|---|---|
| 大语言模型 | `cs.CL` |
| 人工智能 | `cs.AI` |
| 机器学习 | `cs.LG` |
| 计算机视觉 | `cs.CV` |
| 多智能体系统 | `cs.MA` |
| 信息检索 | `cs.IR` |
| 检索增强生成(RAG) | `cs.IR` |

---

## 应用场景

### 1. 学术文献综述

输入"近三年 LLM 推理优化方法有哪些？"，系统自动检索相关论文，给出方法对比、趋势分析和引用来源。

### 2. 方法对比研究

输入"多 Agent 协作策略有哪些主流方案？各自的优缺点？"，Analyst Agent 会横向对比不同论文中的方法，列出优劣。

### 3. 技术趋势追踪

配合定时爬虫，每日自动更新知识库。随时提问即可获取最新 arXiv 论文的发现。

### 4. 跨领域交叉研究

知识库覆盖 7 个 AI 子领域，支持按领域过滤检索，发现跨领域的方法迁移与灵感。

### 5. 论文阅读辅助

表格自动提取为 Markdown，图片标题自动匹配，双栏论文正确还原阅读顺序，让 RAG 更精准地命中论文核心内容。

---

## 设计决策

- **本地 Embedding 优先**：默认使用 `bge-small-zh-v1.5`（免费、中文优化、384 维、无 API 费用），也支持切换远程 API
- **多查询融合检索**：相比单次检索，LLM 提取多关键词后融合检索能显著提高召回率
- **多 Agent 协作而非单 Agent**：将检索、分析、审查分离，每个 Agent 可以独立优化 prompt 和策略，审查 Agent 提供额外的质量保障
- **版面感知解析**：针对 CS/ML 论文普遍的双栏 PDF 做了专门处理，自动检测栏位并还原正确阅读顺序
