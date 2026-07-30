# PaperMind — 多 Agent 协作学术研究平台

基于 **RAG + CrewAI 多 Agent** 的学术论文智能助手。自动爬取 arXiv 论文，构建向量知识库，通过多 Agent 协同回答学术问题。

## 快速开始

```powershell
# 1. 安装依赖
uv venv --python 3.10 && uv pip install -r requirements.txt

# 2. 配置 .env（复制 .env.example 填入 API Key）
# 3. 启动后端 + 前端
python -m src.api.main          # 端口 8000
streamlit run src/ui/app.py     # 端口 8501

# 4. 浏览器打开 http://localhost:8501，点击"更新知识库"
```

## 核心架构

```
用户提问 → [查询改写] → 混合检索(BM25+向量+重排序) → Analyst分析 → Reviewer审查(可选) → Synthesizer综合回答
```

- **检索**：直接混合检索（不经过 LLM Agent），BM25 + 向量语义 + RRF 融合 + Cross-Encoder 重排序
- **问答**：多轮对话 + 查询改写 + 分层记忆（L1/L2/L3）+ Sub-Agent 工作集管理
- **知识库**：本地 Embedding（bge-small-zh-v1.5）+ ChromaDB + Gemini 图片理解（可选）
- **上下文**：自适应预算（根据 LLM 窗口动态分配），Reviewer 独立事实核查

## 技术栈

| 层 | 技术 |
|---|---|
| LLM | DeepSeek / Gemini / OpenAI 兼容 API |
| Agent | CrewAI 多 Agent 编排 |
| Embedding | BAAI/bge-small-zh-v1.5（本地免费） |
| 向量库 | ChromaDB（持久化） |
| PDF 解析 | PyMuPDF（表格提取 + 双栏识别） |
| 后端/前端 | FastAPI + Streamlit |

## 性能评估

提供两层评估体系，从检索到端到端逐层测量：

### 层次 1：BM25 检索质量（`--mode qasper`）

在 QASPER 论文内用 BM25 搜索，对比专家标注的 evidence 段落计算命中率。**测量的是纯检索算法能力**。

| 指标 | 含义 | 解读 |
|---|---|---|
| **Recall@K** | 前 K 个结果中命中了多少相关段落 | 低 = 搜索词和论文段落匹配不上 |
| **MRR** | 第一个相关段落的排名倒数 | 低 = 正确答案排得太靠后 |
| **NDCG@10** | 考虑排序质量 + 相关度的综合指标 | 低 = 排名不合理或遗漏重要段落 |

```powershell
python scripts/evaluate.py --mode qasper --max-samples 100
```

### 层次 2：端到端 RAG 准确率（`--mode qasper-e2e`）

将 QASPER 论文灌入独立向量库，完整走 检索 → LLM 回答，用另一个 LLM 对比参考答案打分。**测量的是检索 + 生成的核心能力**。

| 维度（1-10） | 含义 | 低分意味着 |
|---|---|---|
| **准确度** | 回答的关键事实是否与参考答案一致？ | 检索没找到正确段落，或 LLM 幻觉 |
| **覆盖度** | 回答是否涵盖了参考答案的主要信息点？ | 检索不全面，或 LLM 遗漏细节 |
| **忠实度** | 回答有无编造论文中不存在的内容？ | LLM 过度依赖自身知识而非检索结果 |

```powershell
python scripts/download_qasper.py                   # 下载 QASPER 数据集（仅首次）
python scripts/evaluate.py --mode qasper-e2e --max-samples 20
```

> 两层之间的关系：层次 1 确认"搜得对不对"，层次 2 确认"搜对了之后答得好不好"。如果层次 2 分数低但层次 1 分数高，说明问题在 LLM 回答环节；如果两层都低，问题在检索。

## 项目结构

```
src/
├── agents/        # CrewAI 多 Agent 系统
├── rag/           # RAG 管线（解析/分块/向量化/检索）
├── crawler/       # arXiv 爬虫 + 定时调度
├── api/           # FastAPI 后端
├── ui/            # Streamlit 前端
├── config.py      # 全局配置
└── metrics.py     # 性能指标收集
scripts/
├── evaluate.py         # 评估脚本
└── download_qasper.py  # 数据集下载
```

## 配置

| 环境变量 | 说明 |
|---|---|
| `LLM_API_KEY` | LLM API Key |
| `LLM_BASE_URL` | API 地址（默认 DeepSeek） |
| `LLM_MODEL` | 模型名（默认 deepseek-chat） |
| `GEMINI_API_KEY` | Gemini Key（可选，用于论文图片理解） |

详细说明见 `.env.example`。
