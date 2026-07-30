"""
项目全局配置，通过 .env 文件和环境变量管理。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")

# 设置 Hugging Face 镜像（解决国内无法访问 huggingface.co 的问题）
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")

# ---- 项目路径 ----
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
CHROMA_DIR = DATA_DIR / "chroma_db"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

# 确保目录存在
for d in [DATA_DIR, PAPERS_DIR, CHROMA_DIR, CONVERSATIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- LLM 配置 (DeepSeek / OpenAI 兼容) ----
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.3

# ---- Embedding 配置 ----
# provider: "local" 使用本地 sentence-transformers 模型 | "api" 使用远程 API
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", LLM_BASE_URL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ---- 向量数据库配置 ----
CHROMA_COLLECTION_NAME = "arxiv_papers"

# ---- 爬虫配置 ----
ARXIV_CRAWL_INTERVAL_HOURS = int(os.getenv("ARXIV_CRAWL_INTERVAL_HOURS", "24"))
ARXIV_MAX_RESULTS_PER_FETCH = int(os.getenv("ARXIV_MAX_RESULTS_PER_FETCH", "20"))
ARXIV_REQUEST_DELAY = 3.0  # arXiv API 限速间隔(秒)

# ---- 领域配置 ----
# 预设的可爬取领域，key=领域名，value=arXiv分类
RESEARCH_DOMAINS = {
    "大语言模型": "cs.CL",
    "人工智能": "cs.AI",
    "机器学习": "cs.LG",
    "计算机视觉": "cs.CV",
    "多智能体系统": "cs.MA",
    "信息检索": "cs.IR",
    "检索增强生成(RAG)": "cs.IR",  # 用关键词额外过滤
}

# ---- 分块配置 ----
# 章节感知分块：基准字符数，实际按章节→段落边界切分
CHUNK_SIZE = 1000

# ---- Gemini 多模态配置 (图片理解，可选) ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ---- 上下文管理配置 ----
# LLM 上下文窗口上限（deepseek-chat = 128K）
LLM_MAX_CONTEXT_TOKENS = int(os.getenv("LLM_MAX_CONTEXT_TOKENS", "128000"))
# 窗口安全系数：预留 30% 给模型输出
CONTEXT_SAFE_RATIO = 0.7
# 自适应预算分配比例（总可用上下文 = LLM_MAX_CONTEXT_TOKENS * CONTEXT_SAFE_RATIO）
CONTEXT_BUDGET_L1_RATIO = 0.15    # L1 工作记忆（最近 N 轮完整保留）
CONTEXT_BUDGET_L2_RATIO = 0.10    # L2 中期记忆（压缩后的结构化要点）
CONTEXT_BUDGET_RETRIEVAL_RATIO = 0.45  # 检索结果 / 工作集摘要
CONTEXT_BUDGET_RESERVE_RATIO = 0.30    # 预留给 Agent 输出 + 任务描述

# 分层记忆容量
L1_WORKING_ROUNDS = 5          # L1 工作记忆保留的最近轮数
L2_MEDIUM_ROUNDS = 20          # L2 中期记忆最多保留的轮数（压缩后）
L3_PINNED_MAX = 20             # L3 用户置顶笔记最多条数
MEMORY_COMPRESSION_TOKENS = 150  # 压缩每条旧轮次时 LLM 输出上限

# 工作集 (Working Set) 配置
WORKING_SET_MAX_FRAGMENTS = 8    # FIFO 队列最大容量
WORKING_SET_RELEVANCE_THRESHOLD = 0.6  # 向量相似度命中阈值

# ---- API 配置 ----
API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("API_PORT", "8000"))
UI_PORT = int(os.getenv("UI_PORT", "8501"))
