"""
项目全局配置，通过 .env 文件和环境变量管理。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")

# ---- 项目路径 ----
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
CHROMA_DIR = DATA_DIR / "chroma_db"

# 确保目录存在
for d in [DATA_DIR, PAPERS_DIR, CHROMA_DIR]:
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
ARXIV_MAX_RESULTS_PER_FETCH = int(os.getenv("ARXIV_MAX_RESULTS_PER_FETCH", "100"))
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
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---- API 配置 ----
API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("API_PORT", "8000"))
UI_PORT = int(os.getenv("UI_PORT", "8501"))
