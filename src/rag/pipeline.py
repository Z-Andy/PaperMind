"""
RAG 管线编排：串联文档解析、分块、向量化、存储的完整流程。
"""
import logging
from pathlib import Path
from typing import Optional

from src.config import PAPERS_DIR
from src.rag.document_processor import DocumentProcessor, Document
from src.rag.chunker import SectionChunker, Chunk
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG 管线编排器"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        papers_dir: Path = PAPERS_DIR,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.papers_dir = Path(papers_dir)
        self.processor = DocumentProcessor()
        self.chunker = SectionChunker()
        self.retriever = Retriever(vector_store, embedder)

    def ingest_papers(
        self,
        papers_dir: Optional[Path] = None,
        clear_existing: bool = False,
    ) -> dict:
        """
        完整数据入链流程：PDF → 解析 → 分块 → 向量化 → 存储。

        Args:
            papers_dir: 论文目录（默认使用配置的目录）
            clear_existing: 是否清空现有向量库

        Returns:
            {"parsed": int, "chunks": int, "stored": int}
        """
        if papers_dir is None:
            papers_dir = self.papers_dir

        if clear_existing:
            self.vector_store.clear()

        # Step 1: 解析 PDF
        logger.info("=== Step 1: 解析 PDF 文档 ===")
        documents = self.processor.parse_directory(papers_dir)
        if not documents:
            logger.warning("未找到可解析的 PDF 文件")
            return {"parsed": 0, "chunks": 0, "stored": 0}

        # Step 2: 文本分块
        logger.info(f"=== Step 2: 文本分块 ({len(documents)}篇论文) ===")
        chunks = self.chunker.chunk_documents(documents)

        # Step 3: 向量化 + 存储
        logger.info(f"=== Step 3: 向量化并存储 ({len(chunks)}个文本块) ===")
        stored = self.vector_store.add_chunks(chunks, self.embedder)

        result = {
            "parsed": len(documents),
            "chunks": len(chunks),
            "stored": stored,
        }
        logger.info(f"入链完成: {result}")
        return result

    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_domain: Optional[str] = None,
    ) -> str:
        """
        端到端查询：检索 + 格式化上下文。

        Returns:
            格式化的上下文字符串
        """
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            filter_domain=filter_domain,
        )
        return self.retriever.format_context(results)

    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        return {
            "total_chunks": self.vector_store.get_count(),
            "domains": self.vector_store.get_domain_stats(),
        }
