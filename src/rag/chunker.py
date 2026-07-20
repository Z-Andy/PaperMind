"""
文本分块器：将长文档拆分成适合向量化和检索的小块。
"""
import logging
from dataclasses import dataclass
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """文档分块"""
    text: str
    chunk_index: int
    metadata: dict


class TextChunker:
    """智能文本分块器"""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: Optional[list[str]] = None,
    ):
        """
        Args:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 相邻块的重叠字符数
            separators: 分隔符优先级列表（默认适合中文和学术论文）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            separators = [
                "\n\n",    # 段落
                "\n",      # 换行
                "。",      # 中文句号
                ". ",      # 英文句号
                "；",      # 中文分号
                "; ",      # 英文分号
                " ",       # 空格
                "",        # 字符级
            ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_text(self, text: str, metadata: dict = None) -> list[Chunk]:
        """将文本拆分为块"""
        if not text or not text.strip():
            return []

        chunks_text = self.splitter.split_text(text)
        chunks = []

        for i, chunk_text in enumerate(chunks_text):
            chunk = Chunk(
                text=chunk_text.strip(),
                chunk_index=i,
                metadata={
                    **(metadata or {}),
                    "chunk_index": i,
                    "char_count": len(chunk_text),
                },
            )
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: list) -> list[Chunk]:
        """
        批量分块多个文档。

        Args:
            documents: Document 对象列表

        Returns:
            所有分块列表
        """
        all_chunks = []
        for doc in documents:
            meta = {
                "source": doc.file_path,
                "title": doc.title,
                **doc.metadata,
            }
            chunks = self.chunk_text(doc.content, metadata=meta)
            all_chunks.extend(chunks)

        logger.info(
            f"分块完成: {len(documents)}篇论文 → {len(all_chunks)}个文本块"
        )
        return all_chunks
