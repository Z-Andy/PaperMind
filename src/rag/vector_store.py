"""
向量存储：基于 ChromaDB 的向量持久化和检索。
"""
import logging
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.config import CHROMA_DIR, CHROMA_COLLECTION_NAME
from src.rag.chunker import Chunk
from src.rag.embedder import Embedder

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB 向量存储"""

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_dir: Path = CHROMA_DIR,
        embedder: Optional[Embedder] = None,
    ):
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._embedder = embedder
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self._get_or_create_collection()
        return self._collection

    def _get_or_create_collection(self):
        """获取或创建 collection"""
        try:
            collection = self._client.get_collection(self.collection_name)
            logger.info(f"加载已有向量库: {self.collection_name} ({collection.count()} 条)")
        except Exception:
            collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"创建新向量库: {self.collection_name}")
        return collection

    def add_chunks(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        batch_size: int = 50,
    ) -> int:
        """
        将文本块向量化后存入向量库。

        Args:
            chunks: 文本块列表
            embedder: 向量化器
            batch_size: 每批存储数量

        Returns:
            成功添加的条数
        """
        if not chunks:
            return 0

        collection = self.collection
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]

            # 向量化
            embeddings = embedder.embed_texts(texts)

            # 构造 ID 和元数据
            ids = [str(uuid.uuid4()) for _ in batch]
            metadatas = []
            for chunk in batch:
                meta = {}
                for k, v in chunk.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                metadatas.append(meta)

            try:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                total_added += len(batch)
                logger.debug(f"向量库写入批次: {len(batch)}条")
            except Exception as e:
                logger.error(f"向量库写入失败: {e}")

        logger.info(f"向量库写入完成: {total_added}条，当前总量: {collection.count()}")
        return total_added

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_domain: Optional[str] = None,
    ) -> list[dict]:
        """
        向量相似度搜索。

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数
            filter_domain: 可选的领域过滤

        Returns:
            [{id, text, metadata, distance}, ...] 排序列表
        """
        where_filter = None
        if filter_domain:
            where_filter = {"domain": filter_domain}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            formatted = []
            if results["ids"] and results["ids"][0]:
                for j in range(len(results["ids"][0])):
                    formatted.append({
                        "id": results["ids"][0][j],
                        "text": results["documents"][0][j] if results["documents"] else "",
                        "metadata": results["metadatas"][0][j] if results["metadatas"] else {},
                        "distance": results["distances"][0][j] if results["distances"] else 0.0,
                    })

            return formatted

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def search_by_text(
        self,
        query: str,
        embedder: Embedder,
        top_k: int = 5,
        filter_domain: Optional[str] = None,
    ) -> list[dict]:
        """文本查询（自动向量化）"""
        embedding = embedder.embed_query(query)
        return self.search(embedding, top_k, filter_domain)

    def get_count(self) -> int:
        """获取向量库中的文档总数"""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_domain_stats(self) -> dict:
        """获取各领域的文档统计"""
        try:
            results = self.collection.get(include=["metadatas"])
            stats = {}
            for meta in results.get("metadatas", []):
                domain = meta.get("domain", "unknown")
                stats[domain] = stats.get(domain, 0) + 1
            return stats
        except Exception:
            return {}

    def clear(self):
        """清空向量库"""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"向量库已清空: {self.collection_name}")
        except Exception as e:
            logger.warning(f"清空向量库失败: {e}")
