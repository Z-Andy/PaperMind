"""
检索器：封装检索逻辑，支持混合策略和多轮优化。
"""
import logging
from typing import Optional

from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """知识库检索器"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        default_top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_domain: Optional[str] = None,
        min_relevance: float = 0.0,
    ) -> list[dict]:
        """
        检索相关文档块。

        Args:
            query: 查询文本
            top_k: 返回数量
            filter_domain: 限定领域
            min_relevance: 最小相关度阈值（0~1，越大越严格）

        Returns:
            检索结果列表，已按相关性排序
        """
        if top_k is None:
            top_k = self.default_top_k

        results = self.vector_store.search_by_text(
            query=query,
            embedder=self.embedder,
            top_k=top_k * 2,  # 多取一些用于过滤
            filter_domain=filter_domain,
        )

        # 距离 → 相似度转换（cosine distance = 1 - similarity）
        for r in results:
            r["score"] = max(0.0, 1.0 - r.get("distance", 1.0))

        # 按相似度排序并过滤
        results = [r for r in results if r["score"] >= min_relevance]
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        logger.info(
            f"检索: '{query[:50]}...' → {len(results)}条结果 "
            f"(最高分: {results[0]['score']:.3f})" if results else f"检索: '{query[:50]}...' → 无结果"
        )
        return results

    def retrieve_multi_query(
        self,
        queries: list[str],
        top_k_per_query: int = 3,
        deduplicate: bool = True,
    ) -> list[dict]:
        """
        多查询融合检索（适用于复杂问题拆解后的子查询）。

        Args:
            queries: 多个子查询
            top_k_per_query: 每个子查询取几条
            deduplicate: 是否去重

        Returns:
            合并后的检索结果
        """
        all_results = []
        seen_ids = set()

        for query in queries:
            results = self.retrieve(query, top_k=top_k_per_query)
            for r in results:
                if deduplicate:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        all_results.append(r)
                else:
                    all_results.append(r)

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results

    def format_context(self, results: list[dict], max_chars: int = 4000) -> str:
        """
        将检索结果格式化为 LLM 可用的上下文文本。

        Args:
            results: 检索结果列表
            max_chars: 最大字符数

        Returns:
            格式化后的上下文字符串
        """
        if not results:
            return "未找到相关文献。请尝试更换关键词或扩大检索范围。"

        parts = []
        total_chars = 0

        for i, r in enumerate(results):
            source = r.get("metadata", {}).get("title", "未知来源")
            text = r.get("text", "")

            entry = (
                f"[文献{i + 1}] 来源: {source} (相关性: {r['score']:.2f})\n"
                f"{text}\n"
            )

            if total_chars + len(entry) > max_chars:
                entry = entry[:max_chars - total_chars] + "..."
                parts.append(entry)
                break

            parts.append(entry)
            total_chars += len(entry)

        return "\n---\n".join(parts)

    @property
    def total_documents(self) -> int:
        return self.vector_store.get_count()
