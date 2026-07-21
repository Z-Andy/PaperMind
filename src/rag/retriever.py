"""
检索器：支持混合检索（BM25 + 向量）和 Cross-Encoder 重排序。
"""
import logging
from typing import Optional

from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """知识库检索器（混合检索 + 重排序）"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        default_top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.default_top_k = default_top_k

        # BM25 缓存
        self._bm25_corpus: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_metadatas: list[dict] = []
        self._bm25 = None
        self._corpus_size: int = 0

        # Cross-Encoder 延迟加载
        self._cross_encoder = None

    # ============================================================
    # 混合检索
    # ============================================================

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_domain: Optional[str] = None,
        min_relevance: float = 0.0,
        use_hybrid: bool = True,
        use_rerank: bool = True,
    ) -> list[dict]:
        """
        检索相关文档块（默认混合检索 + 重排序）。

        Args:
            query: 查询文本
            top_k: 最终返回数量
            filter_domain: 限定领域
            min_relevance: 最小相关度阈值
            use_hybrid: 是否启用 BM25+向量混合检索
            use_rerank: 是否启用 Cross-Encoder 重排序
        """
        if top_k is None:
            top_k = self.default_top_k

        if use_hybrid:
            # 混合检索：向量 + BM25，各取 top_k*3 候选
            candidates = self._retrieve_hybrid(
                query, top_k * 3, filter_domain
            )
        else:
            # 纯向量检索
            candidates = self._retrieve_vector(
                query, top_k * 3, filter_domain
            )

        if not candidates:
            return []

        # 重排序
        if use_rerank and len(candidates) > top_k:
            candidates = self._rerank(query, candidates, top_k)

        # 过滤低相关度结果
        results = [r for r in candidates if r["score"] >= min_relevance]
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        logger.info(
            f"检索: '{query[:50]}...' → {len(results)}条"
            + (f" (最高分: {results[0]['score']:.3f})" if results else " (无结果)")
        )
        return results

    def _retrieve_vector(
        self, query: str, top_k: int, filter_domain: Optional[str] = None
    ) -> list[dict]:
        """纯向量检索"""
        results = self.vector_store.search_by_text(
            query=query,
            embedder=self.embedder,
            top_k=top_k,
            filter_domain=filter_domain,
        )
        for r in results:
            r["score"] = max(0.0, 1.0 - r.get("distance", 1.0))
        return results

    def _retrieve_hybrid(
        self, query: str, top_k: int, filter_domain: Optional[str] = None
    ) -> list[dict]:
        """
        BM25 + 向量 混合检索，RRF 融合。

        流程：
        1. 向量检索 → top_k 条
        2. BM25 关键词检索 → top_k 条
        3. Reciprocal Rank Fusion 合并排名
        """
        # 向量检索
        vector_results = self._retrieve_vector(query, top_k, filter_domain)

        # BM25 检索
        bm25_results = self._bm25_search(query, top_k)

        if not bm25_results:
            logger.debug("BM25 未命中，回退纯向量检索")
            return vector_results

        # RRF 融合
        merged = self._reciprocal_rank_fusion(vector_results, bm25_results, k=60)
        logger.debug(
            f"混合检索: 向量{len(vector_results)} + BM25{len(bm25_results)}"
            f" → RRF融合{len(merged)}"
        )
        return merged

    # ============================================================
    # BM25 关键词检索
    # ============================================================

    def _ensure_bm25_index(self):
        """确保 BM25 索引是最新的"""
        current_count = self.vector_store.get_count()
        if self._bm25 is not None and current_count == self._corpus_size:
            return

        # 重建索引
        try:
            from rank_bm25 import BM25Okapi

            # 从 ChromaDB 获取所有文本和元数据
            all_data = self.vector_store.collection.get(
                include=["documents", "metadatas"]
            )
            texts = all_data.get("documents", [])
            ids = all_data.get("ids", [])
            metadatas = all_data.get("metadatas", [])

            if not texts:
                self._bm25 = None
                return

            # 英文分词（简单空格分词）
            tokenized = [
                doc.lower().split() if doc else []
                for doc in texts
            ]

            self._bm25 = BM25Okapi(tokenized)
            self._bm25_corpus = texts
            self._bm25_ids = ids
            self._bm25_metadatas = metadatas
            self._corpus_size = current_count

            logger.debug(f"BM25 索引已更新: {len(texts)} 篇文档")

        except ImportError:
            logger.warning("rank_bm25 未安装，BM25 检索不可用")
            self._bm25 = None
        except Exception as e:
            logger.warning(f"BM25 索引构建失败: {e}")
            self._bm25 = None

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 关键词检索"""
        self._ensure_bm25_index()

        if self._bm25 is None:
            return []

        try:
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)

            # 取 top_k
            if not scores.any():
                return []

            indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:top_k]

            # 归一化分数到 0~1
            max_score = max(scores) if max(scores) > 0 else 1

            results = []
            for idx in indices:
                if scores[idx] <= 0:
                    continue
                meta = self._bm25_metadatas[idx] if idx < len(self._bm25_metadatas) else {}
                results.append({
                    "id": self._bm25_ids[idx],
                    "text": self._bm25_corpus[idx],
                    "metadata": meta,
                    "score": min(1.0, scores[idx] / max_score),
                    "source": "bm25",
                })

            return results

        except Exception as e:
            logger.warning(f"BM25 检索异常: {e}")
            return []

    # ============================================================
    # RRF 融合
    # ============================================================

    @staticmethod
    def _reciprocal_rank_fusion(
        results_a: list[dict],
        results_b: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion：合并两个排序列表。

        原理：不比较分数绝对值，只看排名。
        RRF_score(doc) = Σ 1/(k + rank_i(doc))
        排名越靠前，贡献越大。

        Args:
            results_a: 第一组结果（如向量检索）
            results_b: 第二组结果（如 BM25）
            k: 平滑参数，典型值 60
        """
        rrf_scores: dict[str, tuple[float, dict]] = {}

        for rank, r in enumerate(results_a, start=1):
            doc_id = r.get("id", "")
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = (0.0, r)
            score, _ = rrf_scores[doc_id]
            rrf_scores[doc_id] = (score + 1.0 / (k + rank), r)

        for rank, r in enumerate(results_b, start=1):
            doc_id = r.get("id", "")
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = (0.0, r)
            score, _ = rrf_scores[doc_id]
            rrf_scores[doc_id] = (score + 1.0 / (k + rank), r)

        merged = [
            {**doc, "score": rrf_score}
            for doc_id, (rrf_score, doc) in rrf_scores.items()
        ]
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged

    # ============================================================
    # Cross-Encoder 重排序
    # ============================================================

    def preload_models(self):
        """预加载 Cross-Encoder 模型（在系统启动时调用，避免首次查询卡顿）"""
        self._load_cross_encoder()

    def _load_cross_encoder(self):
        """延迟加载 Cross-Encoder 模型"""
        if self._cross_encoder is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            # 也支持中文: "BAAI/bge-reranker-v2-m3"
            logger.info(f"正在加载重排序模型: {model_name}（首次加载需下载 ~80MB，请耐心等待）...")
            self._cross_encoder = CrossEncoder(model_name)
            logger.info("重排序模型加载完成")

        except ImportError:
            logger.warning("sentence-transformers 未安装，重排序不可用")
            self._cross_encoder = False
        except Exception as e:
            logger.warning(f"重排序模型加载失败: {e}")
            self._cross_encoder = False

    def _rerank(
        self, query: str, candidates: list[dict], top_k: int
    ) -> list[dict]:
        """
        Cross-Encoder 重排序。

        与向量检索的 bi-encoder 不同，cross-encoder 将 query 和文档
        同时输入模型，做完整的交叉注意力计算，精度更高但速度较慢。
        所以仅在候选集较小（< 30条）时使用。
        """
        self._load_cross_encoder()

        if self._cross_encoder in (None, False) or len(candidates) <= top_k:
            return candidates

        try:
            # 构造 (query, document) 对
            pairs = [(query, c.get("text", "")) for c in candidates]
            scores = self._cross_encoder.predict(pairs)

            # 归一化
            min_s = min(scores) if len(scores) > 0 else 0
            max_s = max(scores) if len(scores) > 0 else 1
            score_range = max_s - min_s if max_s > min_s else 1

            for i, c in enumerate(candidates):
                c["score"] = (scores[i] - min_s) / score_range
                c["source"] = c.get("source", "vector") + "+rerank"

            candidates.sort(key=lambda x: x["score"], reverse=True)
            logger.debug(f"重排序: {len(candidates)}条 → top_{top_k}")

            return candidates[:top_k]

        except Exception as e:
            logger.warning(f"重排序异常: {e}，回退原始排序")
            return candidates

    # ============================================================
    # 多查询融合检索
    # ============================================================

    def retrieve_multi_query(
        self,
        queries: list[str],
        top_k_per_query: int = 3,
        deduplicate: bool = True,
    ) -> list[dict]:
        """多查询融合检索（每个子查询做混合检索）"""
        all_results = []
        seen_ids = set()

        for query in queries:
            results = self.retrieve(
                query,
                top_k=top_k_per_query * 2,
                use_hybrid=True,
                use_rerank=False,  # 各子查询先不做重排，统一再做
            )
            for r in results:
                if deduplicate:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        all_results.append(r)
                else:
                    all_results.append(r)

        # 统一重排序
        if len(all_results) > top_k_per_query * len(queries):
            all_queries = " ".join(queries)
            all_results = self._rerank(
                all_queries, all_results, top_k_per_query * len(queries)
            )

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results

    def format_context(self, results: list[dict], max_chars: int = 4000) -> str:
        """检索结果格式化为 LLM 上下文"""
        if not results:
            return "未找到相关文献。请尝试更换关键词或扩大检索范围。"

        parts = []
        total_chars = 0

        for i, r in enumerate(results):
            source = r.get("metadata", {}).get("title", "未知来源")
            text = r.get("text", "")
            source_tag = r.get('source', '')
            if source_tag:
                score_src = f"(相关性: {r['score']:.2f}, {source_tag})"
            else:
                score_src = f"(相关性: {r['score']:.2f})"

            entry = f"[文献{i + 1}] 来源: {source} {score_src}\n{text}\n"

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
