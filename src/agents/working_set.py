"""
Sub-Agent 工作集管理器：FIFO 队列维护当前讨论中引用的论文片段。
主 Agent 不直接读取片段全文，而是通过工作集查询摘要。
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.config import (
    WORKING_SET_MAX_FRAGMENTS,
    WORKING_SET_RELEVANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class Fragment:
    """工作集中的一条论文片段"""
    id: str
    text: str                       # 原始检索文本（~2000 chars）
    embedding: list[float]          # 预计算向量
    source: str                     # 论文标题
    score: float                    # 检索得分
    added_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def touch(self):
        """被引用时刷新时间戳"""
        self.last_accessed = time.time()
        self.access_count += 1

    @property
    def age(self) -> float:
        return time.time() - self.added_at

    @property
    def idle_time(self) -> float:
        return time.time() - self.last_accessed


class RelevanceResult:
    """相关性查询结果"""
    def __init__(
        self,
        hit: bool,
        fragments: list[Fragment] = None,
        confidence: float = 0.0,
        reason: str = "",
        suggested_query: Optional[str] = None,
    ):
        self.hit = hit
        self.fragments = fragments or []
        self.confidence = confidence
        self.reason = reason
        self.suggested_query = suggested_query

    @property
    def summary(self) -> str:
        """生成工作集摘要，返回给主 Agent"""
        if not self.fragments:
            return "工作集中无相关论文片段。"

        lines = ["【工作集摘要】"]
        lines.append(f"当前工作集含 {len(self.fragments)} 条相关片段：")
        for i, frag in enumerate(self.fragments):
            lines.append(
                f"{i+1}. [{frag.source}] (引用 {frag.access_count} 次) "
                f"{frag.text[:300]}..."
            )
        return "\n".join(lines)


class WorkingSetManager:
    """
    工作集 Sub-Agent：FIFO 队列管理论文片段。

    职责：
    1. 接收新检索到的片段并入队（队满驱逐最早且最少使用的）
    2. 按向量相似度 + 关键词判断问题是否命中已有片段
    3. 命中时返回摘要（而非全文）；未命中时返回检索建议词
    4. 被引用的片段自动刷新到队尾（FIFO 优先驱逐未被引用的）
    """

    def __init__(self, embedder, max_fragments: int = WORKING_SET_MAX_FRAGMENTS):
        self._queue: list[Fragment] = []
        self._max_fragments = max_fragments
        self._embedder = embedder
        # 简单 BM25 关键词索引（{word: {fragment_id: count}}）
        self._keyword_index: dict[str, dict[str, int]] = {}
        self._total_added = 0
        self._total_evicted = 0

    # ---- 入队 ----

    def add(self, fragment: Fragment) -> Fragment | None:
        """
        添加片段到工作集。队满时驱逐策略：
        优先驱逐从未被引用过的片段，其次驱逐最久未访问的。
        """
        existing = self._find_by_id(fragment.id)
        if existing:
            # 已存在 → 刷新时间戳并移到队尾
            existing.touch()
            existing.score = max(existing.score, fragment.score)
            logger.debug(f"[WorkingSet] 刷新已有片段: {fragment.source[:30]}...")
            return None

        evicted = None
        if len(self._queue) >= self._max_fragments:
            evicted = self._evict_one()

        self._queue.append(fragment)
        self._total_added += 1
        self._update_keyword_index(fragment)
        logger.info(
            f"[WorkingSet] 入队: {fragment.source[:40]}... "
            f"(队列: {len(self._queue)}/{self._max_fragments})"
        )
        return evicted

    def add_batch(self, fragments: list[Fragment]) -> list[Fragment]:
        """批量添加"""
        evicted = []
        for f in fragments:
            e = self.add(f)
            if e:
                evicted.append(e)
        return evicted

    # ---- 驱逐 ----

    def _evict_one(self) -> Fragment:
        """驱逐一条片段：优先踢从未被引用 + 入队最早的"""
        self._queue.sort(key=lambda f: (f.access_count > 0, f.added_at))
        evicted = self._queue.pop(0)
        self._total_evicted += 1
        self._remove_from_keyword_index(evicted)
        logger.info(
            f"[WorkingSet] 驱逐: {evicted.source[:30]}... "
            f"(access_count={evicted.access_count}, age={evicted.age:.0f}s)"
        )
        return evicted

    # ---- 查询 ----

    def query(self, question: str) -> RelevanceResult:
        """
        查询工作集是否包含与 question 相关的片段。

        两层过滤：
        1. 关键词 BM25 预筛（快速）
        2. 向量余弦相似度验证（精确）
        """
        if not self._queue:
            return RelevanceResult(hit=False, reason="empty")

        # 第 1 层：关键词预筛
        candidates = self._bm25_filter(question)
        if not candidates:
            return RelevanceResult(
                hit=False,
                reason="keyword_mismatch",
                suggested_query=self._extract_keywords(question),
            )

        # 第 2 层：向量相似度验证
        try:
            question_vec = self._embedder.embed_query(question)
        except Exception as e:
            logger.warning(f"[WorkingSet] 向量化失败，回退关键词匹配: {e}")
            return RelevanceResult(
                hit=True,
                fragments=candidates[:3],
                confidence=0.5,
                reason="vector_fallback",
            )

        matched = []
        for frag in candidates:
            sim = self._cosine_similarity(question_vec, frag.embedding)
            if sim >= WORKING_SET_RELEVANCE_THRESHOLD:
                frag.touch()
                matched.append((frag, sim))

        if not matched:
            return RelevanceResult(
                hit=False,
                reason="semantic_mismatch",
                suggested_query=self._extract_keywords(question),
            )

        # 按相似度排序
        matched.sort(key=lambda x: x[1], reverse=True)
        fragments = [f for f, _ in matched]
        return RelevanceResult(
            hit=True,
            fragments=fragments,
            confidence=matched[0][1],
        )

    # ---- BM25 关键词索引 ----

    def _update_keyword_index(self, fragment: Fragment):
        """将新片段的关键词加入索引"""
        words = self._tokenize(fragment.text)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        for w, c in word_counts.items():
            if w not in self._keyword_index:
                self._keyword_index[w] = {}
            self._keyword_index[w][fragment.id] = c

    def _remove_from_keyword_index(self, fragment: Fragment):
        """从关键词索引中移除片段"""
        words = set(self._tokenize(fragment.text))
        for w in words:
            if w in self._keyword_index:
                self._keyword_index[w].pop(fragment.id, None)
                if not self._keyword_index[w]:
                    del self._keyword_index[w]

    def _bm25_filter(self, question: str) -> list[Fragment]:
        """BM25 关键词预筛：返回命中的片段"""
        query_words = self._tokenize(question)
        if not query_words:
            return list(self._queue)

        scores: dict[str, float] = {}
        k1, b = 1.2, 0.75
        avg_len = self._avg_text_length()
        N = len(self._queue)

        for word in query_words:
            if word not in self._keyword_index:
                continue
            postings = self._keyword_index[word]
            df = len(postings)
            idf = np.log(1 + (N - df + 0.5) / (df + 0.5))

            for fid, tf in postings.items():
                frag = self._find_by_id(fid)
                if not frag:
                    continue
                doc_len = len(frag.text)
                norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
                scores[fid] = scores.get(fid, 0.0) + idf * norm_tf

        if not scores:
            return []

        # 排序并返回所有超过阈值的
        threshold = max(0.1, min(scores.values()) * 0.3)
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        result = []
        for fid in sorted_ids:
            if scores[fid] >= threshold:
                frag = self._find_by_id(fid)
                if frag:
                    result.append(frag)
        return result

    def _avg_text_length(self) -> float:
        if not self._queue:
            return 1.0
        return sum(len(f.text) for f in self._queue) / len(self._queue)

    # ---- 工具方法 ----

    def _find_by_id(self, fragment_id: str) -> Fragment | None:
        for f in self._queue:
            if f.id == fragment_id:
                return f
        return None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单中英文分词"""
        import re
        # 英文单词 + 中文字符分别提取
        tokens = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]+|\d+', text.lower())
        # 去重+去短词
        return [t for t in tokens if len(t) > 1]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """余弦相似度"""
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def _extract_keywords(text: str) -> str:
        """从问题中提取关键词作为检索建议"""
        tokens = WorkingSetManager._tokenize(text)
        return " ".join(tokens[:5]) if tokens else text

    # ---- 状态查询 ----

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def is_full(self) -> bool:
        return len(self._queue) >= self._max_fragments

    def get_stats(self) -> dict:
        return {
            "queue_size": len(self._queue),
            "max_size": self._max_fragments,
            "total_added": self._total_added,
            "total_evicted": self._total_evicted,
            "fragments": [
                {
                    "source": f.source,
                    "access_count": f.access_count,
                    "age_s": round(f.age, 0),
                    "idle_s": round(f.idle_time, 0),
                }
                for f in self._queue
            ],
        }

    def clear(self):
        self._queue.clear()
        self._keyword_index.clear()
        logger.info("[WorkingSet] 工作集已清空")
