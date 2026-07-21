"""
指标收集与统计模块：量化系统的响应时间、缓存命中率、LLM 调用次数等。

使用方式：
    from src.metrics import get_metrics
    metrics = get_metrics()

    with metrics.track("retrieve"):
        results = retriever.retrieve(...)

    metrics.count("cache_hit")
    metrics.count("llm_call")
    metrics.add_tokens("rewrite", input=80, output=15)
"""

import logging
import statistics
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


class MetricsTracker:
    """线程安全的指标收集器（内存存储，重启清零）"""

    def __init__(self, max_recent: int = 200):
        self._lock = threading.Lock()
        self._max_recent = max_recent

        # ---- 计数器 ----
        self._counters: dict[str, int] = defaultdict(int)

        # ---- 时间统计（秒） ----
        # key → [values...]
        self._timings: dict[str, list[float]] = defaultdict(list)

        # ---- Token 统计 ----
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # ---- 最近查询明细 ----
        self._recent_queries: list[dict] = []

        # ---- 启动时间 ----
        self._start_time = time.time()

    # ================================================================
    # 计数器
    # ================================================================

    def count(self, key: str, delta: int = 1):
        """计数器 +1（如 cache_hit、llm_call、query_total 等）"""
        with self._lock:
            self._counters[key] += delta

    def get_count(self, key: str) -> int:
        with self._lock:
            return self._counters.get(key, 0)

    # ================================================================
    # 计时
    # ================================================================

    def record_timing(self, key: str, seconds: float):
        """记录一次耗时"""
        with self._lock:
            self._timings[key].append(seconds)

    @contextmanager
    def track(self, key: str):
        """上下文管理器：自动记录耗时。

        with metrics.track("retrieve"):
            do_retrieval()
        """
        t0 = time.time()
        try:
            yield
        finally:
            self.record_timing(key, time.time() - t0)

    def get_timing_stats(self, key: str) -> dict:
        """获取某步骤的耗时统计（avg/p50/p95/p99/min/max）"""
        with self._lock:
            values = self._timings.get(key, [])
        if not values:
            return {"count": 0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def _pct(p):
            idx = max(0, min(n - 1, int(n * p / 100)))
            return sorted_vals[idx]

        return {
            "count": n,
            "avg": round(sum(values) / n, 2),
            "p50": round(_pct(50), 2),
            "p95": round(_pct(95), 2),
            "p99": round(_pct(99), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "total": round(sum(values), 2),
        }

    # ==============================================================
    # Token 统计
    # ================================================================

    def add_tokens(self, step: str, input_tokens: int = 0, output_tokens: int = 0):
        with self._lock:
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self.count(f"tokens_{step}_input", input_tokens)
            self.count(f"tokens_{step}_output", output_tokens)

    def get_token_stats(self) -> dict:
        with self._lock:
            return {
                "total_input": self._total_input_tokens,
                "total_output": self._total_output_tokens,
            }

    # ================================================================
    # 查询明细日志
    # ================================================================

    def log_query(self, entry: dict):
        """记录一次完整的查询详情"""
        with self._lock:
            self._recent_queries.append(entry)
            if len(self._recent_queries) > self._max_recent:
                self._recent_queries = self._recent_queries[-self._max_recent:]

    def get_recent_queries(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return list(reversed(self._recent_queries[-limit:]))

    # ================================================================
    # 汇总报告
    # ================================================================

    def get_summary(self) -> dict:
        """生成汇总统计报告"""
        with self._lock:
            total_queries = self._counters.get("query_total", 0)
            cache_hits = self._counters.get("cache_hit", 0)
            cache_misses = total_queries - cache_hits

            # 总耗时
            total_times = self._timings.get("total", [])
            avg_total = (
                round(sum(total_times) / len(total_times), 2)
                if total_times else 0
            )

        # 各步骤统计
        steps = {}
        for key in ["rewrite", "retrieve", "analyze", "review", "synthesize", "total"]:
            steps[key] = self.get_timing_stats(key)

        return {
            "overview": {
                "uptime_seconds": round(time.time() - self._start_time),
                "total_queries": total_queries,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_rate": (
                    round(cache_hits / total_queries * 100, 1)
                    if total_queries > 0 else 0
                ),
                "avg_total_seconds": avg_total,
            },
            "timings": steps,
            "tokens": self.get_token_stats(),
            "counters": dict(self._counters),
        }


# ---- 全局单例 ----

_metrics: Optional[MetricsTracker] = None


def get_metrics() -> MetricsTracker:
    global _metrics
    if _metrics is None:
        _metrics = MetricsTracker()
    return _metrics
