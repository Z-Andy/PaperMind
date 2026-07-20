"""
爬虫任务调度器：支持定时自动爬取和手动触发。
"""
import logging
from datetime import datetime
from typing import Optional, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from src.config import ARXIV_CRAWL_INTERVAL_HOURS, RESEARCH_DOMAINS

logger = logging.getLogger(__name__)


class CrawlScheduler:
    """论文爬取调度器"""

    def __init__(self, crawl_func: Callable, interval_hours: int = ARXIV_CRAWL_INTERVAL_HOURS):
        """
        Args:
            crawl_func: 爬取函数，签名为 (domains, download) -> dict
            interval_hours: 定时爬取间隔（小时）
        """
        self.crawl_func = crawl_func
        self.interval_hours = interval_hours
        self.scheduler = BackgroundScheduler(
            job_defaults={"max_instances": 1, "coalesce": True}
        )
        self._last_crawl_time: Optional[datetime] = None
        self._last_result: Optional[dict] = None
        self._is_running = False

    def _crawl_job(self):
        """定时任务执行函数"""
        if self._is_running:
            logger.warning("上一次爬取尚未完成，跳过本次调度")
            return

        self._is_running = True
        try:
            logger.info(f"[定时任务] 开始自动爬取，领域数: {len(RESEARCH_DOMAINS)}")
            result = self.crawl_func(RESEARCH_DOMAINS, download=True)
            self._last_result = result
            self._last_crawl_time = datetime.now()
            total = sum(len(v) for v in result.values())
            logger.info(f"[定时任务] 爬取完成，共 {total} 篇论文")
        except Exception as e:
            logger.error(f"[定时任务] 爬取出错: {e}", exc_info=True)
        finally:
            self._is_running = False

    def start(self):
        """启动定时调度"""
        self.scheduler.add_job(
            self._crawl_job,
            trigger=IntervalTrigger(hours=self.interval_hours),
            id="arxiv_crawl",
            name="arXiv论文自动爬取",
            replace_existing=True,
        )
        self.scheduler.start()
        logger.info(f"爬虫调度器已启动，间隔: {self.interval_hours}h")

    def stop(self):
        """停止调度"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("爬虫调度器已停止")

    def trigger_now(self) -> dict:
        """手动立即触发一次爬取"""
        logger.info("[手动触发] 开始爬取...")
        self._crawl_job()
        return self._last_result or {}

    @property
    def last_crawl_time(self) -> Optional[datetime]:
        return self._last_crawl_time

    @property
    def last_result_summary(self) -> Optional[dict]:
        """上次爬取结果摘要"""
        if not self._last_result:
            return None
        return {
            domain: len(papers)
            for domain, papers in self._last_result.items()
        }

    @property
    def is_running(self) -> bool:
        return self._is_running
