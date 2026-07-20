"""
爬取 Agent：管理论文爬取和知识库更新任务。
"""
import logging
from src.agents.base_agent import BaseAgent
from src.crawler import ArxivCrawler

logger = logging.getLogger(__name__)

CRAWLER_PROMPT = """你是一个"知识库管理 Agent"。你的职责是：
1. 根据用户需求选择要爬取的研究领域
2. 触发论文爬取流程
3. 在爬取完成后将新论文入库
4. 向用户汇报知识库更新情况

你不需要自己分析论文内容，那是 Analyst 的职责。"""


class CrawlerAgent(BaseAgent):
    """论文爬取管理 Agent"""

    def __init__(self, crawler: ArxivCrawler, **kwargs):
        super().__init__(
            name="Crawler",
            system_prompt=CRAWLER_PROMPT,
            **kwargs,
        )
        self.crawler = crawler
        self._last_crawl_result: dict = {}

    def execute(self, task: str, **kwargs) -> str:
        """
        执行爬取任务。

        Args:
            task: 描述需要爬取的领域（如 "多智能体系统 和 大语言模型"）

        Returns:
            爬取结果摘要
        """
        logger.info(f"[Crawler] 开始处理爬取任务...")

        # 用 LLM 解析用户指定的领域
        from src.config import RESEARCH_DOMAINS

        domain_list = ", ".join(RESEARCH_DOMAINS.keys())
        parse_prompt = (
            f"已知可选领域: {domain_list}\n\n"
            f"用户需求: {task}\n\n"
            f"请从中选择用户可能关心的领域（用逗号分隔）。"
            f"直接输出领域名即可。"
        )
        selected = self.think(parse_prompt).strip()

        # 匹配领域
        domains_to_crawl = {}
        for name, cat in RESEARCH_DOMAINS.items():
            if name in selected:
                domains_to_crawl[name] = cat

        if not domains_to_crawl:
            # 默认爬取 AI 和 大语言模型
            domains_to_crawl = {
                "人工智能": "cs.AI",
                "大语言模型": "cs.CL",
            }

        logger.info(f"[Crawler] 目标领域: {list(domains_to_crawl.keys())}")

        # 执行爬取
        results = self.crawler.crawl_all_domains(
            domains=domains_to_crawl,
            download=True,
        )
        self._last_crawl_result = results

        # 生成汇报
        total = sum(len(v) for v in results.values())
        domain_details = "\n".join(
            f"  - {domain}: {len(papers)} 篇"
            for domain, papers in results.items()
        )

        report = (
            f"## 知识库更新完成\n"
            f"共爬取 **{total}** 篇新论文\n\n"
            f"{domain_details}\n\n"
            f"新论文已存入知识库，可以开始提问了。"
        )
        return report

    @property
    def last_crawl_result(self) -> dict:
        return self._last_crawl_result
