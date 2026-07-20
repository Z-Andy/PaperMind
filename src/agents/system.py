"""
多 Agent 协作系统：统一的系统入口，组装所有模块。
"""
import logging
from typing import Optional, Generator

from src.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
)
from src.crawler import ArxivCrawler
from src.crawler.scheduler import CrawlScheduler
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import Retriever
from src.agents.orchestrator import OrchestratorAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.agents.crawler_agent import CrawlerAgent

logger = logging.getLogger(__name__)


class MultiAgentSystem:
    """多 Agent 协作系统"""

    def __init__(self):
        # 基础组件
        self.embedder = Embedder()  # 从 config 自动读取 provider/model
        self.vector_store = VectorStore()
        self.rag_pipeline = RAGPipeline(self.vector_store, self.embedder)
        self.retriever = Retriever(self.vector_store, self.embedder)
        self.crawler = ArxivCrawler()

        # Agent 配置
        agent_kwargs = dict(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )

        # 创建各 Agent
        self.retriever_agent = RetrieverAgent(
            retriever=self.retriever, **agent_kwargs
        )
        self.analyst_agent = AnalystAgent(**agent_kwargs)
        self.reviewer_agent = ReviewerAgent(**agent_kwargs)
        self.crawler_agent = CrawlerAgent(
            crawler=self.crawler, **agent_kwargs
        )
        self.orchestrator = OrchestratorAgent(**agent_kwargs)

        # 注册专家 Agent
        self.orchestrator.register_agent(self.retriever_agent)
        self.orchestrator.register_agent(self.analyst_agent)
        self.orchestrator.register_agent(self.reviewer_agent)
        self.orchestrator.register_agent(self.crawler_agent)

        # 爬虫调度
        self.scheduler = CrawlScheduler(
            crawl_func=self._auto_crawl_and_ingest
        )

        logger.info("多 Agent 协作系统初始化完成")

    def _auto_crawl_and_ingest(self, domains: dict, download: bool = True) -> dict:
        """自动爬取 + 自动入链"""
        results = self.crawler.crawl_all_domains(domains, download)
        if results:
            total = sum(len(v) for v in results.values())
            if total > 0:
                self.rag_pipeline.ingest_papers()
        return results

    # ---- 用户接口 ----

    def query(self, question: str, enable_review: bool = True) -> str:
        """
        用户提问接口。

        Args:
            question: 用户问题
            enable_review: 是否启用审查 Agent 复核

        Returns:
            综合回答
        """
        logger.info(f"[Query] {question[:80]}...")

        # Step 1: 检索 Agent 查找相关文献
        retrieved = self.retriever_agent.execute(
            f"用户提问: {question}\n请检索相关论文文献。"
        )

        # Step 2: 分析 Agent 综合信息分析
        analysis_task = (
            f"用户提问: {question}\n\n"
            f"检索到的相关文献如下:\n{retrieved}\n\n"
            "请基于以上文献进行深度分析，给出详细的解答。"
        )
        analysis = self.analyst_agent.execute(analysis_task)

        # Step 3: (可选) 审查 Agent 复核
        if enable_review:
            review_task = (
                f"用户提问: {question}\n\n"
                f"分析结果如下:\n{analysis}\n\n"
                "请审查以上分析结果的准确性、完整性和逻辑性。"
            )
            review = self.reviewer_agent.execute(review_task)

            # 综合审查意见
            final = self.orchestrator.think(
                f"用户提问: {question}\n\n"
                f"分析报告:\n{analysis}\n\n"
                f"审查意见:\n{review}\n\n"
                "请综合分析和审查意见，给出最终的优化答案。"
            )
            return final

        return analysis

    def query_stream(self, question: str) -> Generator[str, None, None]:
        """
        流式查询（简化版，非真正的 token 级流式，分段输出）。

        Yields:
            处理过程中的各阶段输出
        """
        yield "🔄 **Step 1: 检索相关文献...**\n\n"

        retrieved = self.retriever_agent.execute(
            f"用户提问: {question}\n请检索相关论文文献。"
        )
        yield retrieved + "\n\n"

        yield "📊 **Step 2: 深度分析...**\n\n"

        analysis_task = (
            f"用户提问: {question}\n\n"
            f"检索文献:\n{retrieved}\n\n"
            "请基于以上文献进行深度分析。"
        )
        analysis = self.analyst_agent.execute(analysis_task)
        yield analysis + "\n\n"

        yield "✅ **Step 3: 质量审查...**\n\n"

        review_task = (
            f"分析结果:\n{analysis}\n\n请审查并给出改进建议。"
        )
        review = self.reviewer_agent.execute(review_task)
        yield review

    def crawl_domain(self, domain_name: str) -> str:
        """手动触发指定领域爬取"""
        return self.crawler_agent.execute(f"请爬取 {domain_name} 领域的论文")

    def update_knowledge_base(self, domain: Optional[str] = None) -> str:
        """
        手动触发知识库更新（爬取 + 入链）。

        Args:
            domain: 指定领域（None 则全部领域）
        """
        from src.config import RESEARCH_DOMAINS

        if domain:
            domains = {domain: RESEARCH_DOMAINS.get(domain, "cs.AI")}
        else:
            domains = RESEARCH_DOMAINS

        results = self.crawler.crawl_all_domains(domains, download=True)
        total = sum(len(v) for v in results.values())

        if total > 0:
            ingest_result = self.rag_pipeline.ingest_papers()
            return (
                f"知识库更新完成!\n"
                f"  新增论文: {total} 篇\n"
                f"  文本块: {ingest_result.get('chunks', 0)} 个\n"
                f"  向量库总量: {self.vector_store.get_count()} 条"
            )
        return "未发现新论文，知识库已是最新。"

    def get_stats(self) -> dict:
        """获取系统统计信息"""
        return {
            "vector_store": {
                "total_chunks": self.vector_store.get_count(),
                "domains": self.vector_store.get_domain_stats(),
            },
            "papers": {
                domain: len(list(pdfs))
                for domain, pdfs in self.crawler.get_stored_papers().items()
            },
            "scheduler": {
                "running": self.scheduler.scheduler.running if self.scheduler else False,
                "last_crawl": str(self.scheduler.last_crawl_time) if self.scheduler and self.scheduler.last_crawl_time else None,
                "last_result": self.scheduler.last_result_summary if self.scheduler else None,
            },
        }

    def start_scheduler(self):
        """启动定时爬虫"""
        self.scheduler.start()

    def stop_scheduler(self):
        """停止定时爬虫"""
        self.scheduler.stop()


# 全局单例
_system: Optional[MultiAgentSystem] = None


def get_system() -> MultiAgentSystem:
    """获取系统单例"""
    global _system
    if _system is None:
        _system = MultiAgentSystem()
    return _system
