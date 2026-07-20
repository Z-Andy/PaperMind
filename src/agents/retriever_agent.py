"""
检索 Agent：从论文知识库中检索相关文献并返回格式化结果。
"""
import logging
from src.agents.base_agent import BaseAgent
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)

RETRIEVER_PROMPT = """你是一个"知识检索专家"。你的职责是：
1. 从用户任务中提取关键检索词
2. 对每个关键检索词从知识库中检索相关论文
3. 返回最相关的文献片段及其来源信息

注意：你会收到一个来自 Orchestrator 的任务描述，请从中提取出你需要检索的具体内容，
然后用工具函数去检索，最后将检索结果格式化返回。"""


class RetrieverAgent(BaseAgent):
    """知识检索 Agent"""

    def __init__(self, retriever: Retriever, **kwargs):
        super().__init__(
            name="Retriever",
            system_prompt=RETRIEVER_PROMPT,
            **kwargs,
        )
        self.retriever = retriever

    def execute(self, task: str, **kwargs) -> str:
        """
        执行检索任务。

        Args:
            task: 检索任务描述

        Returns:
            格式化的检索结果
        """
        # 先用 LLM 提取关键词
        kw_prompt = (
            f"任务：{task}\n\n"
            "请从以上任务中提取 2-3 个最关键的检索关键词（用逗号分隔），"
            "直接输出关键词即可，不要解释。"
        )
        keywords_str = self.think(kw_prompt).strip()

        # 解析关键词
        keywords = [kw.strip() for kw in keywords_str.replace("\n", ",").split(",") if kw.strip()]
        if not keywords:
            keywords = [task[:100]]  # fallback: 直接用任务文本

        logger.info(f"[Retriever] 检索关键词: {keywords}")

        # 多查询融合检索
        results = self.retriever.retrieve_multi_query(
            queries=keywords,
            top_k_per_query=3,
            deduplicate=True,
        )

        # 格式化结果
        if not results:
            return (
                f"检索关键词: {', '.join(keywords)}\n"
                "未找到高度相关的文献。建议扩大知识库或调整检索方向。"
            )

        context = self.retriever.format_context(results, max_chars=3000)
        summary = (
            f"检索关键词: {', '.join(keywords)}\n"
            f"知识库总量: {self.retriever.total_documents} 个文本块\n"
            f"检索到 {len(results)} 条相关文献:\n\n{context}"
        )
        return summary
