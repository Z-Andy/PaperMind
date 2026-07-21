"""
CrewAI 工具：封装现有功能为 Agent 可调用的 Tool。
"""
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class RetrieveInput(BaseModel):
    """检索工具输入参数"""
    queries: str = Field(
        description=(
            "检索关键词，多个关键词用英文逗号分隔，"
            "例如 'attention mechanism, transformer optimization'"
        )
    )


class RetrievePapersTool(BaseTool):
    """知识库论文检索工具"""
    name: str = "retrieve_papers"
    description: str = (
        "从论文向量知识库中检索与给定关键词最相关的文献片段。"
        "支持多关键词融合检索，自动去重并排序。"
        "返回格式化的检索结果，包含论文来源、内容和相关性分数。"
    )
    args_schema: Type[BaseModel] = RetrieveInput

    retriever: Retriever = None

    def _run(self, queries: str) -> str:
        """
        执行多关键词融合检索。

        Args:
            queries: 逗号分隔的关键词字符串

        Returns:
            格式化的检索结果文本
        """
        try:
            keywords = [
                kw.strip()
                for kw in queries.replace("\n", ",").split(",")
                if kw.strip()
            ]
            if not keywords:
                return "检索失败：未提供有效的关键词。"

            logger.info(f"[RetrieveTool] 检索关键词: {keywords}")

            results = self.retriever.retrieve_multi_query(
                queries=keywords,
                top_k_per_query=3,
                deduplicate=True,
            )

            if not results:
                return (
                    f"检索关键词: {', '.join(keywords)}\n"
                    "未找到相关文献。建议更换关键词或扩大知识库。"
                )

            context = self.retriever.format_context(results, max_chars=4000)
            return (
                f"检索关键词: {', '.join(keywords)}\n"
                f"知识库总量: {self.retriever.total_documents} 个文本块\n"
                f"命中结果: {len(results)} 条\n\n"
                f"{context}"
            )

        except Exception as e:
            logger.error(f"检索工具异常: {e}")
            return f"检索过程出错: {str(e)}"
