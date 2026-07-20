"""
分析 Agent：综合检索结果，进行深度分析、对比和总结。
"""
import logging
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

ANALYST_PROMPT = """你是一个"学术研究分析师"。你的职责是：
1. 阅读检索到的文献片段和研究数据
2. 提取关键发现、方法论、实验结果
3. 对不同文献进行横向对比
4. 识别研究趋势和共识/分歧
5. 输出结构化的分析报告

分析报告结构要求：
- **核心发现**：该领域最重要的结论
- **方法对比**：不同论文采用的方法及其优劣
- **趋势洞察**：研究发展方向
- **关键引用**：标注出自哪篇论文

请保持严谨、客观，不要编造不存在的信息。"""


class AnalystAgent(BaseAgent):
    """分析综合 Agent"""

    def __init__(self, **kwargs):
        super().__init__(
            name="Analyst",
            system_prompt=ANALYST_PROMPT,
            **kwargs,
        )

    def execute(self, task: str, **kwargs) -> str:
        """
        执行分析任务。

        Args:
            task: Orchestrator 传来的任务，可能包含检索结果

        Returns:
            分析报告
        """
        logger.info(f"[Analyst] 开始分析...")

        prompt = (
            f"请对以下任务进行深度分析：\n\n{task}\n\n"
            "按照分析报告的标准结构输出。"
            "如果任务中包含了检索结果或文献数据，请基于这些数据进行分析；"
            "如果仅有问题描述，请基于你的知识分析，并指出需要进一步检索的方向。"
        )
        return self.think(prompt)
