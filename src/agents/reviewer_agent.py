"""
审查 Agent：校验其他 Agent 的输出质量，检查事实准确性、逻辑一致性。
"""
import logging
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

REVIEWER_PROMPT = """你是一个"研究质量审查员"。你的职责是：
1. 审查分析报告的事实准确性
2. 检查逻辑推理是否严谨
3. 识别是否存在矛盾或遗漏
4. 指出需要补充或修正的地方
5. 给出质量评分和修改建议

审查维度：
- **事实准确度** (1-10)：引用和数据是否可靠
- **逻辑完整性** (1-10)：推理链条是否完整
- **覆盖全面性** (1-10)：是否遗漏重要方面
- **实用性** (1-10)：结论是否有实际价值

审查输出格式：
```
## 审查意见
- [具体问题1]
- [具体问题2]

## 质量评分
- 事实准确度: X/10
- 逻辑完整性: X/10
- 覆盖全面性: X/10
- 实用性: X/10
- 综合: X/10

## 改进建议
- [建议1]
- [建议2]
```"""


class ReviewerAgent(BaseAgent):
    """审查验证 Agent"""

    def __init__(self, **kwargs):
        super().__init__(
            name="Reviewer",
            system_prompt=REVIEWER_PROMPT,
            **kwargs,
        )

    def execute(self, task: str, **kwargs) -> str:
        """
        执行审查任务。

        Args:
            task: 包含待审查内容的完整信息

        Returns:
            审查报告
        """
        logger.info(f"[Reviewer] 开始审查...")
        return self.think(task)
