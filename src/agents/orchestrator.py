"""
编排者 Agent：解析用户意图，拆解任务，协调各 Agent 协作。
"""
import json
import logging
from src.agents.base_agent import BaseAgent, AgentMessage

logger = logging.getLogger(__name__)

ORCHESTRATOR_PROMPT = """你是一个研究助理团队的"任务编排者"。你的职责是：
1. 理解用户的学术/技术问题
2. 将复杂问题分解为可独立执行的子任务
3. 决定调用哪些专家 Agent 来完成任务
4. 汇总各 Agent 的结果，形成最终回答

可调用的专家 Agent：
- RetrieverAgent：负责从论文知识库中检索相关文献
- AnalystAgent：负责分析和综合信息，提炼关键结论
- ReviewerAgent：负责审查答案的准确性、完整性
- CrawlerAgent：负责爬取新论文扩充知识库

输出格式要求：当你需要拆解任务时，以 JSON 格式输出子任务列表：
```json
{
  "sub_tasks": [
    {"agent": "Agent名称", "task": "具体任务描述"},
    ...
  ]
}
```

请保持回答专业、准确、有引用来源。"""


class OrchestratorAgent(BaseAgent):
    """任务编排 Agent"""

    def __init__(self, **kwargs):
        super().__init__(
            name="Orchestrator",
            system_prompt=ORCHESTRATOR_PROMPT,
            **kwargs,
        )
        self.agents: dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent):
        """注册专家 Agent"""
        self.agents[agent.name] = agent
        logger.info(f"[Orchestrator] 注册 Agent: {agent.name}")

    def decompose_task(self, user_query: str) -> list[dict]:
        """
        将用户问题分解为子任务。

        Returns:
            [{"agent": "AgentName", "task": "..."}, ...]
        """
        prompt = (
            f"用户提问：{user_query}\n\n"
            "请将这个问题拆解为子任务，明确每个子任务应该交给哪个Agent执行。"
            "以 JSON 格式输出。"
        )
        response = self.think(prompt)

        # 尝试提取 JSON
        try:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            if json_start > 6 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                sub_tasks = data.get("sub_tasks", [])
            else:
                # fallback: 整体作为一个任务交给 Analyst
                sub_tasks = [{"agent": "Analyst", "task": user_query}]
        except (json.JSONDecodeError, ValueError):
            sub_tasks = [{"agent": "Analyst", "task": user_query}]

        logger.info(f"[Orchestrator] 任务分解: {len(sub_tasks)}个子任务")
        for t in sub_tasks:
            logger.info(f"  → {t['agent']}: {t['task'][:60]}...")

        return sub_tasks

    def execute(self, task: str, **kwargs) -> str:
        """
        执行完整的多 Agent 协作流程。

        Args:
            task: 用户原始提问

        Returns:
            最终的综合回答
        """
        logger.info(f"[Orchestrator] 开始处理: {task[:80]}...")

        # Step 1: 任务分解
        sub_tasks = self.decompose_task(task)
        intermediate_results = []

        # Step 2: 委派子任务给各 Agent 执行
        for sub in sub_tasks:
            agent_name = sub["agent"]
            sub_task = sub["task"]
            agent = self.agents.get(agent_name)

            if agent:
                # 通过消息传递方式协作
                msg = self.send_message(agent_name, sub_task, "task")
                result = agent.receive_message(msg)
                intermediate_results.append({
                    "agent": agent_name,
                    "task": sub_task,
                    "result": result,
                })
            else:
                logger.warning(f"[Orchestrator] Agent [{agent_name}] 未注册，跳过")

        # Step 3: 汇总综合
        synthesis = self._synthesize(task, intermediate_results)
        return synthesis

    def _synthesize(self, original_query: str, results: list[dict]) -> str:
        """综合所有 Agent 的结果生成最终答案"""
        if not results:
            return self.think(original_query)

        context_parts = []
        for i, r in enumerate(results):
            context_parts.append(
                f"### Agent [{r['agent']}] 的输出\n{r['result']}"
            )

        context = "\n\n".join(context_parts)
        prompt = (
            f"用户原始提问：{original_query}\n\n"
            f"以下是各专家 Agent 的分析结果：\n\n{context}\n\n"
            "请综合以上信息，给出一份结构清晰、有引用依据的最终回答。"
            "如果不同 Agent 的结果存在矛盾，请指出并分析。"
            "在回答末尾标注参考的论文来源。"
        )

        return self.think(prompt)
