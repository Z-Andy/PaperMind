"""
Agent 基类：定义通用接口，所有 Agent 继承自此。
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from src.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
)

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Agent 间消息"""
    sender: str
    receiver: str
    content: str
    msg_type: str = "info"  # info, task, result, error


@dataclass
class ToolResult:
    """工具调用结果"""
    tool_name: str
    success: bool
    result: str
    error: Optional[str] = None


class BaseAgent(ABC):
    """所有 Agent 的基类"""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = LLM_MODEL,
        api_key: str = LLM_API_KEY,       
        base_url: str = LLM_BASE_URL,     
        temperature: float = LLM_TEMPERATURE,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.messages: list[dict] = []
        self.tools: dict[str, callable] = {}
        self._conversation_history: list[AgentMessage] = []

    def _build_messages(self, user_message: str) -> list[dict]:
        """构建消息列表"""
        return [
            {"role": "system", "content": self.system_prompt},
            *self.messages[-10:],  # 保留最近10轮对话
            {"role": "user", "content": user_message},
        ]

    def think(self, user_message: str) -> str:
        """
        调用 LLM 进行推理（纯文本，无工具调用）。

        Returns:
            LLM 的文本响应
        """
        messages = self._build_messages(user_message)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            reply = response.choices[0].message.content or ""

            self.messages.append({"role": "user", "content": user_message})
            self.messages.append({"role": "assistant", "content": reply})

            logger.debug(f"[{self.name}] 响应: {reply[:100]}...")
            return reply

        except Exception as e:
            logger.error(f"[{self.name}] LLM 调用失败: {e}")
            return f"Agent [{self.name}] 处理失败: {str(e)}"

    def send_message(self, receiver: str, content: str, msg_type: str = "info") -> AgentMessage:
        """向其他 Agent 发送消息"""
        msg = AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            msg_type=msg_type,
        )
        self._conversation_history.append(msg)
        return msg

    def receive_message(self, message: AgentMessage) -> str:
        """接收来自其他 Agent 的消息并处理"""
        prefix = f"来自 [{message.sender}] 的消息 ({message.msg_type}):\n"
        return self.think(prefix + message.content)

    @abstractmethod
    def execute(self, task: str, **kwargs) -> str:
        """执行当前 Agent 的主任务"""
        ...

    def __repr__(self) -> str:
        return f"<Agent:{self.name}>"
