from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.runtime_base import BaseAgentRuntime
from app.runtime_helpers import compress_tool_result
from app.skills import SkillStore
from app.tools import AgentTools


@dataclass
class ADKMessage:
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class _OpenAICompatADKSession:
    """Fallback session that mimics ADK turn behavior using a tool-bound chat model."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def invoke(self, history: list[ADKMessage]) -> ADKMessage:
        lc_history: list[BaseMessage] = []
        for item in history:
            if item.role == "system":
                lc_history.append(SystemMessage(content=item.content))
            elif item.role == "assistant":
                lc_history.append(AIMessage(content=item.content, tool_calls=item.tool_calls or []))
            elif item.role == "tool":
                lc_history.append(ToolMessage(content=item.content, tool_call_id=item.tool_call_id or "", name=item.name))
            else:
                lc_history.append(HumanMessage(content=item.content))

        ai_message: AIMessage = self._llm.invoke(lc_history)
        return ADKMessage(
            role="assistant",
            content=str(ai_message.content),
            tool_calls=ai_message.tool_calls,
        )


class GoogleADKAgentRuntime(BaseAgentRuntime):
    def __init__(
        self,
        model: str = "qwen3-32b",
        skills_dir: str = "skills",
        conversation_log_path: str | Path = "logs/agent_conversations.jsonl",
    ) -> None:
        super().__init__(conversation_log_path=conversation_log_path)
        self.skill_store = SkillStore(skills_dir)
        self.tools = AgentTools(self.skill_store)
        self._logger = logging.getLogger(__name__)
        self.default_model = model
        self.default_base_url = os.getenv("GOOGLE_ADK_BASE_URL", "http://api.openai.rnd.huawei.com/v1/")
        self._default_session = asyncio.run(self._build_adk_session(model=model))

    async def _build_adk_session(
        self,
        model: str,
        session_id: str | None = None,
        base_url: str | None = None,
    ) -> Any:
        # Prefer google-adk when available; fallback to a compatibility shim.
        try:
            from google.adk import __version__  # type: ignore  # noqa: F401
            self._logger.info("google-adk package detected; using compat session wrapper for tool parity")
        except Exception:
            self._logger.info("google-adk package not available; using compatibility session")

        client_headers = {"Session-ID": session_id} if session_id else None
        llm = ChatOpenAI(
            model=model,
            http_client=httpx.Client(trust_env=False, headers=client_headers),
            base_url=base_url or self.default_base_url,
            api_key=os.getenv("GOOGLE_API_KEY", "sk-1234"),
            temperature=0,
        ).bind_tools(await self.tools.langchain_tools())
        return _OpenAICompatADKSession(llm)

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[ADKMessage]:
        converted: list[ADKMessage] = []
        for message in messages:
            role = message["role"]
            content = str(message.get("content", ""))
            if role == "assistant":
                converted.append(ADKMessage(role=role, content=content, tool_calls=message.get("tool_calls")))
            elif role == "tool":
                converted.append(
                    ADKMessage(
                        role=role,
                        content=content,
                        tool_call_id=message.get("tool_call_id"),
                        name=message.get("name"),
                    )
                )
            elif role == "system":
                converted.append(ADKMessage(role=role, content=content))
            else:
                converted.append(ADKMessage(role="user", content=content))
        return converted

    @staticmethod
    def _adk_to_langchain(history: list[ADKMessage]) -> list[BaseMessage]:
        converted: list[BaseMessage] = []
        for item in history:
            if item.role == "system":
                converted.append(SystemMessage(content=item.content))
            elif item.role == "assistant":
                converted.append(AIMessage(content=item.content, tool_calls=item.tool_calls or []))
            elif item.role == "tool":
                converted.append(ToolMessage(content=item.content, tool_call_id=item.tool_call_id or "", name=item.name))
            else:
                converted.append(HumanMessage(content=item.content))
        return converted

    def _system_message(self) -> ADKMessage:
        return ADKMessage(role="system", content=AgentSystemPrompts.SYSTEM)

    def _tool_call_system_message(self) -> ADKMessage:
        return ADKMessage(role="system", content=AgentSystemPrompts.TOOL_CALL_SYSTEM)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_steps: int = 12,
        model: str | None = None,
        session_id: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        session = (
            self._default_session
            if (model is None and session_id is None and base_url is None)
            else await self._build_adk_session(model=model or self.default_model, session_id=session_id, base_url=base_url)
        )

        history: list[ADKMessage] = [self._system_message(), *self._convert_messages(messages)]

        for _ in range(max_steps):
            ai_message: ADKMessage = session.invoke(history)
            history.append(ai_message)

            if not ai_message.tool_calls:
                lc_history = self._adk_to_langchain(history)
                response = {
                    "message": ai_message.content,
                    "steps": self._serialize_steps(lc_history),
                    "compressed_steps": self._serialize_steps(lc_history, compressed=True),
                }
                self._log_conversation(messages, response)
                return response

            for call in ai_message.tool_calls:
                result = await self.tools.dispatch_tool(call["name"], call["args"])
                result = compress_tool_result(call["name"], result)
                history.append(self._tool_call_system_message())
                history.append(
                    ADKMessage(
                        role="tool",
                        content=str(result),
                        tool_call_id=call.get("id", ""),
                        name=call.get("name"),
                    )
                )
                if call["name"] in ("current_properties",) and len(ai_message.tool_calls) <= 1:
                    lc_history = self._adk_to_langchain(history)
                    response = {
                        "message": ai_message.content,
                        "steps": self._serialize_steps(lc_history),
                        "compressed_steps": self._serialize_steps(lc_history, compressed=True),
                    }
                    self._log_conversation(messages, response)
                    return response

        error_response = {"error": "Agent hit max_steps without producing a direct response."}
        self._log_conversation(messages, error_response)
        return error_response


class AgentSystemPrompts:
    SYSTEM = (
        "角色（ROLE）\n"
        "- 你是租房方向的专业房产中介，专注于：找房 / 对比 / 租房 / 退租 / 下架。\n"
        "- 你是专业的工作人员，需要简要且直接地回答问题，不要长篇大论，直接说结论（如：没有房源，有以下房源），不要给出额外建议。\n"
        "- 当前年份：2026。\n\n"
        "核心目标（CORE GOAL）\n"
        "- 在需要时使用工具，帮助用户搜索、对比房源，并执行租房/退租/下架等操作。\n\n"
        "工具使用规则（TOOL USAGE RULE）\n"
        "- 如果不需要搜索或操作即可直接回答 -> 直接回答。\n"
        "- 如果用户要求推荐/查询房源且约束条件清晰，且历史中没有搜索过该房源 -> 必须使用工具搜索。\n"
        "- 如果历史已经搜索过该房源 -> 禁止重新搜索，仅可使用 get_house_listings 获取更多信息。\n"
        "- 严禁编造/臆测任何房源信息；只能使用工具返回的结果。\n\n"
        "意图与必做行为（INTENTS & REQUIRED BEHAVIOR）\n\n"
        "1）搜索意图（SEARCH：用户说找/推荐/看看/查询房源等）\n"
        "- 提取明确约束：\n"
        "  - 预算、区域/商圈、几居、整租/合租、通勤需求、设施/配套、入住时间、其他硬性要求。\n"
        "- 若约束足够清晰 -> 立即搜索房源。\n"
        "- 对候选房源进行核验与对比维度：\n"
        "  - 通勤、性价比、配套/设施、风险/缺点。\n"
        "2）状态变更意图（STATE-CHANGING）：租房 / 退租 / 下架\n"
        "- 若用户明确要求“租”或“退租/解除租约”或“下架/停止出租”等 -> 立即执行对应操作。\n"
        "- 不需要再次向用户确认。\n"
        "- 覆盖表达：租房/帮我租/确认租，退租/解除租约，下架/停止出租/把房源下架。\n\n"
        "3）用户认可触发（ENDORSEMENT：隐式确认）\n"
        "- 若用户明确认可某个具体房源 -> 视为已同意租下。\n"
        "- 示例：“就这个了”“这个不错”“这个更好”。\n"
        "- 行为：立即对该房源执行“租房”操作（无需确认）。\n\n"
        "平台规则（PLATFORM RULE）\n"
        "- 若用户未指定平台 -> 按顺序搜索平台，仅未搜索到结果时尝试下一平台：链家/安居客/58同城。\n\n"
        "状态同步要求（STATE SYNC REQUIREMENT）\n"
        "非常重要（VERY IMPORTANT）：\n"
        "- 只要你的回答中提到任何房源（无论是搜索结果/推荐/正在处理的房源）：\n"
        "  - **如果这次回复你没有调用其他工具，则必须调用 `current_properties`，并传入相关 house_ids。**\n\n"
        "输出质量规则（OUTPUT QUALITY RULES）\n"
        "- 表达要简洁、可操作：给出最优选项、原因、权衡点、下一步建议。\n"
        "- 最终推荐房源不超过 5 个。\n"
    )
    TOOL_CALL_SYSTEM = (
        "特别提醒（IMPORTANT）：\n"
        "- 只要你的回答中提到任何房源（无论是搜索结果/推荐/正在处理的房源）：\n"
        "  - **如果这次回复你没有调用其他工具，则必须调用 `current_properties`，并传入相关 house_ids。**\n\n"
    )
