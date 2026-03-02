from __future__ import annotations

import ast
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.skills import SkillStore
from app.tools import AgentTools
import asyncio

class AgentRuntime:
    def __init__(
        self,
        model: str = "qwen3-32b",
        # model: str = "qwen3",
        skills_dir: str = "skills",
        conversation_log_path: str | Path = "logs/agent_conversations.jsonl",
    ) -> None:
        self.skill_store = SkillStore(skills_dir)
        self.tools = AgentTools(self.skill_store)
        self.conversation_log_path = Path(conversation_log_path)
        self._logger = logging.getLogger(__name__)
        self.default_model = model
        self.default_base_url = "http://api.openai.rnd.huawei.com/v1/"
        self.llm = asyncio.run(self._build_llm(model=model))

    async def _build_llm(self, model: str, session_id: str | None = None, base_url: str | None = None):
        client_headers = {"Session-ID": session_id} if session_id else None
        return ChatOpenAI(
            model=model,
            http_client=httpx.Client(trust_env=False, headers=client_headers),
            base_url=base_url or self.default_base_url,
            api_key="sk-1234",
            # base_url="http://151.210.17.190:11345/v1",
            # api_key="",
            temperature=0,
        ).bind_tools(await self.tools.langchain_tools())

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_steps: int = 12,
        model: str | None = None,
        session_id: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        self._logger.info("Agent chat started | max_steps=%s | message_count=%s", max_steps, len(messages))
        self._logger.info("Incoming messages payload: %s", json.dumps(messages, ensure_ascii=False))
        llm = self.llm if (model is None and session_id is None and base_url is None) else await self._build_llm(
            model=model or self.default_model,
            session_id=session_id,
            base_url=base_url,
        )

        history: list[BaseMessage] = [self._system_message()]
        # self._logger.info("System Prompt: %s", self._system_message())
        history.extend(self._convert_messages(messages))

        for step in range(max_steps):
            self._logger.info("Invoking LLM at step %s", step + 1)
            self._logger.info(f"History | {history}")
            ai_message: AIMessage = llm.invoke(history)
            history.append(ai_message)
            self._logger.info(
                "LLM response at step %s | content=%s | tool_calls=%s",
                step + 1,
                ai_message.content,
                json.dumps(ai_message.tool_calls, ensure_ascii=False),
            )

            if not ai_message.tool_calls:
                response = {
                    "message": str(ai_message.content),
                    "steps": _serialize_steps(history),
                    "compressed_steps": _serialize_steps(history)
                }
                self._logger.info("Agent completed with direct LLM response at step %s", step + 1)
                self._log_conversation(messages, response)
                return response

            for call in ai_message.tool_calls:
                self._logger.info(
                    "Dispatching tool call | step=%s | tool=%s | args=%s",
                    step + 1,
                    call["name"],
                    json.dumps(call["args"], ensure_ascii=False),
                )
                result = await self.tools.dispatch_tool(call["name"], call["args"])

                tool_name = call["name"]
                if tool_name == "get_houses_nearby":
                    result = _compress_get_houses_nearby_result(result)
                elif tool_name == "get_houses_by_platform":
                    result = _compress_get_houses_by_platform_result(result)
                elif tool_name == "get_houses_by_community":
                    result = _compress_get_houses_by_community_result(result)

                self._logger.info(
                    "Tool result | step=%s | tool=%s | result=%s",
                    step + 1,
                    call["name"],
                    result,
                )
                history.append(self._tool_call_system_message())
                history.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))
                if call["name"] in ("current_properties", ) and len(ai_message.tool_calls) <= 1:
                    response = {
                        "message": str(ai_message.content),
                        "steps": _serialize_steps(history),
                        "compressed_steps": _serialize_steps(history)
                    }
                    self._logger.info("Agent completed with direct LLM response at step %s", step + 1)
                    self._log_conversation(messages, response)
                    return response

        self._logger.error("Agent hit max_steps=%s without direct response", max_steps)
        error_response = {"error": "Agent hit max_steps without producing a direct response."}
        self._log_conversation(messages, error_response)
        return error_response

    def _log_conversation(self, messages: list[dict[str, Any]], response: dict[str, Any]) -> None:
        logger = getattr(self, "_logger", logging.getLogger(__name__))
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "messages": messages,
            "response": response,
        }
        try:
            self.conversation_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.conversation_log_path.open("a", encoding="utf-8") as fp:
                fp.write(f"{json.dumps(payload, ensure_ascii=False)}\n")
            logger.info("Conversation logged to %s", self.conversation_log_path)
        except OSError:
            logger.exception("Failed to write conversation log")

    def _system_message(self) -> SystemMessage:
        headers = self.skill_store.headers()
        return SystemMessage(
            content=(
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
        )

    def _tool_call_system_message(self) -> SystemMessage:
        return SystemMessage(
            content=(
                "特别提醒（IMPORTANT）：\n"
                "- 只要你的回答中提到任何房源（无论是搜索结果/推荐/正在处理的房源）：\n"
                "  - **如果这次回复你没有调用其他工具，则必须调用 `current_properties`，并传入相关 house_ids。**\n\n"
            )
        )

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[BaseMessage]:
        converted: list[BaseMessage] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    converted.append(AIMessage(content=content, tool_calls=tool_calls))
                else:
                    converted.append(AIMessage(content=content))
            elif role == "tool":
                converted.append(ToolMessage(content=content, tool_call_id=message.get("tool_call_id", "")))
            else:
                converted.append(HumanMessage(content=content))
        return converted


def _serialize_steps_compressed(history: list[BaseMessage]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    tool_call_name_by_id: dict[str, str] = {}

    for msg in history:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                call_id = call.get("id")
                call_name = call.get("name")
                if isinstance(call_id, str) and isinstance(call_name, str):
                    tool_call_name_by_id[call_id] = call_name
            steps.append({"type": "tool_calls", "tool_calls": msg.tool_calls})
        elif isinstance(msg, ToolMessage):
            # if msg.name in ("current_properties", ):
            #     continue
            content: Any = msg.content
            tool_name = msg.name
            if tool_name == "get_houses_nearby":
                content = _compress_get_houses_nearby_result(msg.content)
            elif tool_name == "get_houses_by_platform":
                content = _compress_get_houses_by_platform_result(msg.content)
            elif tool_name == "get_houses_by_community":
                content = _compress_get_houses_by_community_result(msg.content)
            steps.append({"type": "tool_result",
                          "content": content,
                          "tool_call_id": msg.tool_call_id,
                          "status": msg.status})

    return steps

def _serialize_steps(history: list[BaseMessage]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    tool_call_name_by_id: dict[str, str] = {}
    for msg in history:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                call_id = call.get("id")
                call_name = call.get("name")
                if isinstance(call_id, str) and isinstance(call_name, str):
                    tool_call_name_by_id[call_id] = call_name
            steps.append({"type": "tool_calls", "tool_calls": msg.tool_calls})
        elif isinstance(msg, ToolMessage):
            content: Any = msg.content
            steps.append({"type": "tool_result",
                          "content": content,
                          "tool_call_id": msg.tool_call_id,
                          "status": msg.status})
    return steps


def _parse_tool_payload(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            candidate = parser(content)
            if isinstance(candidate, dict):
                return candidate
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
    return None


def _compress_get_houses_nearby_result(content: Any) -> Any:
    payload = _parse_tool_payload(content)
    if payload is None:
        return content

    data = payload.get("data")
    if not isinstance(data, dict):
        return content

    landmark = data.get("landmark")
    items = data.get("items")
    if not isinstance(items, list):
        return content

    compressed_landmark = {}
    if isinstance(landmark, dict):
        compressed_landmark = {
            "id": landmark.get("id"),
            "name": landmark.get("name"),
        }

    compressed_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compressed_items.append(
            {
                "house_id": item.get("house_id"),
                "status": item.get("status"),
                "distance_to_landmark": item.get("distance_to_landmark"),
                "walking_distance": item.get("walking_distance"),
                "walking_duration": item.get("walking_duration"),
                "listing_platform": item.get("listing_platform"),
            }
        )

    payload["data"] = {
        "landmark": compressed_landmark,
        "items": compressed_items,
    }

    return json.dumps(payload, ensure_ascii=False)


def _compress_get_houses_by_platform_result(content: Any) -> Any:
    payload = _parse_tool_payload(content)
    if payload is None:
        return content

    data = payload.get("data")
    if not isinstance(data, dict):
        return content

    items = data.get("items")
    if not isinstance(items, list):
        return content

    compressed_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compressed_items.append(
            {
                "house_id": item.get("house_id"),
                "status": item.get("status"),
            }
        )

    data["items"] = compressed_items
    payload["data"] = data
    return json.dumps(payload, ensure_ascii=False)



def _compress_get_houses_by_community_result(content: Any) -> Any:
    payload = _parse_tool_payload(content)
    if payload is None:
        return content

    data = payload.get("data")
    if not isinstance(data, dict):
        return content

    items = data.get("items")
    if not isinstance(items, list):
        return content

    compressed_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compressed_items.append(
            {
                "house_id": item.get("house_id"),
                "community": item.get("community"),
                "listing_platform": item.get("listing_platform"),
                "status": item.get("status"),
            }
        )

    data["items"] = compressed_items
    payload["data"] = data
    return json.dumps(payload, ensure_ascii=False)
