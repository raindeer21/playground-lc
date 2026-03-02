from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable

from fastmcp import Client

from app.runtime_base import BaseAgentRuntime
from app.runtime_helpers import compress_tool_result
from app.skills import SkillStore
from app.tools import AgentTools, mcp


class GoogleADKAgentRuntime(BaseAgentRuntime):
    """Google ADK runtime backed by LlmAgent + Runner."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        skills_dir: str = "skills",
        conversation_log_path: str | Path = "logs/agent_conversations.jsonl",
    ) -> None:
        super().__init__(conversation_log_path=conversation_log_path)
        self._logger = logging.getLogger(__name__)
        self.skill_store = SkillStore(skills_dir)
        self.tools = AgentTools(self.skill_store)
        self.default_model = model
        self._runner: Any | None = None
        self._runner_model: str | None = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_steps: int = 12,
        model: str | None = None,
        session_id: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        if base_url:
            self._logger.warning("Google ADK backend currently ignores base_url override: %s", base_url)

        model_name = model or self.default_model
        runner = await self._get_or_build_runner(model_name)

        adk_session_id = session_id or f"session-{uuid.uuid4().hex}"
        user_id = "default-user"
        await self._ensure_session(runner, user_id=user_id, session_id=adk_session_id)

        transcript = self._build_user_transcript(messages)
        content = self._create_adk_content(transcript)

        steps: list[dict[str, Any]] = []
        assistant_chunks: list[str] = []
        tool_name_by_id: dict[str, str] = {}

        event_count = 0
        async for event in runner.run_async(
            user_id=user_id,
            session_id=adk_session_id,
            new_message=content,
        ):
            event_count += 1
            self._collect_event_steps(event, steps, assistant_chunks, tool_name_by_id)
            if event_count >= max_steps * 20:
                break

        message = "\n".join(chunk for chunk in assistant_chunks if chunk).strip()
        if not message:
            response = {"error": "Google ADK runner finished without assistant text response."}
            self._log_conversation(messages, response)
            return response

        response = {
            "message": message,
            "steps": steps,
            "compressed_steps": self._compress_steps(steps),
        }
        self._log_conversation(messages, response)
        return response

    async def _get_or_build_runner(self, model_name: str) -> Any:
        if self._runner is not None and self._runner_model == model_name:
            return self._runner

        from google.adk.agents import LlmAgent
        from google.adk.artifacts import InMemoryArtifactService
        from google.adk.memory import InMemoryMemoryService
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        tools = await self._build_adk_tools()
        agent = LlmAgent(
            name="rental_assistant",
            model=model_name,
            instruction=self._system_instruction(),
            tools=tools,
        )
        self._runner = Runner(
            app_name=agent.name,
            agent=agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
        self._runner_model = model_name
        return self._runner

    async def _ensure_session(self, runner: Any, *, user_id: str, session_id: str) -> None:
        existing = await runner.session_service.get_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if existing is None:
            await runner.session_service.create_session(
                app_name=runner.app_name,
                user_id=user_id,
                session_id=session_id,
            )

    async def _build_adk_tools(self) -> list[Callable[..., Any]]:
        tools = await mcp.list_tools()
        wrapped_tools: list[Callable[..., Any]] = []
        for tool in tools:
            if not tool.name:
                continue
            wrapped_tools.append(
                self._make_tool_callable(
                    name=tool.name,
                    description=tool.description or f"Invoke MCP tool {tool.name}.",
                )
            )
        return wrapped_tools

    def _make_tool_callable(self, *, name: str, description: str) -> Callable[..., Any]:
        async def _tool(**kwargs: Any) -> str:
            async with Client(mcp) as client:
                result = await client.call_tool(name, kwargs)
            if result.is_error:
                return json.dumps({"error": result.data}, ensure_ascii=False)
            return str(result.data)

        _tool.__name__ = name
        _tool.__doc__ = description
        return _tool

    @staticmethod
    def _build_user_transcript(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    @staticmethod
    def _create_adk_content(text: str) -> Any:
        from google.genai import types

        return types.Content(role="user", parts=[types.Part.from_text(text=text)])

    def _collect_event_steps(
        self,
        event: Any,
        steps: list[dict[str, Any]],
        assistant_chunks: list[str],
        tool_name_by_id: dict[str, str],
    ) -> None:
        content = getattr(event, "content", None)
        if content is None:
            return
        parts = getattr(content, "parts", None) or []

        pending_calls: list[dict[str, Any]] = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                assistant_chunks.append(str(text))

            func_call = getattr(part, "function_call", None)
            if func_call is not None:
                name = str(getattr(func_call, "name", ""))
                args = getattr(func_call, "args", {}) or {}
                call_id = str(getattr(func_call, "id", "") or f"call-{uuid.uuid4().hex}")
                tool_name_by_id[call_id] = name
                pending_calls.append({"id": call_id, "name": name, "args": args})

            func_resp = getattr(part, "function_response", None)
            if func_resp is not None:
                call_id = str(getattr(func_resp, "id", "") or "")
                name = str(getattr(func_resp, "name", "") or tool_name_by_id.get(call_id, "unknown"))
                response_payload = getattr(func_resp, "response", {})
                steps.append(
                    {
                        "type": "tool_result",
                        "content": json.dumps(response_payload, ensure_ascii=False),
                        "tool_call_id": call_id,
                        "status": "success",
                        "name": name,
                    }
                )

        if pending_calls:
            steps.append({"type": "tool_calls", "tool_calls": pending_calls})

    @staticmethod
    def _compress_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compressed: list[dict[str, Any]] = []
        for step in steps:
            if step.get("type") != "tool_result":
                compressed.append(step)
                continue
            name = str(step.get("name", ""))
            content = compress_tool_result(name, step.get("content"))
            compressed.append({**step, "content": content})
        return compressed

    def _system_instruction(self) -> str:
        headers = self.skill_store.headers()
        return (
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
            f"\n可用技能（SKILLS）：\n{headers}\n"
        )
