from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.runtime_base import BaseAgentRuntime
from app.runtime_helpers import analyze_token_usage, compress_tool_result, trim_ai_message_for_history
from app.skills import SkillStore
from app.tools import AgentTools
import asyncio



class PropertyAnswer(BaseModel):
    message: str = Field(description="User-facing response summary")
    houses: list[str] = Field(default_factory=list, description="Relevant house ids")

class AgentRuntime(BaseAgentRuntime):
    def __init__(
        self,
        model: str = "qwen3-32b",
        # model: str = "qwen3",
        skills_dir: str = "skills",
        conversation_log_path: str | Path = "logs/agent_conversations.jsonl",
    ) -> None:
        super().__init__(conversation_log_path=conversation_log_path)
        self.skill_store = SkillStore(skills_dir)
        self.tools = AgentTools(self.skill_store)
        self._logger = logging.getLogger(__name__)
        self.default_model = model
        self.default_base_url = "http://api.openai.rnd.huawei.com/v1/"
        self.llm = asyncio.run(self._build_llm(model=model))
        self.structured_llm = asyncio.run(self._build_structured_llm(model=model))

    async def _build_llm(
        self,
        model: str,
        session_id: str | None = None,
        base_url: str | None = None,
        allowed_tools: set[str] | None = None,
    ):
        client_headers = {"Session-ID": session_id} if session_id else None
        return ChatOpenAI(
            model=model,
            http_client=httpx.Client(trust_env=False, headers=client_headers),
            base_url=base_url or self.default_base_url,
            api_key="sk-1234",
            # base_url="http://151.210.17.190:11345/v1",
            # api_key="",
            temperature=0,
        ).bind_tools(await self.tools.langchain_tools(allowed_tools=allowed_tools))


    async def _build_structured_llm(self, model: str, session_id: str | None = None, base_url: str | None = None):
        client_headers = {"Session-ID": session_id} if session_id else None
        return ChatOpenAI(
            model=model,
            http_client=httpx.Client(trust_env=False, headers=client_headers),
            base_url=base_url or self.default_base_url,
            api_key="sk-1234",
            temperature=0,
        ).with_structured_output(PropertyAnswer)

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
        selected_skills = await self._select_skills_for_request(messages)
        self._logger.info("skill_select result | selected_skills=%s", selected_skills)

        target_model = model or self.default_model
        allowed_tools = self.skill_store.tool_whitelist_for(selected_skills)

        if model is None and session_id is None and base_url is None and not selected_skills:
            llm = self.llm
            structured_llm = getattr(self, "structured_llm", None)
        else:
            llm = await self._build_llm(
                model=target_model,
                session_id=session_id,
                base_url=base_url,
                allowed_tools=allowed_tools,
            )
            structured_llm = await self._build_structured_llm(model=target_model, session_id=session_id, base_url=base_url)

        history: list[BaseMessage] = [self._system_message(selected_skills=selected_skills)]
        # self._logger.info("System Prompt: %s", self._system_message())
        history.extend(self._convert_messages(messages))
        token_usage_history: list[BaseMessage] = []

        for step in range(max_steps):
            self._logger.info("Invoking LLM at step %s", step + 1)
            self._logger.info(f"History | {history}")
            raw_ai_message: AIMessage = llm.invoke(history)
            token_usage_history.append(raw_ai_message)
            ai_message = trim_ai_message_for_history(raw_ai_message)
            history.append(ai_message)
            self._logger.info(
                "LLM response at step %s | content=%s | tool_calls=%s",
                step + 1,
                ai_message.content,
                json.dumps(ai_message.tool_calls, ensure_ascii=False),
            )

            if not ai_message.tool_calls:
                formatted_content = await self._format_final_content(ai_message.content, structured_llm)
                token_usage = analyze_token_usage(token_usage_history)
                usage_insights = token_usage.pop("analysis", {})
                self._logger.info(
                    "Token usage insights | llm_calls=%s | tool_call_steps=%s | final_response_steps=%s | avg_tokens_per_call=%s",
                    usage_insights.get("llm_calls", 0),
                    usage_insights.get("tool_call_steps", 0),
                    usage_insights.get("final_response_steps", 0),
                    usage_insights.get("avg_tokens_per_call", 0),
                )
                response = {
                    "message": formatted_content,
                    "steps": self._serialize_steps(history),
                    "compressed_steps": self._serialize_steps(history, compressed=True),
                    "token_usage": token_usage,
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
                result = compress_tool_result(tool_name, result)

                self._logger.info(
                    "Tool result | step=%s | tool=%s | result=%s",
                    step + 1,
                    call["name"],
                    result,
                )
                history.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))

        self._logger.error("Agent hit max_steps=%s without direct response", max_steps)
        error_response = {"error": "Agent hit max_steps without producing a direct response."}
        self._log_conversation(messages, error_response)
        return error_response

    def _system_message(self, selected_skills: list[str] | None = None) -> SystemMessage:
        headers = self.skill_store.headers()
        selected = selected_skills or []
        return SystemMessage(
            content=(
                f"可用技能头（SKILL_HEADERS）: {json.dumps(headers, ensure_ascii=False)}\n"
                f"本轮已选技能（SKILL_SELECT）: {json.dumps(selected, ensure_ascii=False)}\n\n"
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
                "输出格式要求（OUTPUT FORMAT REQUIREMENT）\n"
                "非常重要（VERY IMPORTANT）：\n"
                "- 最终输出必须是 JSON：{\"message\": string, \"houses\": string[]}。\n"
                "- 若没有房源，houses 必须为空数组，message 写结论。\n"
                "- 严禁输出 JSON 以外的多余文字。\n\n"
                "输出质量规则（OUTPUT QUALITY RULES）\n"
                "- 表达要简洁、可操作：给出最优选项、原因、权衡点、下一步建议。\n"
                "- 最终推荐房源不超过 5 个。\n"
            )
        )

    async def _select_skills_for_request(self, messages: list[dict[str, Any]]) -> list[str]:
        headers = self.skill_store.headers()
        if not headers:
            return []

        user_text = self._latest_user_text(messages)
        if not user_text:
            return []

        selector_llm = ChatOpenAI(
            model=self.default_model,
            http_client=httpx.Client(trust_env=False),
            base_url=self.default_base_url,
            api_key="sk-1234",
            temperature=0,
        )

        prompt = (
            "你是skill选择器。请根据用户请求只返回JSON数组，内容是最相关skill_id；无匹配返回[]。"
            "优先少选，最多3个。\n"
            f"skills={json.dumps(headers, ensure_ascii=False)}\n"
            f"request={user_text}"
        )

        try:
            response = selector_llm.invoke([HumanMessage(content=prompt)])
            return self._parse_selected_skill_ids(str(response.content), headers)
        except Exception:
            self._logger.exception("skill_select failed")
            return []

    @staticmethod
    def _latest_user_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    @staticmethod
    def _parse_selected_skill_ids(content: str, headers: list[dict[str, str]]) -> list[str]:
        valid_ids = {item.get("skill_id") for item in headers}
        text = content.strip()

        try:
            parsed = json.loads(text)
        except Exception:
            match = re.search(r"\[[\s\S]*\]", text)
            if not match:
                return []
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                return []

        if not isinstance(parsed, list):
            return []

        selected: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                continue
            if item in valid_ids and item not in selected:
                selected.append(item)
        return selected

    async def _format_final_content(self, content: Any, structured_llm: Any | None = None) -> str:
        text = str(content).strip()
        cleaned_text = self._strip_markdown_code_fence(text)

        parsed = self._parse_property_answer(cleaned_text)
        if parsed is not None:
            message, houses = parsed
            if not houses:
                return message
            return json.dumps({"message": message, "houses": houses}, ensure_ascii=False)

        if structured_llm is not None:
            try:
                normalized: PropertyAnswer = structured_llm.invoke(
                    "请将以下租房助手回复规范化为结构化输出。"
                    "必须返回 message 和 houses 字段，houses 仅保留房源ID字符串列表。"
                    "仅输出纯 JSON，不要使用 markdown 代码块。"
                    f"原始回复：{text}"
                )
                if normalized.houses:
                    return json.dumps(
                        {"message": normalized.message, "houses": normalized.houses},
                        ensure_ascii=False,
                    )
                return normalized.message
            except Exception:
                self._logger.exception("Structured output normalization failed")

        return cleaned_text

    @staticmethod
    def _strip_markdown_code_fence(text: str) -> str:
        candidate = text.strip()
        if not candidate.startswith("```"):
            return candidate

        fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", candidate, flags=re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        return candidate

    @staticmethod
    def _parse_property_answer(text: str) -> tuple[str, list[str]] | None:
        try:
            payload = json.loads(text)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        message = payload.get("message")
        houses = payload.get("houses")
        if not isinstance(message, str) or not isinstance(houses, list):
            return None

        house_ids = [item for item in houses if isinstance(item, str)]
        return message, house_ids

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
