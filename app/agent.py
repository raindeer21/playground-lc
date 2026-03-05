from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any, Optional, TypeVar

import langfuse
from langfuse._client.propagation import propagate_attributes
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.runtime_base import BaseAgentRuntime
from app.runtime_helpers import analyze_token_usage, compress_tool_result, extract_token_usage, trim_ai_message_for_history
from app.skills import SkillStore
from app.tools import AgentTools
import asyncio

existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
extra = "127.0.0.1,localhost"

combined = ",".join([x for x in [existing, extra] if x]).replace(",,", ",")
os.environ["NO_PROXY"] = combined
os.environ["no_proxy"] = combined
langfuse_handler = CallbackHandler()
langfuse_client = langfuse.get_client()


class PropertyAnswer(BaseModel):
    message: str = Field(description="User-facing response summary")
    houses: list[str] = Field(default_factory=list, description="Relevant house ids")


class SelectedSkills(BaseModel):
    selected_skills: list[str] = Field(default_factory=list, description="Selected skill IDs")


class Landmarks(BaseModel):
    landmark_list: list[str] = Field(default_factory=list, description="地标名称列表")


class LandmarkLookupRequest(BaseModel):
    names: list[str] = Field(default_factory=list, description="Landmark names to resolve")


StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)
LLMBuilder = Callable[[str], ChatOpenAI]
TinyAgent = Callable[[str, type[StructuredOutputT]], Awaitable[StructuredOutputT]]


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


    async def _build_structured_llm(self, model: str, session_id: str | None = None,
                                    base_url: str | None = None, allowed_tools: set[str] | None = None,):
        client_headers = {"Session-ID": session_id} if session_id else None
        return ChatOpenAI(
            model=model,
            http_client=httpx.Client(trust_env=False, headers=client_headers),
            base_url=base_url or self.default_base_url,
            api_key="sk-1234",
            temperature=0,
        ).with_structured_output(PropertyAnswer, include_raw=True)


    def _build_llm_builder(self, session_id: str | None = None, base_url: str | None = None) -> LLMBuilder:
        client_headers = {"Session-ID": session_id} if session_id else None

        def builder(model: str) -> ChatOpenAI:
            return ChatOpenAI(
                model=model,
                http_client=httpx.Client(trust_env=False, headers=client_headers),
                base_url=base_url or self.default_base_url,
                api_key="sk-1234",
                temperature=0,
            )

        return builder

    def _make_tiny_agent(
        self,
        llm_builder: LLMBuilder,
        model: str,
        callbacks: list[Any] | None = None,
    ) -> TinyAgent:
        async def tiny_agent(prompt: str, output_class: type[StructuredOutputT]) -> StructuredOutputT:
            tiny_llm = llm_builder(model).with_structured_output(output_class)
            # tiny_llm = llm_builder(model)
            config: dict[str, Any] | None = None
            if callbacks:
                config = {"callbacks": callbacks}
            return tiny_llm.invoke([HumanMessage(content=prompt)], config=config)

        return tiny_agent

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_steps: int = 12,
        model: str | None = None,
        session_id: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        with langfuse_client.start_as_current_observation(as_type="span", name="langchain-call") as span:
            # Propagate session_id to all observations
            span.update(input=messages[-1]["content"])
            langfuse_client.update_current_trace(input=messages[-1]["content"])
            with propagate_attributes(session_id=session_id):
                response = await self._chat(messages, max_steps, model=model, session_id=session_id, base_url=base_url)
                span.update(output=response["message"])
                langfuse_client.update_current_trace(output=response["message"])
                return response

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        max_steps: int = 12,
        model: str | None = None,
        session_id: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        self._logger.info("Agent chat started | max_steps=%s | message_count=%s", max_steps, len(messages))
        self._logger.info("Incoming messages payload: %s", json.dumps(messages, ensure_ascii=False))
        target_model = model or self.default_model
        llm, structured_llm, history = await self._prepare_chat_dependencies(
            messages=messages,
            target_model=target_model,
            model=model,
            session_id=session_id,
            base_url=base_url,
        )
        token_usage_history: list[BaseMessage] = []

        for step in range(max_steps):
            ai_message = self._run_llm_step(llm=llm, history=history, step=step, token_usage_history=token_usage_history)

            if not ai_message.tool_calls:
                response = await self._finalize_direct_response(
                    messages=messages,
                    history=history,
                    structured_llm=structured_llm,
                    token_usage_history=token_usage_history,
                    ai_content=ai_message.content,
                    step=step,
                )
                return response

            tool_response = await self._process_tool_calls(
                messages=messages,
                history=history,
                token_usage_history=token_usage_history,
                tool_calls=ai_message.tool_calls,
                step=step,
            )
            if tool_response is not None:
                return tool_response

        self._logger.error("Agent hit max_steps=%s without direct response", max_steps)
        error_response = {"error": "Agent hit max_steps without producing a direct response."}
        self._log_conversation(messages, error_response)
        return error_response

    async def _prepare_chat_dependencies(
        self,
        messages: list[dict[str, Any]],
        target_model: str,
        model: str | None,
        session_id: str | None,
        base_url: str | None,
    ) -> tuple[Any, Any | None, list[BaseMessage]]:
        llm_builder = self._build_llm_builder(session_id=session_id, base_url=base_url)
        tiny_agent = self._make_tiny_agent(
            llm_builder=llm_builder,
            model=target_model,
            callbacks=[langfuse_handler],
        )

        # selected_skills = await self._select_skills_for_request(messages, tiny_agent=tiny_agent)
        # self._logger.info("skill_select result | selected_skills=%s", selected_skills)
        # landmark_memory_entries = []
        #
        # if selected_skills:
        #     landmark_memory_entries, effective_skills = await self._resolve_landmark_memories(
        #         messages=messages,
        #         selected_skills=selected_skills,
        #         tiny_agent=tiny_agent,
        #     )

        effective_skills = None

        allowed_tools = self.skill_store.tool_whitelist_for(effective_skills)
        llm = await self._build_llm(
            model=target_model,
            session_id=session_id,
            base_url=base_url,
            allowed_tools=allowed_tools,
        )
        structured_llm = await self._build_structured_llm(
            model=target_model,
            session_id=session_id,
            base_url=base_url,
            allowed_tools=allowed_tools,
        )

        history: list[BaseMessage] = [self._system_message(selected_skills=effective_skills)]
        # history.extend(landmark_memory_entries)
        history.extend(self._convert_messages(messages))
        return llm, structured_llm, history

    def _run_llm_step(
        self,
        llm: Any,
        history: list[BaseMessage],
        step: int,
        token_usage_history: list[BaseMessage],
    ) -> AIMessage:
        self._logger.info("Invoking LLM at step %s", step + 1)
        self._logger.info(f"History | {history}")
        raw_ai_message = llm.invoke(history, config={"callbacks": [langfuse_handler]})
        token_usage_history.append(raw_ai_message)
        ai_message = trim_ai_message_for_history(raw_ai_message)
        history.append(ai_message)
        self._logger.info(
            "LLM response at step %s | content=%s | tool_calls=%s",
            step + 1,
            ai_message.content,
            json.dumps(ai_message.tool_calls, ensure_ascii=False),
        )
        return ai_message

    async def _finalize_direct_response(
        self,
        messages: list[dict[str, Any]],
        history: list[BaseMessage],
        structured_llm: Any | None,
        token_usage_history: list[BaseMessage],
        ai_content: Any,
        step: int,
    ) -> dict[str, Any]:
        formatted_content = await self._format_final_content(ai_content, structured_llm)
        response = self._build_chat_response(history=history, message=formatted_content, token_usage_history=token_usage_history)
        self._logger.info("Agent completed with direct LLM response at step %s", step + 1)
        self._log_conversation(messages, response)
        return response

    async def _process_tool_calls(
        self,
        messages: list[dict[str, Any]],
        history: list[BaseMessage],
        token_usage_history: list[BaseMessage],
        tool_calls: list[dict[str, Any]],
        step: int,
    ) -> dict[str, Any] | None:
        for call in tool_calls:
            tool_name = call["name"]
            call_args = call.get("args", {})
            self._logger.info(
                "Dispatching tool call | step=%s | tool=%s | args=%s",
                step + 1,
                tool_name,
                json.dumps(call_args, ensure_ascii=False),
            )
            result = await self.tools.dispatch_tool(tool_name, call_args)
            result = compress_tool_result(tool_name, result)

            self._logger.info(
                "Tool result | step=%s | tool=%s | result=%s",
                step + 1,
                tool_name,
                result,
            )
            history.append(ToolMessage(content=result, tool_call_id=call["id"], name=tool_name))

            if self._is_final_answer_tool_call(call_args):
                final_content = self._build_property_answer_from_tool_result(result)
                history.append(AIMessage(content=final_content))
                response = self._build_chat_response(
                    history=history,
                    message=final_content,
                    token_usage_history=token_usage_history,
                )
                self._logger.info(
                    "Agent completed with final_answer short-circuit | step=%s | tool=%s",
                    step + 1,
                    tool_name,
                )
                self._log_conversation(messages, response)
                return response
        return None

    def _build_chat_response(
        self,
        history: list[BaseMessage],
        message: str,
        token_usage_history: list[BaseMessage],
    ) -> dict[str, Any]:
        token_usage = analyze_token_usage(token_usage_history)
        usage_insights = token_usage.pop("analysis", {})
        self._logger.info(
            "Token usage insights | llm_calls=%s | tool_call_steps=%s | final_response_steps=%s | avg_tokens_per_call=%s",
            usage_insights.get("llm_calls", 0),
            usage_insights.get("tool_call_steps", 0),
            usage_insights.get("final_response_steps", 0),
            usage_insights.get("avg_tokens_per_call", 0),
        )
        return {
            "message": message,
            "steps": self._serialize_steps(history),
            "compressed_steps": self._serialize_steps(history, compressed=True),
            "token_usage": token_usage,
        }

    @staticmethod
    def _is_final_answer_tool_call(args: dict[str, Any]) -> bool:
        return bool(args.get("final_answer") is True)

    @staticmethod
    def _extract_house_ids_from_tool_payload(payload: Any) -> list[str]:
        if not isinstance(payload, dict):
            return []

        houses = payload.get("houses")
        if not isinstance(houses, list):
            return []

        house_ids: list[str] = []
        for item in houses:
            if isinstance(item, str):
                house_ids.append(item)
                continue
            if not isinstance(item, dict):
                continue
            for key in ("houseid", "house_id", "id"):
                value = item.get(key)
                if isinstance(value, str):
                    house_ids.append(value)
                    break
        return house_ids

    @classmethod
    def _build_property_answer_from_tool_result(cls, tool_result: str) -> str:
        house_ids: list[str] = []
        try:
            parsed = json.loads(tool_result)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            house_ids = cls._extract_house_ids_from_tool_payload(parsed)

        payload = PropertyAnswer(
            message="为您找到以下符合条件的房源：" if house_ids else "暂无符合条件的房源",
            houses=house_ids,
        )
        return payload.model_dump_json(ensure_ascii=False)

    def _system_message(
        self,
        selected_skills: list[str] | None = None,
    ) -> SystemMessage:
        return SystemMessage(
            content=(
                "角色（ROLE）\n"
                "- 你是租房方向的专业房产中介，专注于：找房 / 对比 / 租房 / 退租 / 下架。\n"
                "- 你是专业的工作人员，需要简要且直接地回答问题，不要长篇大论，直接说结论（如：没有房源，有以下房源），不要给出额外建议。\n"
                "事实规则（GROUND TRUTH）\n"
                "- 当前年份：2026。\n"
                "- 房源ID全局唯一，相同的房源ID一定对应同一套房子，即使列在不同的平台上。\n"
                "- 当用户以个数（“两套”、“三个” 等）或模糊指向请求时（“这套”，“那个” 等），如果事实存在的房源数目与请求不符合，直接按真实数目处理，不需要向用户确认\n"
                "- 当用户提出更多要求筛选时，必须检查目前每个房子的信息。\n\n"
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
                "  - search_house 调用仅包含**必须的要求**，“如果可以” “有的话更好” “最好有” 等可选条件不能包括。\n"
                "- 若用户提供任意条件时 -> 立即搜索房源。\n"
                "- 如果可以 有的话更好 最好有 等可选条件 -> 同时搜索符合与不符合该条件的房源，优先选取符合的房源。\n"
                # "- 对候选房源进行核验与对比维度：\n"
                # "  - 通勤、性价比、配套/设施、风险/缺点。\n"
                "2）状态变更意图（STATE-CHANGING）：租房 / 退租 / 下架\n"
                "- 若用户明确要求“租”或“退租/解除租约”或“下架/停止出租”等 -> 立即执行对应操作。\n"
                "- '可以租吗' '我想租' -> 仅检查房源是否可用，除非明确要求执行操作。\n\n"
                # "3）用户认可触发（ENDORSEMENT：隐式确认）\n"
                # "- 若用户明确认可某个具体房源 -> 视为已同意租下。\n"
                # "- 示例：“就这个了”“这个不错”“这个更好”。\n"
                # "- 行为：立即对该房源执行“租房”操作（无需确认）。\n\n"
                # "平台规则（PLATFORM RULE）\n"
                # "- 若用户未指定平台 -> 按顺序搜索平台，仅未搜索到结果时尝试下一平台：链家/安居客/58同城。\n\n"
                "输出格式要求（OUTPUT FORMAT REQUIREMENT）\n"
                "非常重要（VERY IMPORTANT）：\n"
                "- 你只能有两种输出：消息输出和工具调用输出"
                "- 消息输出：必须是以下 JSON 格式：{\"message\": string, \"houses\": string[]}。"
                " houses 填写房源 ID（如：HF_36），message 填写处理结论和结果消息，必须同时包含这两个字段。"
                " 若没有房源，houses 必须为空数组，message 写结论。"
                " 严禁输出 message 与 houses 以外的字段。严禁将工具调用输出到消息输出中，严禁输出 JSON 以外的多余文字，仅输出纯 JSON，不要使用 markdown 代码块。\n"
                "- 工具调用输出：必须使用 Tool call 形式，禁止生成到消息输出的json中。"
                "输出质量规则（OUTPUT QUALITY RULES）\n"
                "- 表达要简洁、可操作：给出最优选项、原因、权衡点、下一步建议。\n"
                "- 最终推荐房源不超过 5 个。\n"
            )
        )


    async def _resolve_landmark_memories(
        self,
        messages: list[dict[str, Any]],
        selected_skills: list[str] | None,
        tiny_agent: TinyAgent,
    ) -> tuple[list[BaseMessage], list[str]]:
        selected = list(selected_skills or [])
        if "landmark_search" not in selected:
            return [], selected

        filtered_skills = selected
        user_text = self._latest_user_text(messages)
        if not user_text:
            return [], filtered_skills

        prompt = (
            "你是一个地标名称提取器。根据用户请求提取所有需要查询的地标名称，包含地名、商圈名等（不包含小区名、地铁站、地铁线名、行政区名）。"
            "如果关键词如果出现行政区、市名，请给出分词版本和原版本。不要直接给出行政区、市名。"
            "仅返回 JSON，若无法确定，返回空数组。严禁输出 JSON 以外的多余文字，仅输出纯 JSON，不要使用 markdown 代码块。"
            f"\nrequest={user_text}"
        )

        try:
            extracted = await tiny_agent(prompt, LandmarkLookupRequest)
            candidate_names = [name.strip() for name in extracted.names if isinstance(name, str) and name.strip()]
            unique_names: list[str] = []
            for candidate in candidate_names:
                if candidate not in unique_names:
                    unique_names.append(candidate)
            if not unique_names:
                return [], filtered_skills

            memory_entries: list[BaseMessage] = []
            for landmark_name in unique_names:
                raw_result = await self.tools.dispatch_tool("search_landmarks", {"name": landmark_name})
                parsed = json.loads(raw_result)
                if not isinstance(parsed, dict):
                    continue

                landmark_id = parsed.get("id")
                resolved_name = parsed.get("name", landmark_name)
                if not isinstance(landmark_id, str) or not landmark_id:
                    continue
                if not isinstance(resolved_name, str) or not resolved_name:
                    resolved_name = landmark_name

                tool_call_id = f"landmark_memory_{landmark_id}"
                tool_call = {
                    "id": tool_call_id,
                    "name": "search_landmarks",
                    "args": {"name": resolved_name},
                }
                memory_entries.append(AIMessage(content="\n\n", tool_calls=[tool_call]))
                memory_entries.append(
                    ToolMessage(
                        content=json.dumps({"name": resolved_name, "id": landmark_id}, ensure_ascii=False),
                        tool_call_id=tool_call_id,
                        name="search_landmarks",
                    )
                )

            self._logger.info("landmark memories resolved | entries=%s", len(memory_entries))
            if memory_entries:
                filtered_skills = [skill for skill in selected if skill != "landmark_search"]
            return memory_entries, filtered_skills
        except Exception:
            self._logger.exception("landmark memory resolve failed")
            return [], filtered_skills

    async def _select_skills_for_request(
        self,
        messages: list[dict[str, Any]],
        tiny_agent: TinyAgent | None = None,
    ) -> list[Any] | list[str] | None:
        headers = self.skill_store.headers()
        if not headers:
            self._logger.info("skill_select skipped | reason=no_headers")
            return []

        request_text = self._messages_for_skill_selection(messages)
        if not request_text:
            self._logger.info("skill_select skipped | reason=no_human_or_ai_text")
            return []

        prompt = (
            """你是skill选择器,请根据提供的对话，选择语义相关的skill。
            示例输出：{"selected_skills": ["skill1", "skill2"]}。
            严禁输出 JSON 以外的多余文字，不要使用 markdown 代码块。
"""
            f"skills={json.dumps(headers, ensure_ascii=False)}\n"
            f"request={request_text}"
        )

        self._logger.info(
            "skill_select input | %s",
            json.dumps(
                {
                    "request": request_text,
                    "skill_count": len(headers),
                    "skill_ids": [item.get("skill_id") for item in headers],
                },
                ensure_ascii=False,
            ),
        )

        try:
            response = await tiny_agent(prompt, SelectedSkills)
            selected_skills = list({s for s in response.selected_skills})
            # self._logger.info(
            #     "skill_select token_usage | %s",
            #     json.dumps(extract_token_usage(response), ensure_ascii=False),
            # )
            self._logger.info(
                "skill_select output | %s",
                json.dumps(
                    {
                        "selected_skills": selected_skills,
                    },
                    ensure_ascii=False,
                ),
            )
            return selected_skills
        except Exception:
            self._logger.exception("skill_select failed")
            return None

    async def _run_tiny_agent(
        self,
        prompt: str,
        output_class: type[StructuredOutputT],
    ) -> StructuredOutputT:
        tiny_agent = self._make_tiny_agent(
            llm_builder=self._build_llm_builder(),
            model=self.default_model,
        )
        return await tiny_agent(prompt, output_class)

    @staticmethod
    def _latest_user_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    @staticmethod
    def _messages_for_skill_selection(messages: list[dict[str, Any]]) -> str:
        selected_messages: list[str] = []
        for message in messages:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue

            content = message.get("content", "")
            if role == "assistant":
                content = json.loads(content).get("message")
            if isinstance(content, str):
                normalized_content = content.strip()
            else:
                normalized_content = str(content).strip()

            if normalized_content:
                selected_messages.append(f"{role}: {normalized_content}")

        return "\n".join(selected_messages)

    @staticmethod
    def _parse_selected_skill_ids(content: str, headers: list[dict[str, str]]) -> list[str]:
        valid_ids = {item.get("skill_id") for item in headers}
        text = content.strip()

        try:
            parsed = json.loads(text)
        except Exception:
            matches = list(re.finditer(r"\[[\s\S]*?\]", text))
            if not matches:
                return []
            parsed: Optional[SelectedSkills] = None
            for match in reversed(matches):
                try:
                    parsed = json.loads(match.group(0))
                    break
                except Exception:
                    continue
            if parsed is None:
                return []

        if isinstance(parsed, dict):
            selected = AgentRuntime._coerce_selected_skill_ids(parsed.get("selected_skills"), valid_ids)
            if selected:
                return selected
            raw_response = parsed.get("raw_response")
            if isinstance(raw_response, str):
                return AgentRuntime._parse_selected_skill_ids(raw_response, headers)
            return []

        if not isinstance(parsed, list):
            return []

        return AgentRuntime._coerce_selected_skill_ids(parsed, valid_ids)

    @staticmethod
    def _coerce_selected_skill_ids(payload: Any, valid_ids: set[str | None]) -> list[str]:
        if not isinstance(payload, list):
            return []

        selected: list[str] = []
        for item in payload:
            skill_id: str | None = None
            if isinstance(item, str):
                skill_id = item
            elif isinstance(item, dict):
                candidate = item.get("skill_id")
                if isinstance(candidate, str):
                    skill_id = candidate
            if skill_id in valid_ids and skill_id not in selected:
                selected.append(skill_id)
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

        # if structured_llm is not None:
        #     try:
        #         normalized: PropertyAnswer = structured_llm.invoke(
        #             "请将以下租房助手回复规范化为结构化输出。"
        #             "必须返回 message 和 houses 字段，houses 仅保留房源ID字符串列表。"
        #             "仅输出纯 JSON，不要使用 markdown 代码块。"
        #             f"原始回复：{text}"
        #         )
        #         if normalized.houses:
        #             return json.dumps(
        #                 {"message": normalized.message, "houses": normalized.houses},
        #                 ensure_ascii=False,
        #             )
        #         return normalized.message
        #     except Exception:
        #         self._logger.exception("Structured output normalization failed")

        self._logger.exception(f"STRUCTURED OUTPUT VIOLATION | {cleaned_text}")
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
