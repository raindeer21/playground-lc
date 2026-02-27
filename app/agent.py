from __future__ import annotations

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
        self.llm = self._build_llm(model=model)

    def _build_llm(self, model: str, session_id: str | None = None, base_url: str | None = None):
        client_headers = {"Session-ID": session_id} if session_id else None
        return ChatOpenAI(
            model=model,
            http_client=httpx.Client(trust_env=False, headers=client_headers),
            base_url=base_url or self.default_base_url,
            api_key="sk-1234",
            # base_url="http://151.210.17.190:11345/v1",
            # api_key="",
            temperature=0,
        ).bind_tools(self.tools.langchain_tools())

    def chat(
        self,
        messages: list[dict[str, str]],
        max_steps: int = 12,
        model: str | None = None,
        session_id: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        self._logger.info("Agent chat started | max_steps=%s | message_count=%s", max_steps, len(messages))
        self._logger.info("Incoming messages payload: %s", json.dumps(messages, ensure_ascii=False))
        llm = self.llm if (model is None and session_id is None and base_url is None) else self._build_llm(
            model=model or self.default_model,
            session_id=session_id,
            base_url=base_url,
        )

        history: list[BaseMessage] = [self._system_message()]
        self._logger.info("System Prompt: %s", self._system_message())
        history.extend(self._convert_messages(messages))

        for step in range(max_steps):
            self._logger.info("Invoking LLM at step %s", step + 1)
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
                result = self.tools.dispatch_tool(call["name"], call["args"])
                self._logger.info(
                    "Tool result | step=%s | tool=%s | result=%s",
                    step + 1,
                    call["name"],
                    result,
                )
                history.append(ToolMessage(content=result, tool_call_id=call["id"]))

        self._logger.error("Agent hit max_steps=%s without direct response", max_steps)
        error_response = {"error": "Agent hit max_steps without producing a direct response."}
        self._log_conversation(messages, error_response)
        return error_response

    def _log_conversation(self, messages: list[dict[str, str]], response: dict[str, Any]) -> None:
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
                "You are a house-renting assistant. \n"
                "- Use tools when needed; if a direct answer is sufficient, reply directly. \n"
                "- Workflow: first inspect the available skill headers below, call get_skills(skill_ids=[...]) when skills are relevant; "
                "if only one skill is needed, get_skills(skill_id=...) is allowed. Prefer loading multiple skills in one call. \n"
                "- Use web_request when you need web data.\n\n"
                "For rental scenarios, enforce this policy: "
                "(Important) YOU MUST USE SKILLS PROVIDED, NEVER MAKE ASSUMPTIONS OR MADE UP PROPERTIES, ALWAYS SEARCH FOR LISTINGS FIRST IF CONSTRAINS ARE CLEAR. "
                "- extract explicit constraints (budget, district/area, bedrooms, rental type, commute, facilities), "
                "- ask one focused follow-up question when key constraints are missing or ambiguous, proactively gather information from"
                "- verify and compare candidate listings across dimensions (commute, price-performance, amenities, facilities, risk), "
                "- return practical recommendations with clear pros/cons, and "
                "- cap final recommended candidates to at most 5 listings, and when a property-find request is completed call provide_property_result_list with the final house_ids.\n\n"
                f"Available skills:\n{json.dumps(headers, indent=2)}"
            )
        )

    @staticmethod
    def _convert_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
        converted: list[BaseMessage] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        return converted


def _serialize_steps(history: list[BaseMessage]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for msg in history:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            steps.append({"type": "tool_calls", "tool_calls": msg.tool_calls})
        elif isinstance(msg, ToolMessage):
            steps.append({"type": "tool_result", "content": msg.content})
    return steps
