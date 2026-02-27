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
        skills_dir: str = "skills",
        conversation_log_path: str | Path = "logs/agent_conversations.jsonl",
    ) -> None:
        self.skill_store = SkillStore(skills_dir)
        self.tools = AgentTools(self.skill_store)
        self.conversation_log_path = Path(conversation_log_path)
        self._logger = logging.getLogger(__name__)
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            http_client=httpx.Client(trust_env=False),
            base_url="http://api.openai.rnd.huawei.com/v1/",
            api_key="sk-1234",
            temperature=0
        ).bind_tools(self.tools.langchain_tools(), tool_choice="required")

    def chat(self, messages: list[dict[str, str]], max_steps: int = 12) -> dict[str, Any]:
        history: list[BaseMessage] = [self._system_message()]
        history.extend(self._convert_messages(messages))

        for _ in range(max_steps):
            ai_message: AIMessage = self.llm.invoke(history)
            history.append(ai_message)

            if not ai_message.tool_calls:
                error_response = {"error": "Model returned no tool call; tool_choice='required' should prevent this."}
                self._log_conversation(messages, error_response)
                return error_response

            for call in ai_message.tool_calls:
                result = self.tools.dispatch_tool(call["name"], call["args"])
                history.append(ToolMessage(content=result, tool_call_id=call["id"]))

                if call["name"] == "respond_to_user":
                    payload = json.loads(result)
                    response = {
                        "message": payload["final"],
                        "steps": _serialize_steps(history),
                    }
                    self._log_conversation(messages, response)
                    return response

        error_response = {"error": "Agent hit max_steps without calling respond_to_user."}
        self._log_conversation(messages, error_response)
        return error_response

    def _log_conversation(self, messages: list[dict[str, str]], response: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "messages": messages,
            "response": response,
        }
        try:
            self.conversation_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.conversation_log_path.open("a", encoding="utf-8") as fp:
                fp.write(f"{json.dumps(payload, ensure_ascii=False)}\n")
        except OSError:
            self._logger.exception("Failed to write conversation log")

    def _system_message(self) -> SystemMessage:
        headers = self.skill_store.headers()
        return SystemMessage(
            content=(
                "You are a tools-only agent. You must ALWAYS return one or more tool calls and never plain text. "
                "Workflow: first inspect the available skill headers below, then call get_skills(skill_id) when a skill is relevant, "
                "use web_request if you need web data, and ALWAYS finish by calling respond_to_user.\n\n"
                "For rental scenarios, enforce this policy: "
                "(1) extract explicit constraints (budget, district/area, bedrooms, rental type, commute, facilities), "
                "(2) ask one focused follow-up question when key constraints are missing or ambiguous, "
                "(3) verify and compare candidate listings across dimensions (commute, price-performance, amenities, facilities, risk), "
                "(4) return practical recommendations with clear pros/cons, and "
                "(5) cap final recommended candidates to at most 5 listings.\n\n"
                f"Available skill headers:\n{json.dumps(headers, indent=2)}"
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
