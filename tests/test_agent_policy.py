from __future__ import annotations

import json
from pathlib import Path

from app.agent import AgentRuntime


def test_system_prompt_includes_rental_requirement_policy() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)

    class _DummySkillStore:
        def headers(self):
            return [{"skill_id": "rental-house-search", "name": "search", "description": "..."}]

    runtime.skill_store = _DummySkillStore()

    system_message = runtime._system_message()
    content = system_message.content

    assert "extract explicit constraints" in content
    assert "ask one focused follow-up question" in content
    assert "verify and compare candidate listings" in content
    assert "cap final recommended candidates to at most 5 listings" in content


def test_convert_messages_maps_roles() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u"},
    ]

    converted = AgentRuntime._convert_messages(messages)

    assert [m.type for m in converted] == ["system", "ai", "human"]


def test_log_conversation_writes_jsonl(tmp_path) -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.conversation_log_path = tmp_path / "agent_conversations.jsonl"

    messages = [{"role": "user", "content": "hello"}]
    response = {"message": "world", "steps": []}

    runtime._log_conversation(messages, response)

    lines = runtime.conversation_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["messages"] == messages
    assert payload["response"] == response
    assert "timestamp" in payload


async def test_chat_returns_direct_response_when_llm_emits_no_tool_calls() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)

    class _DummySkillStore:
        def headers(self):
            return []

    class _DummyLLM:
        def invoke(self, _history):
            from langchain_core.messages import AIMessage

            return AIMessage(content="direct answer", tool_calls=[])

    import logging

    runtime.skill_store = _DummySkillStore()
    runtime.llm = _DummyLLM()
    runtime._logger = logging.getLogger(__name__)
    runtime.conversation_log_path = Path('/tmp/test_agent_conversations.jsonl')

    response = await runtime.chat([{"role": "user", "content": "hello"}], max_steps=1)

    assert response["message"] == "direct answer"
    assert response["steps"] == []


def test_langchain_tools_excludes_respond_to_user() -> None:
    class _DummySkillStore:
        def get(self, _skill_id):
            return None

    from app.tools import AgentTools

    tool_names = [tool["function"]["name"] for tool in AgentTools(_DummySkillStore()).langchain_tools()]

    assert "respond_to_user" not in tool_names
