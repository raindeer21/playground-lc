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



def test_serialize_steps_compresses_get_houses_nearby_tool_result() -> None:
    from langchain_core.messages import AIMessage, ToolMessage
    from app.agent import _serialize_steps

    output = (
        "{'code': 0, 'message': 'success', 'data': {"
        "'landmark': {'id': 'SS_001', 'name': '西二旗站', 'longitude': 116.3289, 'latitude': 40.0567}, "
        "'total': 2, "
        "'items': ["
        "{'house_id': 'HF_36', 'status': 'available', 'distance_to_landmark': 0, 'walking_distance': 0, 'walking_duration': 0, 'listing_platform': '安居客', 'price': 12550}, "
        "{'house_id': 'HF_37', 'status': 'available', 'distance_to_landmark': 10, 'walking_distance': 50, 'walking_duration': 1, 'listing_platform': '链家', 'price': 8200}"
        "]}}"
    )

    history = [
        AIMessage(content='', tool_calls=[{'id': 'call_1', 'name': 'get_houses_nearby', 'args': {}}]),
        ToolMessage(content=output, tool_call_id='call_1'),
    ]

    steps = _serialize_steps(history)
    compressed = json.loads(steps[1]['content'])

    assert compressed['data']['landmark'] == {'id': 'SS_001', 'name': '西二旗站'}
    assert compressed['data']['items'] == [
        {
            'house_id': 'HF_36',
            'status': 'available',
            'distance_to_landmark': 0,
            'walking_distance': 0,
            'walking_duration': 0,
            'listing_platform': '安居客',
        },
        {
            'house_id': 'HF_37',
            'status': 'available',
            'distance_to_landmark': 10,
            'walking_distance': 50,
            'walking_duration': 1,
            'listing_platform': '链家',
        },
    ]


def test_serialize_steps_keeps_other_tool_result_unmodified() -> None:
    from langchain_core.messages import AIMessage, ToolMessage
    from app.agent import _serialize_steps

    output = '{"foo": "bar", "data": {"landmark": {"id": "x"}, "items": []}}'

    history = [
        AIMessage(content='', tool_calls=[{'id': 'call_2', 'name': 'current_properties', 'args': {}}]),
        ToolMessage(content=output, tool_call_id='call_2'),
    ]

    steps = _serialize_steps(history)
    assert steps[1]['content'] == output



def test_serialize_steps_compresses_get_houses_by_platform_tool_result() -> None:
    from langchain_core.messages import AIMessage, ToolMessage
    from app.agent import _serialize_steps

    output = (
        "{'code': 0, 'message': 'success', 'data': {'total': 108, 'page': 1, 'page_size': 5, 'items': ["
        "{'house_id': 'HF_3', 'community': '建清园(南区)', 'status': 'available', 'price': 7600}, "
        "{'house_id': 'HF_33', 'community': '车道沟南里小区', 'status': 'available', 'price': 5500}"
        "]}}"
    )

    history = [
        AIMessage(content='', tool_calls=[{'id': 'call_3', 'name': 'get_houses_by_platform', 'args': {}}]),
        ToolMessage(content=output, tool_call_id='call_3'),
    ]

    steps = _serialize_steps(history)
    compressed = json.loads(steps[1]['content'])

    assert compressed['data']['total'] == 108
    assert compressed['data']['page'] == 1
    assert compressed['data']['page_size'] == 5
    assert compressed['data']['items'] == [
        {'house_id': 'HF_3', 'status': 'available'},
        {'house_id': 'HF_33', 'status': 'available'},
    ]



def test_serialize_steps_compresses_get_houses_by_community_tool_result() -> None:
    from langchain_core.messages import AIMessage, ToolMessage
    from app.agent import _serialize_steps

    output = (
        "{'code': 0, 'message': 'success', 'data': {'total': 5, 'page': 1, 'page_size': 5, 'items': ["
        "{'house_id': 'HF_36', 'community': '智学苑', 'listing_platform': '安居客', 'status': 'available', 'price': 12550}, "
        "{'house_id': 'HF_37', 'community': '智学苑', 'listing_platform': '安居客', 'status': 'available', 'price': 8200}"
        "]}}"
    )

    history = [
        AIMessage(content='', tool_calls=[{'id': 'call_4', 'name': 'get_houses_by_community', 'args': {}}]),
        ToolMessage(content=output, tool_call_id='call_4'),
    ]

    steps = _serialize_steps(history)
    compressed = json.loads(steps[1]['content'])

    assert compressed['data']['total'] == 5
    assert compressed['data']['page'] == 1
    assert compressed['data']['page_size'] == 5
    assert compressed['data']['items'] == [
        {
            'house_id': 'HF_36',
            'community': '智学苑',
            'listing_platform': '安居客',
            'status': 'available',
        },
        {
            'house_id': 'HF_37',
            'community': '智学苑',
            'listing_platform': '安居客',
            'status': 'available',
        },
    ]


def test_format_final_content_returns_message_only_when_houses_empty() -> None:
    payload = '{"message":"暂无符合条件的房源","houses":[]}'
    assert AgentRuntime._format_final_content(payload) == "暂无符合条件的房源"


def test_format_final_content_returns_json_when_houses_present() -> None:
    payload = '{"message":"为您找到以下符合条件的房源：","houses":["HF_1"]}'
    assert json.loads(AgentRuntime._format_final_content(payload)) == json.loads(payload)
