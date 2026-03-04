from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.agent import AgentRuntime


def test_system_prompt_includes_rental_requirement_policy() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)

    class _DummySkillStore:
        def headers(self):
            return [{"skill_id": "property_search", "name": "search", "description": "..."}]

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

        def tool_whitelist_for(self, _selected_skills):
            return None

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
    runtime = AgentRuntime.__new__(AgentRuntime)
    payload = '{"message":"暂无符合条件的房源","houses":[]}'
    assert asyncio.run(runtime._format_final_content(payload, None)) == "暂无符合条件的房源"


def test_format_final_content_returns_json_when_houses_present() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)
    payload = '{"message":"为您找到以下符合条件的房源：","houses":["HF_1"]}'
    assert json.loads(asyncio.run(runtime._format_final_content(payload, None))) == json.loads(payload)


def test_format_final_content_handles_markdown_json_block() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)
    payload = '```json\n{"message":"您好","houses":[]}\n```'
    assert asyncio.run(runtime._format_final_content(payload, None)) == "您好"


def test_format_final_content_filters_non_string_house_ids() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)
    payload = '{"message":"为您找到房源","houses":["HF_1", 2, null]}'
    assert json.loads(asyncio.run(runtime._format_final_content(payload, None))) == {
        "message": "为您找到房源",
        "houses": ["HF_1"],
    }


def test_parse_selected_skill_ids_filters_unknown_and_duplicates() -> None:
    headers = [
        {"skill_id": "property_search", "name": "search", "description": ""},
        {"skill_id": "property_management", "name": "actions", "description": ""},
    ]

    parsed = AgentRuntime._parse_selected_skill_ids(
        '["property_search", "unknown", "property_search", "property_management"]',
        headers,
    )

    assert parsed == ["property_search", "property_management"]


def test_parse_selected_skill_ids_handles_structured_output_with_raw_response() -> None:
    headers = [
        {"skill_id": "property_search", "name": "search", "description": ""},
        {"skill_id": "property_management", "name": "actions", "description": ""},
    ]

    parsed = AgentRuntime._parse_selected_skill_ids(
        '{"selected_skills": [], "raw_response": "\\n\\n[{\"skill_id\": \"property_search\"}]"}',
        headers,
    )

    assert parsed == ["property_search"]


def test_system_prompt_contains_skill_select_block() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)

    class _DummySkillStore:
        def headers(self):
            return [{"skill_id": "property_search", "name": "search", "description": "..."}]

    runtime.skill_store = _DummySkillStore()

    system_message = runtime._system_message(selected_skills=["property_search"])
    content = system_message.content

    assert "SKILL_HEADERS" in content
    assert "SKILL_SELECT" in content
    assert "property_search" in content


def test_chat_short_circuits_when_tool_call_sets_final_answer_true() -> None:
    from langchain_core.messages import AIMessage
    import logging

    runtime = AgentRuntime.__new__(AgentRuntime)

    class _DummySkillStore:
        def headers(self):
            return []

        def tool_whitelist_for(self, _selected_skills):
            return None

    class _DummyLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, _history):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("LLM should not be invoked again after final_answer tool call")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_final_1",
                        "name": "get_houses_by_platform",
                        "args": {"district": "海淀", "page_size": 5, "final_answer": True},
                    }
                ],
            )

    class _DummyTools:
        async def dispatch_tool(self, _name, _args):
            return json.dumps({"houses": [{"houseid": "HF_4", "platforms": ["链家"]}, {"houseid": "HF_6", "platforms": ["安居客"]}]}, ensure_ascii=False)

    async def _select_skills(_messages):
        return []

    runtime.skill_store = _DummySkillStore()
    runtime.tools = _DummyTools()
    runtime.llm = _DummyLLM()
    runtime.structured_llm = None
    runtime._logger = logging.getLogger(__name__)
    runtime.conversation_log_path = Path('/tmp/test_agent_conversations.jsonl')
    runtime.default_model = "qwen3-32b"
    runtime._select_skills_for_request = _select_skills

    response = asyncio.run(runtime.chat([{"role": "user", "content": "帮我找房"}], max_steps=3))

    assert response["message"] == '{"message":"为您找到以下符合条件的房源：","houses":["HF_4","HF_6"]}'
    assert len(response["steps"]) == 2
    assert response["steps"][0]["type"] == "tool_calls"
    assert response["steps"][1]["type"] == "tool_result"


def test_select_skills_uses_tiny_agent_abstraction() -> None:
    import logging

    runtime = AgentRuntime.__new__(AgentRuntime)

    class _DummySkillStore:
        def headers(self):
            return [
                {"skill_id": "property_search", "name": "search", "description": ""},
                {"skill_id": "landmark_search", "name": "landmark", "description": ""},
            ]

    captured: dict[str, object] = {}

    async def _fake_tiny_agent(prompt, output_class):
        captured["prompt"] = prompt
        captured["output_class"] = output_class
        return output_class(selected_skills=["property_search"])

    runtime.skill_store = _DummySkillStore()
    runtime._logger = logging.getLogger(__name__)
    runtime.default_model = "qwen3-32b"
    runtime.default_base_url = "http://api.openai.rnd.huawei.com/v1/"

    selected = asyncio.run(
        runtime._select_skills_for_request(
            [{"role": "user", "content": "帮我找西二旗附近房子"}],
            tiny_agent=_fake_tiny_agent,
        )
    )

    assert selected == ["property_search"]
    assert "request=帮我找西二旗附近房子" in str(captured["prompt"])
    assert captured["output_class"].__name__ == "SelectedSkills"


def test_run_tiny_agent_returns_structured_output() -> None:
    import app.agent as agent_module

    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.default_model = "qwen3-32b"
    runtime.default_base_url = "http://api.openai.rnd.huawei.com/v1/"

    class _Output(BaseModel):
        selected_skills: list[str] = Field(default_factory=list)

    class _FakeStructuredLLM:
        def __init__(self, output_class):
            self._output_class = output_class

        def invoke(self, _messages, config=None):
            assert config is None
            return self._output_class(selected_skills=["property_search"])

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def with_structured_output(self, output_class):
            return _FakeStructuredLLM(output_class)

    original = agent_module.ChatOpenAI
    agent_module.ChatOpenAI = _FakeChatOpenAI
    try:
        output = asyncio.run(runtime._run_tiny_agent(prompt="choose", output_class=_Output))
    finally:
        agent_module.ChatOpenAI = original

    assert output.selected_skills == ["property_search"]


def test_chat_passes_base_url_and_session_id_into_tiny_agent_runner() -> None:
    import logging
    import app.agent as agent_module
    from langchain_core.messages import AIMessage

    runtime = AgentRuntime.__new__(AgentRuntime)

    class _DummySkillStore:
        def tool_whitelist_for(self, _selected_skills):
            return None

    class _DummyLLM:
        def invoke(self, _history, config=None):
            return AIMessage(content='{"message":"ok","houses":[]}', tool_calls=[])

    class _Output(BaseModel):
        selected_skills: list[str] = Field(default_factory=list)

    captured: dict[str, object] = {}

    class _FakeStructuredLLM:
        def __init__(self, output_class):
            self._output_class = output_class

        def invoke(self, _messages, config=None):
            captured["invoke_config"] = config
            return self._output_class(selected_skills=[])

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["chat_kwargs"] = kwargs

        def with_structured_output(self, output_class):
            captured["output_class"] = output_class
            return _FakeStructuredLLM(output_class)

    async def _fake_select_skills(messages, tiny_agent=None):
        assert tiny_agent is not None
        await tiny_agent("ping", _Output)
        return []

    original = agent_module.ChatOpenAI
    agent_module.ChatOpenAI = _FakeChatOpenAI
    try:
        runtime.skill_store = _DummySkillStore()
        runtime._logger = logging.getLogger(__name__)
        runtime.default_model = "qwen3-32b"
        runtime.default_base_url = "http://default.local/v1"
        runtime.llm = _DummyLLM()
        runtime.structured_llm = None
        runtime.conversation_log_path = Path('/tmp/test_agent_conversations.jsonl')
        runtime._select_skills_for_request = _fake_select_skills

        response = asyncio.run(
            runtime._chat(
                [{"role": "user", "content": "hello"}],
                max_steps=1,
                base_url="http://custom.local/v1",
                session_id="session-123",
            )
        )
    finally:
        agent_module.ChatOpenAI = original

    assert response["message"] == "ok"
    chat_kwargs = captured["chat_kwargs"]
    assert chat_kwargs["base_url"] == "http://custom.local/v1"
    assert chat_kwargs["model"] == "qwen3-32b"
    assert captured["invoke_config"] == {"callbacks": [agent_module.langfuse_handler]}


def test_resolve_landmark_memories_returns_tool_result_entries() -> None:
    from langchain_core.messages import AIMessage, ToolMessage
    import logging

    runtime = AgentRuntime.__new__(AgentRuntime)

    class _DummyTools:
        async def dispatch_tool(self, _name, _args):
            return '{"name": "西二旗站", "id": "LM_1"}'

    class _LandmarkOutput:
        def __init__(self, names):
            self.names = names

    async def _fake_tiny_agent(_prompt, _output_class):
        return _LandmarkOutput(names=["西二旗", "西二旗"])

    runtime.tools = _DummyTools()
    runtime._logger = logging.getLogger(__name__)

    entries, filtered_skills = asyncio.run(
        runtime._resolve_landmark_memories(
            messages=[{"role": "user", "content": "帮我找西二旗附近"}],
            selected_skills=["property_search", "landmark_search"],
            tiny_agent=_fake_tiny_agent,
        )
    )

    assert filtered_skills == ["property_search"]
    assert len(entries) == 2
    assert isinstance(entries[0], AIMessage)
    assert isinstance(entries[1], ToolMessage)
    assert entries[0].tool_calls[0]["name"] == "search_landmarks"
    assert entries[1].name == "search_landmarks"
    assert entries[1].content == '{"name": "西二旗站", "id": "LM_1"}'
