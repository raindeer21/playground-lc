import json

from app.agent_google_adk import ADKMessage, GoogleADKAgentRuntime


def _make_runtime() -> GoogleADKAgentRuntime:
    runtime = GoogleADKAgentRuntime.__new__(GoogleADKAgentRuntime)
    return runtime


def test_convert_messages_supports_all_roles() -> None:
    runtime = _make_runtime()
    converted = runtime._convert_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "tool_calls": [{"id": "c1", "name": "x", "args": {}}]},
            {"role": "tool", "content": "t", "tool_call_id": "c1", "name": "x"},
        ]
    )

    assert [m.role for m in converted] == ["system", "user", "assistant", "tool"]
    assert converted[2].tool_calls == [{"id": "c1", "name": "x", "args": {}}]
    assert converted[3].tool_call_id == "c1"


def test_step_serialization_parity_for_tool_calls_and_results() -> None:
    runtime = _make_runtime()
    history = [
        ADKMessage(role="assistant", content="", tool_calls=[{"id": "call_1", "name": "current_properties", "args": {"houses": ["HF_1"]}}]),
        ADKMessage(role="tool", content='{"message": "ok", "houses": ["HF_1"]}', tool_call_id="call_1", name="current_properties"),
    ]

    steps = runtime._serialize_steps(runtime._adk_to_langchain(history))
    assert steps[0]["type"] == "tool_calls"
    assert steps[1]["type"] == "tool_result"
    assert steps[1]["tool_call_id"] == "call_1"


def test_step_serialization_compression_hook_kept() -> None:
    runtime = _make_runtime()
    history = [
        ADKMessage(role="assistant", content="", tool_calls=[{"id": "call_2", "name": "get_houses_by_platform", "args": {}}]),
        ADKMessage(
            role="tool",
            content=(
                "{'code': 0, 'message': 'success', 'data': {'total': 1, 'items': "
                "[{'house_id': 'HF_3', 'community': '建清园(南区)', 'status': 'available', 'price': 7600}]}}"
            ),
            tool_call_id="call_2",
            name="get_houses_by_platform",
        ),
    ]

    compressed_steps = runtime._serialize_steps(runtime._adk_to_langchain(history), compressed=True)
    payload = json.loads(compressed_steps[1]["content"])
    assert payload["data"]["items"] == [{"house_id": "HF_3", "status": "available"}]
