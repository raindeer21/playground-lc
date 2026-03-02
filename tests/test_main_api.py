import importlib

from fastapi.testclient import TestClient


class _DummyRuntime:
    def __init__(self):
        self.last_call = None

    async def chat(self, _messages, **kwargs):
        self.last_call = kwargs
        return {
            "message": "已为你筛选 3 套高匹配房源",
            "steps": [
                {"type": "tool_calls", "tool_calls": [{"name": "bash", "args": {}}]},
                {"type": "tool_result", "content": "ok", "tool_call_id": "c1"},
            ],
            "compressed_steps": [
                {"type": "tool_calls", "tool_calls": [{"name": "bash", "args": {}}]},
                {"type": "tool_result", "content": "ok", "tool_call_id": "c1"},
            ],
        }


class _ErrorRuntime:
    async def chat(self, _messages, **_kwargs):
        return {"error": "Agent hit max_steps without producing a direct response."}


def test_chat_completions_success_shape(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(main, "runtime", _DummyRuntime())
    client = TestClient(main.app)

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "帮我找房"}]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "已为你筛选 3 套高匹配房源"
    assert isinstance(payload["steps"], list)


def test_chat_completions_error_shape(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(main, "runtime", _ErrorRuntime())
    client = TestClient(main.app)

    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "帮我找房"}]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "chatcmpl-error"


def test_agent_chat_passes_base_url_and_session_to_runtime(monkeypatch) -> None:
    from app import main

    dummy_runtime = _DummyRuntime()
    monkeypatch.setattr(main, "runtime", dummy_runtime)
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/chat",
        json={"model_ip": "http://127.0.0.1:11434/v1", "session_id": "abc-session", "message": "查询海淀区的房源"},
    )
    assert response.status_code == 200
    assert dummy_runtime.last_call == {"session_id": "abc-session", "base_url": "http://127.0.0.1:11434/v1"}


def test_runtime_backend_selection(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_RUNTIME_BACKEND", "google_adk")
    from app import main

    importlib.reload(main)
    assert main.runtime.__class__.__name__ == "GoogleADKAgentRuntime"

    monkeypatch.setenv("AGENT_RUNTIME_BACKEND", "langchain")
    importlib.reload(main)
    assert main.runtime.__class__.__name__ == "AgentRuntime"


def test_api_response_schema_parity_between_backends(monkeypatch) -> None:
    from app import main

    client = TestClient(main.app)
    for runtime in (_DummyRuntime(), _DummyRuntime()):
        monkeypatch.setattr(main, "runtime", runtime)
        response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "帮我找房"}]})
        assert response.status_code == 200
        payload = response.json()
        assert set(payload.keys()) == {"id", "object", "choices", "steps"}
