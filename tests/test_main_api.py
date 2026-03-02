import os

from fastapi.testclient import TestClient


class _DummyRuntime:
    def __init__(self):
        self.last_call = None

    def chat(self, _messages, **kwargs):
        self.last_call = kwargs
        return {
            "message": "已为你筛选 3 套高匹配房源",
            "steps": [
                {"type": "tool_calls", "tool_calls": [{"name": "bash", "args": {}}]},
                {"type": "tool_result", "content": "ok"},
            ],
        }




class _PropertyResultRuntime:
    def chat(self, _messages, **_kwargs):
        return {
            "message": "done",
            "steps": [
                {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "provide_property_result_list",
                            "args": {
                                "message": "为您找到以下符合条件的房源：",
                                "houses": ["HF_4", "HF_6", "HF_277"],
                            },
                        }
                    ],
                },
                {
                    "type": "tool_result",
                    "content": '{"message":"为您找到以下符合条件的房源：","houses":["HF_4","HF_6","HF_277"]}',
                },
            ],
        }

class _ErrorRuntime:
    def chat(self, _messages, **_kwargs):
        return {"error": "Agent hit max_steps without calling respond_to_user."}


def test_chat_completions_success_shape(monkeypatch) -> None:
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    from app import main

    monkeypatch.setattr(main, "runtime", _DummyRuntime())
    client = TestClient(main.app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "帮我找房"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "已为你筛选 3 套高匹配房源"
    assert isinstance(payload["steps"], list)


def call_with_history(client, history, msg):
    history.append({"role": "user", "content": msg})
    response = client.post(
        "/v1/chat/completions",
        json={"messages": history},
    )
    assert response.status_code == 200
    history.append(response.json()["choices"][0]["message"])
    return response.json()


def test_chat() -> None:

    from app import main
    history = []
    client = TestClient(main.app)

    print(call_with_history(client, history, "帮我在西二旗租一个二房的房子"))
    print(call_with_history(client, history, "整租 预算5000"))


def test_chat_completions_passes_model_to_runtime(monkeypatch) -> None:
    from app import main

    dummy_runtime = _DummyRuntime()
    monkeypatch.setattr(main, "runtime", dummy_runtime)
    client = TestClient(main.app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "custom-model", "messages": [{"role": "user", "content": "帮我找房"}]},
    )

    assert response.status_code == 200
    assert dummy_runtime.last_call == {"model": "custom-model"}


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




def test_agent_chat_treats_model_ip_as_base_url(monkeypatch) -> None:
    from app import main

    dummy_runtime = _DummyRuntime()
    monkeypatch.setattr(main, "runtime", dummy_runtime)
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/chat",
        json={
            "model_ip": "http://127.0.0.1:11434/v1",
            "session_id": "abc-session-2",
            "message": "查询海淀区的房源",
        },
    )

    assert response.status_code == 200
    assert dummy_runtime.last_call == {
        "session_id": "abc-session-2",
        "base_url": "http://127.0.0.1:11434/v1",
    }

def call_with_history(client, history, msg):
    history.append({"role": "user", "content": msg})
    response = client.post(
        "/v1/chat/completions",
        json={"messages": history},
    )
    assert response.status_code == 200
    history.append(response.json()["choices"][0]["message"])
    return response.json()

def test_chat() -> None:

    from app import main
    main.runtime = _DummyRuntime()
    history = []
    client = TestClient(main.app)

    print(call_with_history(client, history, "帮我在西二旗租一个二房的房子"))
    print(call_with_history(client, history, "整租 预算5000"))

def test_chat_completions_error_shape(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(main, "runtime", _ErrorRuntime())
    client = TestClient(main.app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "帮我找房"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "chatcmpl-error"
    assert "max_steps" in payload["choices"][0]["message"]["content"]


def test_agent_chat_success_shape(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(main, "runtime", _DummyRuntime())
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/chat",
        json={"model_ip": "http://127.0.0.1:11434/v1", "session_id": "abc123", "message": "查询海淀区的房源"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "abc123"
    assert payload["response"] == "已为你筛选 3 套高匹配房源"
    assert payload["status"] == "success"
    assert payload["tool_results"][0]["name"] == "bash"
    assert payload["tool_results"][0]["success"] is True
    assert isinstance(payload["timestamp"], int)
    assert isinstance(payload["duration_ms"], int)


def test_agent_chat_error_shape(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(main, "runtime", _ErrorRuntime())
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/chat",
        json={"model_ip": "http://127.0.0.1:11434/v1", "session_id": "abc124", "message": "查询海淀区的房源"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "abc124"
    assert payload["status"] == "error"
    assert payload["tool_results"] == []


def test_agent_chat_returns_property_result_when_tool_called(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(main, "runtime", _PropertyResultRuntime())
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/chat",
        json={"model_ip": "http://127.0.0.1:11434/v1", "session_id": "abc125", "message": "帮我找房"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["message"] == "为您找到以下符合条件的房源："
    assert payload["houses"] == ["HF_4", "HF_6", "HF_277"]
