import os

from fastapi.testclient import TestClient


class _DummyRuntime:
    def chat(self, _messages):
        return {
            "message": "已为你筛选 3 套高匹配房源",
            "steps": [{"type": "tool_calls", "tool_calls": []}],
        }


class _ErrorRuntime:
    def chat(self, _messages):
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
