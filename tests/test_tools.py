from __future__ import annotations

import json

from app.tools import AgentTools


class _Skill:
    def __init__(self, body: str):
        self.body = body


class _DummySkillStore:
    def get(self, skill_id: str):
        if skill_id == "demo":
            return _Skill("demo content")
        return None


async def test_dispatch_tool_uses_mcp_for_get_skills() -> None:
    tools = AgentTools(_DummySkillStore())

    result = await tools.dispatch_tool("get_skills", {"skill_id": "demo"})
    payload = json.loads(result)

    assert payload["skills"] == [{"skill_id": "demo", "content": "demo content"}]


async def test_dispatch_tool_supports_multi_skill_payload() -> None:
    tools = AgentTools(_DummySkillStore())

    result = await tools.dispatch_tool("get_skills", {"skill_ids": ["demo", "missing"]})
    payload = json.loads(result)

    assert payload["skills"] == [{"skill_id": "demo", "content": "demo content"}]
    assert payload["errors"] == [{"skill_id": "missing", "error": "Unknown skill_id: missing"}]


async def test_dispatch_tool_unknown_tool_returns_error() -> None:
    tools = AgentTools(_DummySkillStore())

    payload = json.loads(await tools.dispatch_tool("missing_tool", {}))

    assert "error" in payload


def test_web_request_uses_httpx_and_disables_proxy_env(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        status_code = 200
        headers = {"content-type": "text/plain"}
        text = "ok"

    class _DummyClient:
        def __init__(self, *args, **kwargs):
            captured["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, **kwargs):
            captured["request_kwargs"] = kwargs
            return _Response()

    monkeypatch.setattr("app.tools.httpx.Client", _DummyClient)

    from app.tools import web_request

    payload = json.loads(web_request(method="get", url="https://example.com", headers={"x": "y"}, body="demo"))

    assert captured["client_kwargs"]["trust_env"] is False
    assert captured["request_kwargs"] == {
        "method": "GET",
        "url": "https://example.com",
        "headers": {"x": "y"},
        "content": "demo",
    }
    assert payload["status_code"] == 200


async def test_dispatch_tool_provide_property_result_list_returns_payload() -> None:
    tools = AgentTools(_DummySkillStore())

    result = await tools.dispatch_tool(
        "provide_property_result_list",
        {"message": "为您找到以下符合条件的房源：", "houses": ["HF_4", "HF_6", "HF_277"]},
    )
    payload = json.loads(result)

    assert payload == {
        "message": "为您找到以下符合条件的房源：",
        "houses": ["HF_4", "HF_6", "HF_277"],
    }


def test_agent_tools_openapi_uses_async_http_client(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _DummyChildServer:
        pass

    def _fake_from_openapi(spec, client):
        captured["client"] = client
        return _DummyChildServer()

    async def _fake_import_server(self, server, prefix=""):
        return None

    monkeypatch.setattr("app.tools.FastMCP.from_openapi", _fake_from_openapi)
    monkeypatch.setattr("app.tools.FastMCP.import_server", _fake_import_server)
    monkeypatch.setattr("app.tools._openapi_loaded", False)

    AgentTools(_DummySkillStore())

    import httpx

    assert isinstance(captured["client"], httpx.AsyncClient)
