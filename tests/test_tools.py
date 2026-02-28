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


def test_house_search_calls_all_platforms_and_returns_house_platform_pairs(monkeypatch) -> None:
    from app.tools import house_search

    captured_requests: list[dict[str, object]] = []

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _DummyClient:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path, params):
            captured_requests.append({"path": path, "params": params})
            platform = params["listing_platform"]
            payload_by_platform = {
                "链家": {"houses": [{"house_id": "A1"}, {"house_id": "A2"}]},
                "安居客": {"houses": [{"house_id": "A2"}, {"house_id": "B1"}]},
                "58同城": {"houses": [{"house_id": "C1"}]},
            }
            return _Response(payload_by_platform[platform])

    monkeypatch.setattr("app.tools.httpx.Client", _DummyClient)

    payload = json.loads(house_search(district="海淀", min_price=3000, page_size=5))

    assert [r["params"]["listing_platform"] for r in captured_requests] == ["链家", "安居客", "58同城"]
    assert all(r["path"] == "/api/houses/by_platform" for r in captured_requests)
    assert all(r["params"]["page_size"] == 5 for r in captured_requests)
    assert payload["houses"] == [
        {"houseid": "A1", "platform": "链家"},
        {"houseid": "A2", "platform": "链家"},
        {"houseid": "A2", "platform": "安居客"},
        {"houseid": "B1", "platform": "安居客"},
        {"houseid": "C1", "platform": "58同城"},
    ]


def test_get_houses_list_nearby_calls_all_platforms_and_returns_house_platform_pairs(monkeypatch) -> None:
    from app.tools import get_houses_list_nearby

    captured_requests: list[dict[str, object]] = []

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _DummyClient:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path, params):
            captured_requests.append({"path": path, "params": params})
            platform = params["listing_platform"]
            payload_by_platform = {
                "链家": {"houses": [{"house_id": "N1"}, {"house_id": "N2"}]},
                "安居客": {"houses": [{"house_id": "N3"}]},
                "58同城": {"houses": [{"house_id": "N4"}]},
            }
            return _Response(payload_by_platform[platform])

    monkeypatch.setattr("app.tools.httpx.Client", _DummyClient)

    payload = json.loads(get_houses_list_nearby(landmark_id="LM_1", max_distance=1200, page_size=5))

    assert [r["params"]["listing_platform"] for r in captured_requests] == ["链家", "安居客", "58同城"]
    assert all(r["path"] == "/api/houses/nearby" for r in captured_requests)
    assert all(r["params"]["page_size"] == 5 for r in captured_requests)
    assert payload["houses"] == [
        {"houseid": "N1", "platform": "链家"},
        {"houseid": "N2", "platform": "链家"},
        {"houseid": "N3", "platform": "安居客"},
        {"houseid": "N4", "platform": "58同城"},
    ]
