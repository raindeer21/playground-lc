from __future__ import annotations

import json

from app.openapi_tools import OpenAPIToolRegistry
from app.tools import AgentTools


SPEC = {
    "openapi": "3.0.3",
    "servers": [{"url": "http://example.com"}],
    "paths": {
        "/api/landmarks": {
            "get": {
                "operationId": "get_landmarks",
                "parameters": [
                    {"name": "district", "in": "query", "required": False, "schema": {"type": "string"}}
                ],
            }
        },
        "/api/houses/{house_id}": {
            "get": {
                "operationId": "get_house_by_id",
                "parameters": [
                    {"name": "house_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
            }
        },
    },
}


class _DummySkillStore:
    def get(self, _skill_id):
        return None


def test_openapi_registry_parses_operations() -> None:
    registry = OpenAPIToolRegistry(SPEC)

    assert sorted(registry.operations.keys()) == ["get_house_by_id", "get_landmarks"]


def test_openapi_operation_call_builds_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = '{"ok":true}'

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

    monkeypatch.setattr("app.openapi_tools.httpx.Client", _DummyClient)

    registry = OpenAPIToolRegistry(SPEC)
    payload = json.loads(registry.call_operation("get_house_by_id", house_id="HF_1"))

    assert captured["client_kwargs"]["trust_env"] is False
    assert captured["request_kwargs"] == {
        "method": "GET",
        "url": "http://example.com/api/houses/HF_1",
        "params": {},
        "headers": {},
    }
    assert payload["status_code"] == 200


def test_agent_tools_exposes_openapi_operations_as_langchain_tools() -> None:
    registry = OpenAPIToolRegistry(SPEC)
    tools = AgentTools(_DummySkillStore(), openapi_registry=registry)

    tool_names = [tool.name for tool in tools.langchain_tools()]

    assert "get_landmarks" in tool_names
    assert "get_house_by_id" in tool_names
