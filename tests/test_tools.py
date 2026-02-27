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


def test_dispatch_tool_uses_mcp_for_get_skills() -> None:
    tools = AgentTools(_DummySkillStore())

    result = tools.dispatch_tool("get_skills", {"skill_id": "demo"})
    payload = json.loads(result)

    assert payload["skills"] == [{"skill_id": "demo", "content": "demo content"}]


def test_dispatch_tool_supports_multi_skill_payload() -> None:
    tools = AgentTools(_DummySkillStore())

    result = tools.dispatch_tool("get_skills", {"skill_ids": ["demo", "missing"]})
    payload = json.loads(result)

    assert payload["skills"] == [{"skill_id": "demo", "content": "demo content"}]
    assert payload["errors"] == [{"skill_id": "missing", "error": "Unknown skill_id: missing"}]


def test_dispatch_tool_unknown_tool_returns_error() -> None:
    tools = AgentTools(_DummySkillStore())

    payload = json.loads(tools.dispatch_tool("missing_tool", {}))

    assert "error" in payload
