from __future__ import annotations

import asyncio
import json
import logging
from functools import partial

import httpx
from fastmcp import Client, FastMCP
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator

from app.skills import SkillStore

logger = logging.getLogger(__name__)

mcp = FastMCP("agent-tools")
_skill_store: SkillStore | None = None


class GetSkillsInput(BaseModel):
    skill_id: str | None = Field(
        default=None,
        description="Single skill_id to load. Prefer skill_ids for loading multiple skills in one call.",
    )
    skill_ids: list[str] | None = Field(
        default=None,
        description="List of skill_ids to load in a single call. Prefer this to reduce tool round-trips.",
    )

    @model_validator(mode="after")
    def _validate_inputs(self) -> "GetSkillsInput":
        if self.skill_id is None and not self.skill_ids:
            raise ValueError("Provide skill_id or skill_ids")
        if self.skill_id is not None and self.skill_ids:
            raise ValueError("Provide either skill_id or skill_ids, not both")
        return self


class WebRequestInput(BaseModel):
    method: str = Field(default="GET", description="HTTP method")
    url: str = Field(..., description="Target URL")
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = Field(default=None)


def set_skill_store(skill_store: SkillStore) -> None:
    global _skill_store
    _skill_store = skill_store


def _require_skill_store() -> SkillStore:
    if _skill_store is None:
        raise RuntimeError("Skill store is not initialized")
    return _skill_store


def get_skills(
    skill_store: SkillStore,
    skill_id: str | None = None,
    skill_ids: list[str] | None = None,
) -> str:
    requested_ids = [skill_id] if skill_id is not None else (skill_ids or [])
    logger.info("get_skills called | skill_ids=%s", requested_ids)

    found: list[dict[str, str]] = []
    unknown: list[str] = []
    for requested_id in requested_ids:
        skill = skill_store.get(requested_id)
        if not skill:
            unknown.append(requested_id)
            continue
        found.append({"skill_id": requested_id, "content": skill.body})

    payload: dict[str, object] = {"skills": found}
    if unknown:
        payload["errors"] = [{"skill_id": sid, "error": f"Unknown skill_id: {sid}"} for sid in unknown]
    return json.dumps(payload)


@mcp.tool(
    name="get_skills",
    description="Return full SKILL.md content for one or more skill_ids. Prefer skill_ids to fetch multiple skills in one call.",
)
def get_skills_mcp(skill_id: str | None = None, skill_ids: list[str] | None = None) -> str:
    return get_skills(skill_store=_require_skill_store(), skill_id=skill_id, skill_ids=skill_ids)


def web_request(method: str, url: str, headers: dict[str, str] | None = None, body: str | None = None) -> str:
    logger.info("web_request called | method=%s | url=%s | headers=%s | body_preview=%s", method, url, headers or {}, (body or "")[:300])
    try:
        with httpx.Client(timeout=20, trust_env=False) as client:
            response = client.request(method=method.upper(), url=url, headers=headers or {}, content=body)
        truncated = response.text[:4000]
        payload = json.dumps(
            {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": truncated,
            }
        )
        logger.info("web_request result | status=%s | body_preview=%s", response.status_code, truncated[:300])
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.exception("web_request failed")
        return json.dumps({"error": str(exc)})


@mcp.tool(name="web_request", description="Perform an HTTP request and return status, headers, and truncated body.")
def web_request_mcp(
    method: str = "GET",
    url: str = "",
    headers: dict[str, str] | None = None,
    body: str | None = None,
) -> str:
    return web_request(method=method, url=url, headers=headers, body=body)


class AgentTools:
    def __init__(self, skill_store: SkillStore) -> None:
        self.skill_store = skill_store
        set_skill_store(skill_store)

    def langchain_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                name="get_skills",
                description="Return full SKILL.md content for one or more skill_ids. Prefer skill_ids to fetch multiple skills in one call.",
                args_schema=GetSkillsInput,
                func=partial(get_skills, skill_store=self.skill_store),
            ),
            StructuredTool.from_function(
                name="web_request",
                description="Perform an HTTP request and return status, headers, and truncated body.",
                args_schema=WebRequestInput,
                func=web_request,
            ),
        ]

    async def _dispatch_tool_async(self, name: str, args: dict) -> str:
        async with Client(mcp) as client:
            result = await client.call_tool(name, args, raise_on_error=False)
        if result.is_error:
            return json.dumps({"error": result.data})
        return str(result.data)

    def dispatch_tool(self, name: str, args: dict) -> str:
        logger.info("dispatch_tool called | name=%s | args=%s", name, json.dumps(args, ensure_ascii=False))
        try:
            result = asyncio.run(self._dispatch_tool_async(name, args))
        except Exception:  # noqa: BLE001
            logger.exception("dispatch_tool failed")
            return json.dumps({"error": f"Unknown tool {name}"})
        logger.info("dispatch_tool finished | name=%s | result_preview=%s", name, result[:500])
        return result
