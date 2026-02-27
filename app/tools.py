from __future__ import annotations

import json
import logging
from functools import partial
from typing import Callable

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.skills import SkillStore

logger = logging.getLogger(__name__)


class GetSkillsInput(BaseModel):
    skill_id: str = Field(..., description="The skill_id to load. Use one from skill headers.")


class RespondInput(BaseModel):
    message: str = Field(..., description="The final assistant response for the user.")


class WebRequestInput(BaseModel):
    method: str = Field(default="GET", description="HTTP method")
    url: str = Field(..., description="Target URL")
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = Field(default=None)


def get_skills(skill_id: str, skill_store: SkillStore) -> str:
    logger.info("get_skills called | skill_id=%s", skill_id)
    skill = skill_store.get(skill_id)
    if not skill:
        return json.dumps({"error": f"Unknown skill_id: {skill_id}"})
    return json.dumps({"skill_id": skill_id, "content": skill.body})


def web_request(method: str, url: str, headers: dict[str, str] | None = None, body: str | None = None) -> str:
    logger.info("web_request called | method=%s | url=%s | headers=%s | body_preview=%s", method, url, headers or {}, (body or "")[:300])
    try:
        response = requests.request(method=method.upper(), url=url, headers=headers or {}, data=body, timeout=20)
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


def respond_to_user(message: str) -> str:
    logger.info("respond_to_user called | message_preview=%s", message[:500])
    return json.dumps({"final": message})


class AgentTools:
    def __init__(self, skill_store: SkillStore) -> None:
        self.skill_store = skill_store

    def langchain_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                name="get_skills",
                description="Return full SKILL.md content for a given skill_id.",
                args_schema=GetSkillsInput,
                func=partial(get_skills, skill_store=self.skill_store),
            ),
            StructuredTool.from_function(
                name="web_request",
                description="Perform an HTTP request and return status, headers, and truncated body.",
                args_schema=WebRequestInput,
                func=web_request,
            ),
            StructuredTool.from_function(
                name="respond_to_user",
                description="End the session by returning the final response to the user.",
                args_schema=RespondInput,
                func=respond_to_user,
            ),
        ]

    def dispatch_tool(self, name: str, args: dict) -> str:
        logger.info("dispatch_tool called | name=%s | args=%s", name, json.dumps(args, ensure_ascii=False))
        dispatch_map: dict[str, Callable[..., str]] = {
            "get_skills": lambda **kwargs: get_skills(skill_store=self.skill_store, **kwargs),
            "web_request": web_request,
            "respond_to_user": respond_to_user,
        }
        handler = dispatch_map.get(name)
        if not handler:
            logger.error("Unknown tool requested: %s", name)
            return json.dumps({"error": f"Unknown tool {name}"})
        result = handler(**args)
        logger.info("dispatch_tool finished | name=%s | result_preview=%s", name, result[:500])
        return result
