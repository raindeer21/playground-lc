from __future__ import annotations

import json
from functools import partial
from typing import Callable

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.skills import SkillStore


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
    skill = skill_store.get(skill_id)
    if not skill:
        return json.dumps({"error": f"Unknown skill_id: {skill_id}"})
    return json.dumps({"skill_id": skill_id, "content": skill.body})


def web_request(method: str, url: str, headers: dict[str, str] | None = None, body: str | None = None) -> str:
    try:
        response = requests.request(method=method.upper(), url=url, headers=headers or {}, data=body, timeout=20)
        truncated = response.text[:4000]
        return json.dumps(
            {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": truncated,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})


def respond_to_user(message: str) -> str:
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
        dispatch_map: dict[str, Callable[..., str]] = {
            "get_skills": lambda **kwargs: get_skills(skill_store=self.skill_store, **kwargs),
            "web_request": web_request,
            "respond_to_user": respond_to_user,
        }
        handler = dispatch_map.get(name)
        if not handler:
            return json.dumps({"error": f"Unknown tool {name}"})
        return handler(**args)
