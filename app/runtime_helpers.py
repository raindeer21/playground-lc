from __future__ import annotations

import ast
import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def parse_tool_payload(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            candidate = parser(content)
            if isinstance(candidate, dict):
                return candidate
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
    return None


def compress_get_houses_nearby_result(content: Any) -> Any:
    payload = parse_tool_payload(content)
    if payload is None:
        return content

    data = payload.get("data")
    if not isinstance(data, dict):
        return content

    landmark = data.get("landmark")
    items = data.get("items")
    if not isinstance(items, list):
        return content

    compressed_landmark: dict[str, Any] = {}
    if isinstance(landmark, dict):
        compressed_landmark = {
            "id": landmark.get("id"),
            "name": landmark.get("name"),
        }

    compressed_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compressed_items.append(
            {
                "house_id": item.get("house_id"),
                "status": item.get("status"),
                "distance_to_landmark": item.get("distance_to_landmark"),
                "walking_distance": item.get("walking_distance"),
                "walking_duration": item.get("walking_duration"),
                "listing_platform": item.get("listing_platform"),
            }
        )

    payload["data"] = {
        "landmark": compressed_landmark,
        "items": compressed_items,
    }

    return json.dumps(payload, ensure_ascii=False)


def compress_get_houses_by_platform_result(content: Any) -> Any:
    payload = parse_tool_payload(content)
    if payload is None:
        return content

    data = payload.get("data")
    if not isinstance(data, dict):
        return content

    items = data.get("items")
    if not isinstance(items, list):
        return content

    compressed_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compressed_items.append(
            {
                "house_id": item.get("house_id"),
                "status": item.get("status"),
            }
        )

    data["items"] = compressed_items
    payload["data"] = data
    return json.dumps(payload, ensure_ascii=False)


def compress_get_houses_by_community_result(content: Any) -> Any:
    payload = parse_tool_payload(content)
    if payload is None:
        return content

    data = payload.get("data")
    if not isinstance(data, dict):
        return content

    items = data.get("items")
    if not isinstance(items, list):
        return content

    compressed_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compressed_items.append(
            {
                "house_id": item.get("house_id"),
                "community": item.get("community"),
                "listing_platform": item.get("listing_platform"),
                "status": item.get("status"),
            }
        )

    data["items"] = compressed_items
    payload["data"] = data
    return json.dumps(payload, ensure_ascii=False)


def compress_tool_result(tool_name: str, content: Any) -> Any:
    if tool_name == "get_houses_nearby":
        return compress_get_houses_nearby_result(content)
    if tool_name == "get_houses_by_platform":
        return compress_get_houses_by_platform_result(content)
    if tool_name == "get_houses_by_community":
        return compress_get_houses_by_community_result(content)
    return content


def serialize_steps(history: list[BaseMessage], compressed: bool = False) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for msg in history:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            steps.append({"type": "tool_calls", "tool_calls": msg.tool_calls})
        elif isinstance(msg, ToolMessage):
            content: Any = msg.content
            if compressed:
                content = compress_tool_result(msg.name or "", content)
            steps.append(
                {
                    "type": "tool_result",
                    "content": content,
                    "tool_call_id": msg.tool_call_id,
                    "status": msg.status,
                }
            )
    return steps
