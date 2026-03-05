from __future__ import annotations

import ast
import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def trim_ai_message_for_history(ai_message: AIMessage) -> AIMessage:
    response_metadata = getattr(ai_message, "response_metadata", None)
    trimmed_response_metadata: dict[str, Any] = {}
    if isinstance(response_metadata, dict):
        finish_reason = response_metadata.get("finish_reason")
        if finish_reason is not None:
            trimmed_response_metadata["finish_reason"] = finish_reason

    kwargs: dict[str, Any] = {
        "content": ai_message.content,
        "tool_calls": ai_message.tool_calls,
    }
    if trimmed_response_metadata:
        kwargs["response_metadata"] = trimmed_response_metadata

    return AIMessage(**kwargs)


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def extract_token_usage(ai_message: AIMessage) -> dict[str, int]:
    usage: dict[str, Any] = {}

    response_metadata = getattr(ai_message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            usage = token_usage

    if not usage:
        usage_metadata = getattr(ai_message, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            usage = {
                "prompt_tokens": usage_metadata.get("input_tokens"),
                "completion_tokens": usage_metadata.get("output_tokens"),
                "total_tokens": usage_metadata.get("total_tokens"),
            }

    prompt_tokens = _safe_int(usage.get("prompt_tokens"))
    completion_tokens = _safe_int(usage.get("completion_tokens"))
    total_tokens = _safe_int(usage.get("total_tokens"))

    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def analyze_token_usage(history: list[BaseMessage]) -> dict[str, Any]:
    per_step: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for msg in history:
        if not isinstance(msg, AIMessage):
            continue

        usage = extract_token_usage(msg)
        if usage["total_tokens"] <= 0:
            continue

        stage = "tool_call" if msg.tool_calls else "final_response"
        per_step.append({"stage": stage, **usage})
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_tokens += usage["total_tokens"]

    llm_calls = len(per_step)
    tool_call_steps = sum(1 for item in per_step if item["stage"] == "tool_call")
    final_response_steps = llm_calls - tool_call_steps
    avg_tokens_per_call = int(total_tokens / llm_calls) if llm_calls else 0

    return {
        "totals": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        },
        "analysis": {
            "llm_calls": llm_calls,
            "tool_call_steps": tool_call_steps,
            "final_response_steps": final_response_steps,
            "avg_tokens_per_call": avg_tokens_per_call,
        },
        "per_step": per_step,
    }


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
                    "name": msg.name if msg.name else "",
                }
            )
    return steps
