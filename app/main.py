from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from app.agent import AgentRuntime


app = FastAPI(title="Skill-aware Agent API")
runtime = AgentRuntime()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="skill-agent-v1")
    messages: list[ChatMessage]


class AgentChatRequest(BaseModel):
    model_ip: str
    session_id: str
    message: str


_session_histories: dict[str, list[dict[str, str]]] = defaultdict(list)
_session_lock = Lock()


def _extract_tool_results(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_results: list[dict[str, Any]] = []
    pending_names: list[str] = []
    for step in steps:
        if step.get("type") == "tool_calls":
            for call in step.get("tool_calls", []):
                pending_names.append(call.get("name", "unknown"))
        elif step.get("type") == "tool_result":
            output = str(step.get("content", ""))
            name = pending_names.pop(0) if pending_names else "unknown"
            tool_results.append(
                {
                    "name": name,
                    "success": '"error"' not in output,
                    "output": output,
                }
            )
    return tool_results


def _extract_property_result(tool_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(tool_results):
        if item.get("name") != "provide_property_result_list":
            continue
        try:
            parsed = item.get("output")
            if isinstance(parsed, str):
                payload = json.loads(parsed)
            elif isinstance(parsed, dict):
                payload = parsed
            else:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("houses", []), list):
                return payload
        except Exception:
            continue
    return None


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    logger.info("Incoming /v1/chat/completions request with %s message(s)", len(request.messages))
    result = runtime.chat([m.model_dump() for m in request.messages], model=request.model)

    if "error" in result:
        return {
            "id": "chatcmpl-error",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result["error"]}}],
        }

    return {
        "id": "chatcmpl-skill-agent",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["message"]},
                "finish_reason": "stop",
            }
        ],
        "steps": result["steps"],
    }


@app.post("/api/v1/chat")
def agent_chat(request: AgentChatRequest):
    start = time.perf_counter()
    logger.info("Incoming /api/v1/chat request | session_id=%s | model_ip=%s", request.session_id, request.model_ip)

    with _session_lock:
        history = list(_session_histories[request.session_id])
    history.append({"role": "user", "content": request.message})

    result = runtime.chat(history, model=request.model_ip, session_id=request.session_id)
    duration_ms = int((time.perf_counter() - start) * 1000)
    timestamp = int(time.time())

    if "error" in result:
        return {
            "session_id": request.session_id,
            "response": result["error"],
            "status": "error",
            "tool_results": [],
            "timestamp": timestamp,
            "duration_ms": duration_ms,
        }

    with _session_lock:
        _session_histories[request.session_id] = [
            *history,
            {"role": "assistant", "content": result["message"]},
        ]

    tool_results = _extract_tool_results(result.get("steps", []))
    response_payload = {
        "session_id": request.session_id,
        "response": result["message"],
        "status": "success",
        "tool_results": tool_results,
        "timestamp": timestamp,
        "duration_ms": duration_ms,
    }
    property_result = _extract_property_result(tool_results)
    if property_result is not None:
        response_payload["message"] = property_result.get("message", "为您找到以下符合条件的房源：")
        response_payload["houses"] = property_result.get("houses", [])

    return response_payload


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
