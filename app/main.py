from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from app.agent import AgentRuntime


app = FastAPI(title="Skill-aware Agent API")
access_log_file = Path(__file__).resolve().parent / "access.log"
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

access_file_handler = logging.FileHandler(access_log_file, encoding="utf-8")
access_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)
root_logger.addHandler(access_file_handler)

runtime = AgentRuntime()
logger = logging.getLogger(__name__)

agent_chat_logger = logging.getLogger("agent_chat")
if not agent_chat_logger.handlers:
    log_file = Path(__file__).resolve().parent / "agent_chat.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    agent_chat_logger.addHandler(file_handler)
agent_chat_logger.setLevel(logging.INFO)
agent_chat_logger.propagate = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="qwen3-32b")
    messages: list[ChatMessage]


class AgentChatRequest(BaseModel):
    model_ip: str | None = None
    session_id: str
    message: str


_session_histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
        if item.get("name") != "current_properties":
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


def _build_history_entries(result: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for step in result.get("steps", []):
        step_type = step.get("type")
        if step_type == "tool_calls":
            entries.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": step.get("tool_calls", []),
                }
            )
        elif step_type == "tool_result":
            entries.append(
                {
                    "role": "tool",
                    "content": str(step.get("content", "")),
                    "tool_call_id": step.get("tool_call_id", ""),
                }
            )
    entries.append({"role": "assistant", "content": result.get("message", "")})
    return entries


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    logger.info("Incoming /v1/chat/completions request with %s message(s)", len(request.messages))
    result = await runtime.chat([m.model_dump() for m in request.messages], model=request.model)

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
                "message": {"role": "assistant", "content": result["message"], },
                "finish_reason": "stop",
            }
        ],
        "steps": result["steps"],
    }


@app.post("/api/v1/chat")
async def agent_chat(request: AgentChatRequest):
    start = time.perf_counter()
    logger.info(
        "Incoming /api/v1/chat request | session_id=%s | base_url=%s",
        request.session_id,
        request.model_ip,
    )
    agent_chat_logger.info(
        "request=%s",
        json.dumps(request.model_dump(), ensure_ascii=False),
    )

    with _session_lock:
        history = list(_session_histories[request.session_id])
    history.append({"role": "user", "content": request.message})

    if request.model_ip:
        base_url = f"http://{request.model_ip}:8888/v1"
    else:
        base_url = "http://api.openai.rnd.huawei.com/v1"

    result = await runtime.chat(history, session_id=request.session_id, base_url=base_url)
    duration_ms = int((time.perf_counter() - start) * 1000)
    timestamp = int(time.time())

    if "error" in result:
        error_payload = {
            "session_id": request.session_id,
            "response": result["error"],
            "status": "error",
            "tool_results": [],
            "timestamp": timestamp,
            "duration_ms": duration_ms,
        }
        agent_chat_logger.info("response=%s", json.dumps(error_payload, ensure_ascii=False))
        return error_payload

    with _session_lock:
        _session_histories[request.session_id] = [*history, *_build_history_entries(result)]

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
        response_payload["response"] = json.dumps(property_result, ensure_ascii=False)

    logger.info(f"Final Response | {response_payload}")
    agent_chat_logger.info("response=%s", json.dumps(response_payload, ensure_ascii=False))
    return response_payload


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
