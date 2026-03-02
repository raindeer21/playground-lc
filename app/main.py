from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, UTC
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from app.agent import AgentRuntime
from app.agent_google_adk import GoogleADKAgentRuntime
from app.runtime_base import BaseAgentRuntime


app = FastAPI(title="Skill-aware Agent API")
access_log_file = Path(__file__).resolve().parent / "access.log"
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)
root_logger.addHandler(stream_handler)

access_file_handler = logging.FileHandler(access_log_file, encoding="utf-8")
access_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)
root_logger.addHandler(access_file_handler)



def _build_runtime() -> BaseAgentRuntime:
    backend = os.getenv("AGENT_RUNTIME_BACKEND", "langchain").strip().lower()
    if backend == "google_adk":
        return GoogleADKAgentRuntime()
    return AgentRuntime()


runtime: BaseAgentRuntime = _build_runtime()
logger = logging.getLogger(__name__)

agent_chat_logger = logging.getLogger("agent_chat")
if not agent_chat_logger.handlers:
    log_file = Path(__file__).resolve().parent / "agent_chat.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    agent_chat_logger.addHandler(file_handler)
    agent_chat_logger.addHandler(terminal_handler)
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


def _trim_old_tool_history(
    history: list[dict[str, Any]],
    max_recent_user_requests: int = 2,
) -> list[dict[str, Any]]:
    if max_recent_user_requests <= 0:
        return [
            item
            for item in history
            if item.get("role") not in {"tool"}
            and not (item.get("role") == "assistant" and item.get("tool_calls"))
        ]

    total_user_requests = sum(1 for item in history if item.get("role") == "user")
    keep_from_user_index = max(1, total_user_requests - max_recent_user_requests + 1)

    trimmed: list[dict[str, Any]] = []
    current_user_index = 0
    for item in history:
        role = item.get("role")
        if role == "user":
            current_user_index += 1
            trimmed.append(item)
            continue

        is_tool_entry = role == "tool" or (role == "assistant" and item.get("tool_calls"))
        if is_tool_entry and current_user_index < keep_from_user_index:
            continue
        trimmed.append(item)
    return trimmed


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


def _parse_property_response(content: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    message = payload.get("message")
    houses = payload.get("houses")
    if not isinstance(message, str) or not isinstance(houses, list):
        return None
    return {"message": message, "houses": houses}


def _default_token_usage() -> dict[str, Any]:
    return {
        "totals": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "per_step": [],
    }


def _build_request_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"{prefix}-{timestamp}-{uuid4().hex[:12]}"




def _log_agent_chat_token_usage(request_id: str, session_id: str, token_usage: dict[str, Any]) -> None:
    totals = token_usage.get("totals", {}) if isinstance(token_usage, dict) else {}
    per_step = token_usage.get("per_step", []) if isinstance(token_usage, dict) else []

    llm_calls = len(per_step) if isinstance(per_step, list) else 0
    tool_call_steps = 0
    final_response_steps = 0
    if isinstance(per_step, list):
        tool_call_steps = sum(1 for step in per_step if isinstance(step, dict) and step.get("stage") == "tool_call")
        final_response_steps = sum(1 for step in per_step if isinstance(step, dict) and step.get("stage") == "final_response")

    total_tokens = totals.get("total_tokens", 0) if isinstance(totals, dict) else 0
    avg_tokens_per_call = int(total_tokens / llm_calls) if llm_calls else 0

    agent_chat_logger.info(
        "token_usage_insights | request_id=%s | session_id=%s | prompt_tokens=%s | completion_tokens=%s | total_tokens=%s | llm_calls=%s | tool_call_steps=%s | final_response_steps=%s | avg_tokens_per_call=%s",
        request_id,
        session_id,
        totals.get("prompt_tokens", 0) if isinstance(totals, dict) else 0,
        totals.get("completion_tokens", 0) if isinstance(totals, dict) else 0,
        total_tokens,
        llm_calls,
        tool_call_steps,
        final_response_steps,
        avg_tokens_per_call,
    )

def _build_history_entries(result: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for step in result.get("compressed_steps", []):
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
    request_id = _build_request_id("chatcmpl")
    logger.info(
        "Incoming /v1/chat/completions request | request_id=%s | message_count=%s",
        request_id,
        len(request.messages),
    )
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
        "token_usage": result.get("token_usage", _default_token_usage()),
    }


@app.post("/api/v1/chat")
async def agent_chat(request: AgentChatRequest):
    start = time.perf_counter()
    request_id = _build_request_id("agentchat")
    logger.info(
        "Incoming /api/v1/chat request | request_id=%s | session_id=%s | base_url=%s",
        request_id,
        request.session_id,
        request.model_ip,
    )
    agent_chat_logger.info(
        "request_id=%s | request=%s",
        request_id,
        json.dumps(request.model_dump(), ensure_ascii=False),
    )

    with _session_lock:
        history = list(_session_histories[request.session_id])
    history.append({"role": "user", "content": request.message})
    history = _trim_old_tool_history(history)

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
            "token_usage": _default_token_usage(),
        }
        _log_agent_chat_token_usage(request_id, request.session_id, error_payload["token_usage"])
        agent_chat_logger.info("request_id=%s | response=%s", request_id, json.dumps(error_payload, ensure_ascii=False))
        return error_payload

    with _session_lock:
        _session_histories[request.session_id] = _trim_old_tool_history(
            [*history, *_build_history_entries(result)]
        )

    tool_results = _extract_tool_results(result.get("steps", []))
    response_payload = {
        "session_id": request.session_id,
        "response": result["message"],
        "status": "success",
        "tool_results": tool_results,
        "timestamp": timestamp,
        "duration_ms": duration_ms,
        "token_usage": result.get("token_usage", _default_token_usage()),
    }
    property_result = _parse_property_response(result.get("message", ""))
    if property_result is not None:
        if property_result["houses"]:
            response_payload["response"] = json.dumps(property_result, ensure_ascii=False)
            response_payload["message"] = property_result["message"]
            response_payload["houses"] = property_result["houses"]
        else:
            response_payload["response"] = property_result["message"]

    _log_agent_chat_token_usage(request_id, request.session_id, response_payload["token_usage"])
    logger.info("Final Response | request_id=%s | payload=%s", request_id, response_payload)
    agent_chat_logger.info("request_id=%s | response=%s", request_id, json.dumps(response_payload, ensure_ascii=False))
    return response_payload


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
