from __future__ import annotations

import logging

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any
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


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    logger.info("Incoming /v1/chat/completions request with %s message(s)", len(request.messages))
    result = runtime.chat([m.model_dump() for m in request.messages])

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


class OpenAPILoadRequest(BaseModel):
    openapi: dict[str, Any]


@app.post("/v1/openapi/load")
def load_openapi_tools(request: OpenAPILoadRequest):
    operation_ids = runtime.load_openapi_spec(request.openapi)
    return {
        "object": "openapi.tool_load",
        "operation_count": len(operation_ids),
        "operations": operation_ids,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
