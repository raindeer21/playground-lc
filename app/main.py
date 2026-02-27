from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from app.agent import AgentRuntime


app = FastAPI(title="Skill-aware Agent API")
runtime = AgentRuntime()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="skill-agent-v1")
    messages: list[ChatMessage]


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
