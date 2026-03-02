from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.agent import AgentRuntime
from app.runtime_base import BaseAgentRuntime


class GoogleADKAgentRuntime(BaseAgentRuntime):
    """Google ADK backend implementation.

    This runtime currently falls back to the existing AgentRuntime behavior so it
    conforms to the shared runtime interface while keeping external API behavior
    stable.
    """

    def __init__(
        self,
        model: str = "qwen3-32b",
        skills_dir: str = "skills",
        conversation_log_path: str | Path = "logs/agent_conversations.jsonl",
    ) -> None:
        super().__init__(conversation_log_path=conversation_log_path)
        self._logger = logging.getLogger(__name__)
        self._delegate = AgentRuntime(
            model=model,
            skills_dir=skills_dir,
            conversation_log_path=conversation_log_path,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_steps: int = 12,
        model: str | None = None,
        session_id: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        self._logger.info("GoogleADK runtime delegating chat call to compatibility runtime")
        return await self._delegate.chat(
            messages=messages,
            max_steps=max_steps,
            model=model,
            session_id=session_id,
            base_url=base_url,
        )
