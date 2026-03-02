from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from app.runtime_helpers import serialize_steps


class BaseAgentRuntime(ABC):
    def __init__(self, conversation_log_path: str | Path = "logs/agent_conversations.jsonl") -> None:
        self.conversation_log_path = Path(conversation_log_path)
        self._logger = logging.getLogger(self.__class__.__module__)

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_steps: int = 12,
        model: str | None = None,
        session_id: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        ...

    def _log_conversation(self, messages: list[dict[str, Any]], response: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "messages": messages,
            "response": response,
        }
        try:
            self.conversation_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.conversation_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            self._logger.exception("Failed to write conversation log")

    @staticmethod
    def _serialize_steps(history: list[BaseMessage], compressed: bool = False) -> list[dict[str, Any]]:
        return serialize_steps(history, compressed=compressed)
