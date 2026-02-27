from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, create_model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenAPIOperation:
    operation_id: str
    method: str
    path: str
    base_url: str
    description: str
    args_schema: type[BaseModel]


class OpenAPIToolRegistry:
    def __init__(self, spec: dict[str, Any], default_headers: dict[str, str] | None = None) -> None:
        self.spec = spec
        self.default_headers = default_headers or {}
        self.operations = self._parse_spec(spec)

    @classmethod
    def from_json(cls, raw_spec: str, default_headers: dict[str, str] | None = None) -> "OpenAPIToolRegistry":
        return cls(spec=json.loads(raw_spec), default_headers=default_headers)

    @classmethod
    def from_env(cls) -> "OpenAPIToolRegistry | None":
        raw = os.getenv("OPENAPI_SPEC_JSON")
        if not raw:
            return None
        return cls.from_json(raw)

    def _parse_spec(self, spec: dict[str, Any]) -> dict[str, OpenAPIOperation]:
        servers = spec.get("servers", [])
        base_url = servers[0]["url"].rstrip("/") if servers else ""
        operations: dict[str, OpenAPIOperation] = {}

        for path, methods in spec.get("paths", {}).items():
            for method, op in methods.items():
                operation_id = op.get("operationId")
                if not operation_id:
                    continue
                args_model = self._build_args_schema(operation_id, op.get("parameters", []))
                operations[operation_id] = OpenAPIOperation(
                    operation_id=operation_id,
                    method=method.upper(),
                    path=path,
                    base_url=base_url,
                    description=op.get("description") or op.get("summary") or operation_id,
                    args_schema=args_model,
                )
        return operations

    def _build_args_schema(self, operation_id: str, parameters: list[dict[str, Any]]) -> type[BaseModel]:
        fields: dict[str, tuple[Any, Any]] = {}
        for parameter in parameters:
            name = parameter["name"]
            required = parameter.get("required", False)
            schema = parameter.get("schema", {})
            py_type = _json_type_to_python(schema.get("type", "string"))
            description = schema.get("description") or parameter.get("description") or ""
            default = ... if required else None
            annotation = py_type if required else (py_type | None)
            fields[name] = (annotation, Field(default=default, description=description))

        fields["x_user_id"] = (
            str | None,
            Field(default=None, description="Optional X-User-ID header override."),
        )

        model = create_model(  # type: ignore[call-overload]
            f"{_sanitize_name(operation_id)}Input",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        return model

    def call_operation(self, operation_id: str, **kwargs: Any) -> str:
        operation = self.operations[operation_id]
        path = operation.path

        for token in re.findall(r"{([^}]+)}", path):
            if token not in kwargs:
                raise ValueError(f"Missing required path parameter: {token}")
            value = kwargs.pop(token)
            path = path.replace("{" + token + "}", str(value))

        query = {k: v for k, v in kwargs.items() if v is not None and k != "x_user_id"}

        headers = dict(self.default_headers)
        user_id = kwargs.get("x_user_id") or headers.get("X-User-ID") or os.getenv("AGENT_USER_ID")
        if user_id:
            headers["X-User-ID"] = user_id

        url = f"{operation.base_url}{path}"

        with httpx.Client(timeout=20, trust_env=False) as client:
            response = client.request(method=operation.method, url=url, params=query, headers=headers)

        return json.dumps(
            {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text[:4000],
            },
            ensure_ascii=False,
        )


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


def _json_type_to_python(json_type: str) -> type[Any]:
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    return mapping.get(json_type, str)
