# Skill-aware LangChain Agent

This project implements a LangChain-powered agent runtime with a **chat completion style API**.

## Features

- `POST /v1/chat/completions` endpoint similar to OpenAI-style request shape.
- Skills are discovered from `skills/*/SKILL.md`.
- The model is always prompted with **skill headers** (name/description/id).
- A `get_skills` tool provides **full SKILL.md content on demand**.
- The model is configured with `tool_choice="required"` to force function-calling behavior.
- `respond_to_user` tool is required to end a run.
- Conversations are logged to `logs/agent_conversations.jsonl` for auditing/debugging.
- Tools are implemented as standalone functions and composed via a dedicated `AgentTools` class.
- `web_request` tool supports controlled HTTP requests.
- Tool dispatch now runs through a local FastMCP server via in-memory transport (no external MCP process needed).
- OpenAPI tools are auto-imported at startup from `app/openapi_fake_app_agent_api.json` via `FastMCP.from_openapi(...)` (FastMCP 3 compatible).

## OpenAPI auto-load (FastMCP 3)

The runtime now loads MCP tools from the saved OpenAPI file automatically.

```python
child_server = FastMCP.from_openapi(
    openapi_spec,
    httpx.AsyncClient(base_url=server, timeout=20, trust_env=False),
)
await parent_server.import_server(server=child_server, prefix="")
```

> Note: In FastMCP 3, `import_server` supports `prefix` but not `tool_separator` / `prompt_separator` / `resource_separator`.

## Run

```bash
./run.sh
```

`run.sh` creates/uses `.venv`, installs `requirements.txt`, and starts the API server.

```bash
export OPENAI_API_KEY=...
HOST=0.0.0.0 PORT=8080 ./run.sh
```

## Example request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "skill-agent-v1",
    "messages": [
      {"role": "user", "content": "Read docs for LangChain and summarize key points"}
    ]
  }'
```

## Notes

- Add more skill folders under `skills/<skill-id>/SKILL.md`.
- `steps` in the response provides a transparent trace of tool calls/results.
