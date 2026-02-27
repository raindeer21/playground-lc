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

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
uvicorn app.main:app --reload
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
