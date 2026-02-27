from app.agent import AgentRuntime


def test_system_prompt_includes_rental_requirement_policy() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)

    class _DummySkillStore:
        def headers(self):
            return [{"skill_id": "rental-house-search", "name": "search", "description": "..."}]

    runtime.skill_store = _DummySkillStore()

    system_message = runtime._system_message()
    content = system_message.content

    assert "extract explicit constraints" in content
    assert "ask one focused follow-up question" in content
    assert "verify and compare candidate listings" in content
    assert "cap final recommended candidates to at most 5 listings" in content


def test_convert_messages_maps_roles() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u"},
    ]

    converted = AgentRuntime._convert_messages(messages)

    assert [m.type for m in converted] == ["system", "ai", "human"]
