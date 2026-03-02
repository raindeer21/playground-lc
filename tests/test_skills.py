from app.skills import SkillStore


def test_skill_store_parses_allowed_tools_and_whitelist() -> None:
    store = SkillStore('skills')

    allowed = store.tool_whitelist_for(['rental-house-search', 'rental-house-actions'])

    assert allowed is not None
    assert 'web_request' in allowed
