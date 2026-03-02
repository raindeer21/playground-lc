from app.skills import SkillStore


def test_skill_store_parses_allowed_tools_and_whitelist() -> None:
    store = SkillStore('skills')

    allowed = store.tool_whitelist_for(['property_search', 'property_management'])

    assert allowed is not None
    assert 'rent_house' in allowed
    assert 'get_houses_by_platform' in allowed
