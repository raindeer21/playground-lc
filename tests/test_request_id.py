import re


def test_build_request_id_is_unique_and_searchable() -> None:
    from app.main import _build_request_id

    request_id_one = _build_request_id("agentchat")
    request_id_two = _build_request_id("agentchat")

    assert request_id_one != request_id_two
    assert request_id_one.startswith("agentchat-")
    assert re.fullmatch(r"agentchat-\d{8}T\d{12}-[0-9a-f]{12}", request_id_one)
