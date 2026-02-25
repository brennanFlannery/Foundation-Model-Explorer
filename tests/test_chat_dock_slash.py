from __future__ import annotations

from chat import ChatDockWidget


def test_extract_slash_prefix_returns_prefix_while_typing_command():
    assert ChatDockWidget._extract_slash_prefix("/d") == "d"
    assert ChatDockWidget._extract_slash_prefix("   /mo") == "mo"
    assert ChatDockWidget._extract_slash_prefix("/") == ""


def test_extract_slash_prefix_hides_after_space_or_non_slash():
    assert ChatDockWidget._extract_slash_prefix("/data ") is None
    assert ChatDockWidget._extract_slash_prefix("hello") is None
    assert ChatDockWidget._extract_slash_prefix("") is None


def test_get_slash_matches_filters_by_startswith():
    catalog = [
        {"command": "data", "display": "/data", "description": "d"},
        {"command": "describe", "display": "/describe", "description": "d2"},
        {"command": "models", "display": "/models", "description": "m"},
    ]
    matches = ChatDockWidget._get_slash_matches("d", catalog)
    assert [m["command"] for m in matches] == ["data", "describe"]

    matches_empty_prefix = ChatDockWidget._get_slash_matches("", catalog)
    assert [m["command"] for m in matches_empty_prefix] == ["data", "describe", "models"]
