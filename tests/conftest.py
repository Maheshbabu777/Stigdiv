"""Shared pytest fixtures for stigdiv tests."""

from __future__ import annotations

import pytest

from src.graph.build_graph import build_graph
from src.storage.session_store import clear_session


@pytest.fixture()
def graph_app():
    """Build the LangGraph application once per test."""
    return build_graph()


@pytest.fixture()
def clean_session():
    """Return a helper that auto-clears a session before and after a test."""
    sessions: list[str] = []

    def _make(name: str = "fixture-test") -> str:
        clear_session(name)
        sessions.append(name)
        return name

    yield _make

    for sid in sessions:
        clear_session(sid)
