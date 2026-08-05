from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.graph.build_graph import build_graph
from src.storage.session_store import (
    active_session_count,
    add_message,
    clear_session,
    format_chat_for_prompt,
    get_chat_history,
    get_latest_report,
    get_reports,
    save_report,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_session():
    clear_session("test_session")
    clear_session("concurrent_session")
    clear_session("empty_session")
    yield
    clear_session("test_session")
    clear_session("concurrent_session")
    clear_session("empty_session")


def test_save_multiple_reports():
    save_report("test_session", {"text": "one"})
    save_report("test_session", {"text": "two"})
    latest = get_latest_report("test_session")
    assert latest is not None
    assert latest["text"] == "two"


def test_get_reports_all_in_order():
    save_report("test_session", {"text": "one"})
    save_report("test_session", {"text": "two"})
    reports = get_reports("test_session")
    assert len(reports) == 2
    assert reports[0]["text"] == "one"
    assert reports[1]["text"] == "two"


def test_clear_empty_session():
    assert clear_session("empty_session") == 0


def test_clear_session_twice():
    save_report("test_session", {"text": "one"})
    assert clear_session("test_session") == 1
    assert clear_session("test_session") == 0


def test_reports_maintain_insertion_order():
    for i in range(10):
        save_report("test_session", {"val": i})
    reports = get_reports("test_session")
    assert len(reports) == 10
    for i in range(10):
        assert reports[i]["val"] == i


def test_report_has_created_at():
    report = save_report("test_session", {"text": "one"})
    assert "created_at" in report


def test_get_latest_non_existent():
    assert get_latest_report("non_existent") is None


def test_concurrent_saves():
    def save(i):
        save_report("concurrent_session", {"val": i})

    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(20):
            executor.submit(save, i)
    
    reports = get_reports("concurrent_session")
    assert len(reports) == 20


# ---------------------------------------------------------------------------
# Dynamic UUID Session Deletion & Stress Tests
# ---------------------------------------------------------------------------

def test_dynamic_uuid_creation_and_full_deletion():
    """Stress test creating 50 unique dynamic UUID sessions and deleting them."""
    session_ids = [uuid.uuid4().hex for _ in range(50)]

    for sid in session_ids:
        save_report(sid, {"topic": "Tesla", "ticker": "TSLA"})
        add_message(sid, "user", "Research TSLA")
        add_message(sid, "assistant", "TSLA report ready.")

    # Verify all 50 sessions exist and have data
    for sid in session_ids:
        assert len(get_reports(sid)) == 1
        assert len(get_chat_history(sid)) == 2

    # Delete first 25 sessions
    to_delete = session_ids[:25]
    to_keep = session_ids[25:]

    for sid in to_delete:
        removed = clear_session(sid)
        assert removed == 1
        # Verify completely wiped
        assert get_reports(sid) == []
        assert get_chat_history(sid) == []
        assert get_latest_report(sid) is None
        assert format_chat_for_prompt(sid) == ""

    # Verify remaining 25 are 100% intact and untouched
    for sid in to_keep:
        assert len(get_reports(sid)) == 1
        assert len(get_chat_history(sid)) == 2

    # Cleanup remaining
    for sid in to_keep:
        clear_session(sid)


def test_session_deletion_with_whitespace_sanitization():
    raw_sid = uuid.uuid4().hex
    padded_sid = f"  {raw_sid}  \n"

    save_report(padded_sid, {"topic": "Apple", "ticker": "AAPL"})
    add_message(padded_sid, "user", "Apple research")

    # Fetch using clean SID
    assert len(get_reports(raw_sid)) == 1
    assert len(get_chat_history(raw_sid)) == 1

    # Delete using padded SID
    removed = clear_session(padded_sid)
    assert removed == 1
    assert get_reports(raw_sid) == []
    assert get_chat_history(raw_sid) == []


def test_api_delete_session_endpoint():
    """Verify DELETE /session/{session_id} HTTP endpoint."""
    sid = uuid.uuid4().hex
    save_report(sid, {"topic": "NVDA", "ticker": "NVDA"})
    add_message(sid, "user", "Analyze NVDA")

    resp = client.delete(f"/session/{sid}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    assert resp.json()["removed_reports"] == 1

    # Verify records in store are gone
    assert get_reports(sid) == []
    assert get_chat_history(sid) == []


def test_api_post_session_end_endpoint():
    """Verify POST /session/end HTTP endpoint."""
    sid = uuid.uuid4().hex
    save_report(sid, {"topic": "MSFT", "ticker": "MSFT"})
    add_message(sid, "user", "Analyze MSFT")

    resp = client.post("/session/end", json={"session_id": sid})
    assert resp.status_code == 200
    assert resp.json()["removed_reports"] == 1

    assert get_reports(sid) == []
    assert get_chat_history(sid) == []


def test_recall_after_deletion_returns_fresh_clarify():
    """Verify LangGraph recall query after deletion has no prior memory."""
    sid = uuid.uuid4().hex
    graph = build_graph()

    # 1. First run a research
    save_report(sid, {"topic": "Amazon", "ticker": "AMZN", "final_report": "AMZN is bullish."})
    add_message(sid, "user", "Research AMZN")
    add_message(sid, "assistant", "AMZN is bullish.")

    # 2. Delete session
    clear_session(sid)

    # 3. Ask a recall question on the deleted session
    result = graph.invoke({
        "user_query": "Why did you say that about AMZN?",
        "session_id": sid,
        "use_llm": False,
        "use_live_data": False,
    })

    # Since session was wiped, intent routes to clarify because no active report exists in session
    assert result["intent"] == "clarify"
    assert "no previous" in result["response"].lower() or "first" in result["response"].lower() or "no stock" in result["response"].lower()
