from src.llm.client import rule_based_generate
from src.storage.session_store import clear_session, get_latest_report, save_report


def test_session_store_saves_and_clears_report():
    clear_session("test")
    save_report("test", {"topic": "Nvidia", "ticker": "NVDA", "final_report": "ok"})

    latest = get_latest_report("test")

    assert latest is not None
    assert latest["ticker"] == "NVDA"
    assert "created_at" in latest
    assert clear_session("test") == 1
    assert get_latest_report("test") is None


def test_rule_based_router_fallback_can_recall():
    text = rule_based_generate("why did you say it was divergent?", task="router")

    assert "recall" in text
