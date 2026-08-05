from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.agents.common import KNOWN_UNLISTED_COMPANIES, resolve_stock_identity
from src.agents.news_agent import fetch_news
from src.api.main import app
from src.config import Settings
from src.graph.build_graph import build_graph
from src.storage.session_store import (
    add_message,
    clear_session,
    compress_chat_history,
    format_chat_for_prompt,
    get_chat_history,
    needs_compression,
)
from src.tools.firecrawl_client import FirecrawlClient, get_firecrawl

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Bot Identity and Self-Introduction Tests
# ---------------------------------------------------------------------------

def test_bot_identity_in_general_chat():
    session_id = "test_bot_identity"
    clear_session(session_id)
    graph = build_graph()
    result = graph.invoke({
        "user_query": "who are you and what do you do?",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert result["intent"] == "general_chat"
    resp = result["response"]
    assert "Signal Divergence Agent" in resp
    assert "market" in resp.lower()
    assert "news" in resp.lower()
    assert "social" in resp.lower()


def test_bot_greeting_contains_capabilities():
    session_id = "test_bot_greeting"
    clear_session(session_id)
    graph = build_graph()
    result = graph.invoke({
        "user_query": "hello there",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert result["intent"] == "general_chat"
    assert "Signal Divergence Agent" in result["response"]


# ---------------------------------------------------------------------------
# 2. Deep Company Search & Unlisted/Private Companies
# ---------------------------------------------------------------------------

def test_unlisted_companies_detected():
    for name in ["OpenAI", "SpaceX", "Stripe", "ByteDance", "Anthropic", "Databricks"]:
        resolved = resolve_stock_identity(name)
        assert resolved["ticker"] is None
        assert resolved["is_listed"] is False
        assert resolved["note"] is not None
        assert "privately held" in resolved["note"].lower() or "not publicly" in resolved["note"].lower()


def test_unlisted_company_graph_routing():
    session_id = "test_unlisted_graph"
    clear_session(session_id)
    graph = build_graph()
    result = graph.invoke({
        "user_query": "research OpenAI stock",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert result["intent"] == "clarify"
    assert result["ticker"] is None
    assert "privately held" in result["response"].lower() or "not publicly" in result["response"].lower()


def test_nonexistent_company_graph_routing():
    session_id = "test_nonexistent_graph"
    clear_session(session_id)
    graph = build_graph()
    result = graph.invoke({
        "user_query": "research xyzrandomfakecorp999",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert result["intent"] == "clarify"
    assert result["ticker"] is None
    resp = result["response"].lower()
    assert (
        "no publicly traded stock" in resp
        or "could not" in resp
        or "not found" in resp
        or "unlisted" in resp
        or "privately held" in resp
        or "not listed" in resp
    )


# ---------------------------------------------------------------------------
# 3. Chat History, 15-20 Message Recall, Compression & Isolation
# ---------------------------------------------------------------------------

def test_chat_history_recording():
    session_id = "test_chat_record"
    clear_session(session_id)
    add_message(session_id, "user", "hi")
    add_message(session_id, "assistant", "Hello! I am the Signal Divergence Agent.")
    history = get_chat_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_chat_history_compression_trigger():
    session_id = "test_chat_compress"
    clear_session(session_id)
    # Add 22 messages (exceeding MAX_HISTORY=20)
    for i in range(22):
        add_message(session_id, "user" if i % 2 == 0 else "assistant", f"Message {i}")

    assert needs_compression(session_id) is True
    compress_chat_history(session_id, "Summary of turns 0 to 6.")
    
    history = get_chat_history(session_id)
    # Compressed history should be 1 system summary + 15 recent messages = 16 messages
    assert len(history) == 16
    assert history[0]["role"] == "system"
    assert "Compressed context" in history[0]["content"]
    assert history[-1]["content"] == "Message 21"


def test_format_chat_for_prompt():
    session_id = "test_prompt_format"
    clear_session(session_id)
    add_message(session_id, "user", "Research TSLA")
    add_message(session_id, "assistant", "TSLA report generated.")
    formatted = format_chat_for_prompt(session_id)
    assert "User: Research TSLA" in formatted
    assert "Assistant: TSLA report generated." in formatted


def test_session_isolation():
    session_a = "user_alice"
    session_b = "user_bob"
    clear_session(session_a)
    clear_session(session_b)

    add_message(session_a, "user", "Alice query on AAPL")
    add_message(session_b, "user", "Bob query on MSFT")

    assert len(get_chat_history(session_a)) == 1
    assert "Alice" in get_chat_history(session_a)[0]["content"]
    assert "Bob" not in get_chat_history(session_a)[0]["content"]

    assert len(get_chat_history(session_b)) == 1
    assert "Bob" in get_chat_history(session_b)[0]["content"]
    assert "Alice" not in get_chat_history(session_b)[0]["content"]

    clear_session(session_a)
    assert len(get_chat_history(session_a)) == 0
    assert len(get_chat_history(session_b)) == 1


# ---------------------------------------------------------------------------
# 4. Async API and Chat Endpoints
# ---------------------------------------------------------------------------

def test_api_chat_history_endpoint():
    session_id = "api_test_session"
    clear_session(session_id)
    
    # Send a message to chat endpoint
    resp = client.post("/chat", json={"message": "hello", "session_id": session_id, "use_llm": False})
    assert resp.status_code == 200

    # Inspect chat history endpoint
    chat_resp = client.get(f"/session/{session_id}/chat")
    assert chat_resp.status_code == 200
    data = chat_resp.json()
    assert data["session_id"] == session_id
    assert len(data["history"]) >= 2  # user + assistant

    # End session and verify history is deleted
    end_resp = client.post("/session/end", json={"session_id": session_id})
    assert end_resp.status_code == 200

    chat_resp_after = client.get(f"/session/{session_id}/chat")
    assert len(chat_resp_after.json()["history"]) == 0


# ---------------------------------------------------------------------------
# 5. On-Demand Firecrawl Client Tests
# ---------------------------------------------------------------------------

def test_firecrawl_unavailable_graceful():
    client = FirecrawlClient(
        settings=Settings(
            groq_api_key=None,
            gemini_api_key=None,
            openrouter_api_key=None,
            firecrawl_api_key=None,
            provider_order=("groq",),
            groq_model="llama",
            gemini_model="gemini",
            openrouter_model="openrouter",
            request_timeout_sec=5,
        )
    )
    assert client.is_available is False
    assert client.scrape_url("https://example.com") is None
    assert client.search("test query") == []


def test_firecrawl_mock_search():
    client = FirecrawlClient(
        settings=Settings(
            groq_api_key=None,
            gemini_api_key=None,
            openrouter_api_key=None,
            firecrawl_api_key="fc_test_key",
            provider_order=("groq",),
            groq_model="llama",
            gemini_model="gemini",
            openrouter_model="openrouter",
            request_timeout_sec=5,
        )
    )
    assert client.is_available is True

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"title": "Palantir Q4 Earnings", "url": "https://example.com/pltr", "markdown": "Strong revenue growth"},
        ]
    }

    with patch("requests.post", return_value=mock_resp):
        results = client.search("Palantir stock news", limit=1)
        assert len(results) == 1
        assert results[0]["title"] == "Palantir Q4 Earnings"
        assert results[0]["url"] == "https://example.com/pltr"


def test_firecrawl_mock_scrape():
    client = FirecrawlClient(
        settings=Settings(
            groq_api_key=None,
            gemini_api_key=None,
            openrouter_api_key=None,
            firecrawl_api_key="fc_test_key",
            provider_order=("groq",),
            groq_model="llama",
            gemini_model="gemini",
            openrouter_model="openrouter",
            request_timeout_sec=5,
        )
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {"markdown": "# Company Overview\nPublic company on NYSE.", "metadata": {"title": "Company"}}
    }

    with patch("requests.post", return_value=mock_resp):
        res = client.scrape_url("https://example.com")
        assert res is not None
        assert "Company Overview" in res["markdown"]


def test_fetch_news_uses_firecrawl_fallback_when_rss_empty():
    mock_feed = MagicMock()
    mock_feed.parse.return_value = MagicMock(entries=[])
    with patch.dict("sys.modules", {"feedparser": mock_feed}):
        with patch("src.agents.news_agent.get_firecrawl") as mock_get_fc:
            fc_instance = MagicMock()
            fc_instance.is_available = True
            fc_instance.search.return_value = [
                {"title": "Palantir lands $400M AI contract", "url": "https://news.com/pltr"},
            ]
            mock_get_fc.return_value = fc_instance

            data = fetch_news("Palantir", limit=2)
            assert len(data["items"]) == 1
            assert data["items"][0]["title"] == "Palantir lands $400M AI contract"
            assert data["error"] is None
