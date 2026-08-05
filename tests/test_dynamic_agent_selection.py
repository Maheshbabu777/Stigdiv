from __future__ import annotations

import pytest
from src.agents.supervisor_agent import build_final_report, classify_divergence
from src.graph.build_graph import build_graph
from src.graph.router import next_nodes, route_message


def test_router_selects_news_only_agent():
    state = {
        "user_query": "What is the latest news on Tesla?",
        "session_id": "test_agent_news",
        "use_llm": False,
    }
    routed = route_message(state)
    assert routed["intent"] == "stock_research"
    assert routed["ticker"] == "TSLA"
    assert routed["agents"] == ["news"]
    assert next_nodes(routed) == ["news"]


def test_router_selects_market_only_agent():
    state = {
        "user_query": "Show me the price trend and chart for Apple",
        "session_id": "test_agent_market",
        "use_llm": False,
    }
    routed = route_message(state)
    assert routed["intent"] == "stock_research"
    assert routed["ticker"] == "AAPL"
    assert routed["agents"] == ["market"]
    assert next_nodes(routed) == ["market"]


def test_router_selects_social_only_agent():
    state = {
        "user_query": "What is the reddit sentiment on GameStop?",
        "session_id": "test_agent_social",
        "use_llm": False,
    }
    routed = route_message(state)
    assert routed["intent"] == "stock_research"
    assert routed["ticker"] == "GME"
    assert routed["agents"] == ["social"]
    assert next_nodes(routed) == ["social"]


def test_router_selects_full_divergence_agents():
    state = {
        "user_query": "Research TSLA for 1 year",
        "session_id": "test_agent_full",
        "use_llm": False,
    }
    routed = route_message(state)
    assert routed["intent"] == "stock_research"
    assert routed["ticker"] == "TSLA"
    assert set(routed["agents"]) == {"news", "market", "social"}
    assert set(next_nodes(routed)) == {"news", "market", "social"}


def test_single_agent_supervisor_synthesis():
    news_report = build_final_report(
        topic="Tesla",
        ticker="TSLA",
        news_summary="Tesla reports record Q2 vehicle deliveries.",
        news_sentiment="bullish",
        llm=None,
    )
    assert news_report["divergence_verdict"] == "news_briefing"
    assert "News Intelligence Briefing" in news_report["final_report"]
    assert "Tesla" in news_report["final_report"]
