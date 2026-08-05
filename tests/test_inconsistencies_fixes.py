from __future__ import annotations

import pytest
from src.agents.common import extract_ticker_and_topic, resolve_stock_identity
from src.graph.build_graph import build_graph
from src.graph.router import route_message
from src.storage.session_store import clear_session, find_report_for_query, save_report


def test_gold_in_india_routes_to_general_chat():
    res = route_message({
        "user_query": "should i buy gold in india or not",
        "session_id": "test_gold",
        "use_llm": False,
    })
    assert res["intent"] == "general_chat"


def test_ola_resolves_to_ola_electric_not_orla():
    resolved = resolve_stock_identity("do a research on ola")
    assert resolved["ticker"] == "OLAELEC.NS"
    assert "Ola" in resolved["topic"]
    assert "Orla" not in resolved["topic"]


def test_ola_motors_and_tata_motors_resolution():
    res_ola_motors = resolve_stock_identity("ola motors")
    assert res_ola_motors["ticker"] == "OLAELEC.NS"

    res_tata_motors = resolve_stock_identity("tata motors")
    assert res_tata_motors["ticker"] == "TATAMOTORS.NS"

    res_motors = resolve_stock_identity("motors")
    assert res_motors["ticker"] == "TATAMOTORS.NS"


def test_multi_report_session_lookup():
    session_id = "test_multi_report_session"
    clear_session(session_id)

    # Save 3 distinct reports in the same session
    save_report(session_id, {
        "topic": "Tesla",
        "ticker": "TSLA",
        "divergence_verdict": "aligned",
        "final_report": "Tesla report: Bullish across all signals.",
    })
    save_report(session_id, {
        "topic": "Apple",
        "ticker": "AAPL",
        "divergence_verdict": "divergent",
        "final_report": "Apple report: Price up but news bearish.",
    })
    save_report(session_id, {
        "topic": "Ola Electric",
        "ticker": "OLAELEC.NS",
        "divergence_verdict": "mixed",
        "final_report": "Ola Electric report: High volume and mixed sentiment.",
    })

    # Lookup Tesla specifically
    tsla_report = find_report_for_query(session_id, "Why was Tesla aligned?")
    assert tsla_report is not None
    assert tsla_report["ticker"] == "TSLA"

    # Lookup Apple specifically
    aapl_report = find_report_for_query(session_id, "Tell me more about Apple")
    assert aapl_report is not None
    assert aapl_report["ticker"] == "AAPL"

    # Default to latest when no specific stock mentioned
    latest_report = find_report_for_query(session_id, "Explain why it was mixed")
    assert latest_report is not None
    assert latest_report["ticker"] == "OLAELEC.NS"


def test_graph_multi_report_flow():
    session_id = "test_graph_multi"
    clear_session(session_id)
    graph = build_graph()

    # Turn 1: Research Tesla
    res1 = graph.invoke({
        "user_query": "research TSLA",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert res1["ticker"] == "TSLA"

    # Turn 2: Research Apple in the SAME session
    res2 = graph.invoke({
        "user_query": "research AAPL",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert res2["ticker"] == "AAPL"

    # Turn 3: Ask follow-up about Tesla in the SAME session
    res3 = graph.invoke({
        "user_query": "why was TSLA divergent?",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert res3["intent"] == "recall"
    assert res3["ticker"] == "TSLA"


def test_followup_timeframe_query_inherits_session_stock():
    session_id = "test_timeframe_followup"
    clear_session(session_id)

    # First research Tesla
    save_report(session_id, {
        "topic": "Tesla",
        "ticker": "TSLA",
        "period": "5d",
        "divergence_verdict": "aligned",
        "final_report": "Tesla initial report.",
    })

    # Follow-up: Ask for last year's to till date data without retyping Tesla
    routed = route_message({
        "user_query": "can you pull last year's to till date data",
        "session_id": session_id,
        "use_llm": False,
    })
    assert routed["intent"] == "stock_research"
    assert routed["ticker"] == "TSLA"
    assert routed["period"] in {"2y", "1y", "ytd"}
    assert "market" in routed["agents"]

