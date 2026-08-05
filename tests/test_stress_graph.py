import pytest
from src.graph.build_graph import build_graph
from src.storage.session_store import clear_session, get_latest_report

def test_graph_international_stock_reliance():
    session_id = "test_graph_international_stock_reliance"
    clear_session(session_id)
    app = build_graph()
    result = app.invoke({
        "user_query": "research Reliance",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert result.get("intent") == "stock_research"
    assert result.get("ticker") == "RELIANCE.NS"

def test_graph_international_stock_toyota():
    session_id = "test_graph_international_stock_toyota"
    clear_session(session_id)
    app = build_graph()
    result = app.invoke({
        "user_query": "research Toyota",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert result.get("intent") == "stock_research"
    assert result.get("ticker") == "TM"

def test_graph_international_stock_samsung():
    session_id = "test_graph_international_stock_samsung"
    clear_session(session_id)
    app = build_graph()
    result = app.invoke({
        "user_query": "research Samsung",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert result.get("intent") == "stock_research"
    assert result.get("ticker") == "005930.KS"

def test_graph_sequential_research_recall_references_latest():
    session_id = "test_graph_sequential_research_recall_references_latest"
    clear_session(session_id)
    app = build_graph()
    
    app.invoke({
        "user_query": "research TSLA",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    
    app.invoke({
        "user_query": "research AAPL",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    
    result = app.invoke({
        "user_query": "why was it divergent?",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    
    assert result.get("intent") == "recall"
    assert result.get("ticker") == "AAPL"

def test_graph_research_then_clear_then_recall():
    session_id = "test_graph_research_then_clear_then_recall"
    clear_session(session_id)
    app = build_graph()
    
    app.invoke({
        "user_query": "research TSLA",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    
    clear_session(session_id)
    
    result = app.invoke({
        "user_query": "why was it divergent?",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    
    assert result.get("intent") == "clarify"
    assert result.get("ticker") is None
    assert "company or ticker" in result.get("response", "")

def test_graph_greeting_after_research_is_general_chat():
    session_id = "test_graph_greeting_after_research_is_general_chat"
    clear_session(session_id)
    app = build_graph()
    
    app.invoke({
        "user_query": "research TSLA",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    
    result = app.invoke({
        "user_query": "hello",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    
    assert result.get("intent") == "general_chat"

def test_graph_bare_ticker_msft():
    session_id = "test_graph_bare_ticker_msft"
    clear_session(session_id)
    app = build_graph()
    result = app.invoke({
        "user_query": "MSFT",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert result.get("intent") == "stock_research"
    assert result.get("ticker") == "MSFT"

def test_graph_bare_company_meta():
    session_id = "test_graph_bare_company_meta"
    clear_session(session_id)
    app = build_graph()
    result = app.invoke({
        "user_query": "meta",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert result.get("intent") == "stock_research"
    assert result.get("ticker") == "META"

def test_graph_thanks_is_general():
    session_id = "test_graph_thanks_is_general"
    clear_session(session_id)
    app = build_graph()
    result = app.invoke({
        "user_query": "thanks",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert result.get("intent") == "general_chat"

def test_graph_research_includes_chart_data():
    session_id = "test_graph_research_includes_chart_data"
    clear_session(session_id)
    app = build_graph()
    result = app.invoke({
        "user_query": "research NVDA",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert "chart_data" in result
    chart_data = result["chart_data"]
    assert "ticker" in chart_data
    assert "rows" in chart_data
    assert "period" in chart_data
    assert "interval" in chart_data

def test_graph_research_with_period():
    session_id = "test_graph_research_with_period"
    clear_session(session_id)
    app = build_graph()
    result = app.invoke({
        "user_query": "research TSLA for 1 year",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert result.get("intent") == "stock_research"
    # The period extraction should work

def test_graph_explain_divergence_with_session_is_recall():
    session_id = "test_graph_explain_divergence_with_session_is_recall"
    clear_session(session_id)
    app = build_graph()
    
    app.invoke({
        "user_query": "research AAPL",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    
    result = app.invoke({
        "user_query": "explain the divergence",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    
    assert result.get("intent") == "recall"
    assert result.get("ticker") == "AAPL"

def test_graph_explain_divergence_without_session_is_clarify():
    session_id = "test_graph_explain_divergence_without_session_is_clarify"
    clear_session(session_id)
    app = build_graph()
    result = app.invoke({
        "user_query": "explain the divergence",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    assert result.get("intent") == "clarify"

def test_graph_saved_report_has_chart_data():
    session_id = "test_graph_saved_report_has_chart_data"
    clear_session(session_id)
    app = build_graph()
    
    app.invoke({
        "user_query": "research NVDA",
        "session_id": session_id,
        "use_llm": False,
        "use_live_data": False,
    })
    
    report = get_latest_report(session_id)
    assert report is not None
    assert "chart_data" in report
    assert "rows" in report["chart_data"]
    assert len(report["chart_data"]["rows"]) > 0
