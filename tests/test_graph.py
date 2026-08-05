from src.graph.build_graph import build_graph
from src.storage.session_store import clear_session, get_latest_report


def test_graph_new_research_saves_report_with_stub_data():
    clear_session("graph-test")
    app = build_graph()

    result = app.invoke(
        {
            "user_query": "research NVDA",
            "session_id": "graph-test",
            "use_llm": False,
            "use_live_data": False,
        }
    )

    latest = get_latest_report("graph-test")
    assert result["intent"] == "stock_research"
    assert result["ticker"] == "NVDA"
    assert result["divergence_verdict"] == "divergent"
    assert latest is not None
    assert latest["ticker"] == "NVDA"


def test_graph_recall_uses_session_memory():
    clear_session("recall-test")
    app = build_graph()
    app.invoke(
        {
            "user_query": "research AAPL",
            "session_id": "recall-test",
            "use_llm": False,
            "use_live_data": False,
        }
    )

    result = app.invoke(
        {
            "user_query": "why was it divergent?",
            "session_id": "recall-test",
            "use_llm": False,
            "use_live_data": False,
        }
    )

    assert result["intent"] == "recall"
    assert "current session report" in result["response"]
    assert result["ticker"] == "AAPL"


def test_graph_followup_without_memory_does_not_invent_research():
    clear_session("empty-recall")
    app = build_graph()

    result = app.invoke(
        {
            "user_query": "why was it divergent?",
            "session_id": "empty-recall",
            "use_llm": False,
            "use_live_data": False,
        }
    )

    assert result["intent"] == "clarify"
    assert result["topic"] is None
    assert result["ticker"] is None
    assert "company or ticker" in result["response"]


def test_graph_buy_question_without_memory_does_not_resolve_random_ticker():
    clear_session("empty-buy-question")
    app = build_graph()

    result = app.invoke(
        {
            "user_query": "why is it divergent why is it not a good time to buy?",
            "session_id": "empty-buy-question",
            "use_llm": False,
            "use_live_data": False,
        }
    )

    assert result["intent"] == "clarify"
    assert result["topic"] is None
    assert result["ticker"] is None
    assert "company or ticker" in result["response"]


def test_graph_why_is_there_divergence_without_memory_does_not_lookup_stock():
    clear_session("empty-divergence-question")
    app = build_graph()

    result = app.invoke(
        {
            "user_query": "why is there a divergence?",
            "session_id": "empty-divergence-question",
            "use_llm": False,
            "use_live_data": False,
        }
    )

    assert result["intent"] == "clarify"
    assert result["topic"] is None
    assert result["ticker"] is None
    assert "company or ticker" in result["response"]


def test_graph_general_question_does_not_lookup_random_stock():
    clear_session("general-question")
    app = build_graph()

    result = app.invoke(
        {
            "user_query": "what is divergence?",
            "session_id": "general-question",
            "use_llm": False,
            "use_live_data": False,
        }
    )

    assert result["intent"] == "general_chat"
    assert result["topic"] is None
    assert result["ticker"] is None


def test_graph_single_company_name_can_spawn_stock_agents():
    clear_session("company-name")
    app = build_graph()

    result = app.invoke(
        {
            "user_query": "apple",
            "session_id": "company-name",
            "use_llm": False,
            "use_live_data": False,
        }
    )

    assert result["intent"] == "stock_research"
    assert result["ticker"] == "AAPL"


def test_graph_vague_investment_question_asks_for_target():
    clear_session("vague-invest")
    app = build_graph()

    result = app.invoke(
        {
            "user_query": "is it a good time to buy?",
            "session_id": "vague-invest",
            "use_llm": False,
            "use_live_data": False,
        }
    )

    assert result["intent"] == "clarify"
    assert result["ticker"] is None


def test_graph_market_request_with_company_uses_stock_tools():
    clear_session("company-market-question")
    app = build_graph()

    result = app.invoke(
        {
            "user_query": "what is happening with Microsoft stock today?",
            "session_id": "company-market-question",
            "use_llm": False,
            "use_live_data": False,
        }
    )

    assert result["intent"] == "stock_research"
    assert result["ticker"] == "MSFT"


def test_graph_market_request_with_today_cleans_target():
    clear_session("toyota-market-question")
    app = build_graph()

    result = app.invoke(
        {
            "user_query": "what is happening with Toyota stock today?",
            "session_id": "toyota-market-question",
            "use_llm": False,
            "use_live_data": False,
        }
    )

    assert result["intent"] == "stock_research"
    assert result["ticker"] == "TM"
