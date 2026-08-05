import pytest

from src.graph.router import route_message, _clean_topic
from src.storage.session_store import clear_session, save_report


@pytest.fixture(autouse=True)
def setup_teardown():
    clear_session("test-session")
    yield
    clear_session("test-session")


def test_general_chat_detection():
    # 'hi'
    state = {"user_query": "hi", "session_id": "test-session", "use_llm": False}
    assert route_message(state)["intent"] == "general_chat"
    
    # 'hello'
    state = {"user_query": "hello", "session_id": "test-session", "use_llm": False}
    assert route_message(state)["intent"] == "general_chat"
    
    # 'what can you do'
    state = {"user_query": "what can you do", "session_id": "test-session", "use_llm": False}
    assert route_message(state)["intent"] == "general_chat"
    
    # 'what is divergence?'
    state = {"user_query": "what is divergence?", "session_id": "test-session", "use_llm": False}
    assert route_message(state)["intent"] == "general_chat"
    
    # 'thanks'
    state = {"user_query": "thanks", "session_id": "test-session", "use_llm": False}
    assert route_message(state)["intent"] != "stock_research"


def test_recall_detection_with_session():
    save_report("test-session", {"topic": "Apple", "ticker": "AAPL"})
    
    state = {"user_query": "why was it divergent?", "session_id": "test-session", "use_llm": False}
    assert route_message(state)["intent"] == "recall"
    
    state = {"user_query": "explain the divergence", "session_id": "test-session", "use_llm": False}
    assert route_message(state)["intent"] == "recall"
    
    state = {"user_query": "should i buy?", "session_id": "test-session", "use_llm": False}
    assert route_message(state)["intent"] == "recall"
    
    state = {"user_query": "you said it was mixed", "session_id": "test-session", "use_llm": False}
    assert route_message(state)["intent"] == "recall"


def test_recall_detection_without_session():
    state = {"user_query": "why was it divergent?", "session_id": "test-session", "use_llm": False}
    assert route_message(state)["intent"] == "clarify"
    
    state = {"user_query": "explain the divergence", "session_id": "test-session", "use_llm": False}
    assert route_message(state)["intent"] != "stock_research"


def test_stock_research_detection():
    # 'research TSLA'
    state = {"user_query": "research TSLA", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] == "stock_research"
    assert res["ticker"] == "TSLA"
    
    # 'analyze Nvidia'
    state = {"user_query": "analyze Nvidia", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] == "stock_research"
    assert res["ticker"] == "NVDA"
    
    # 'what is happening with Microsoft stock today?'
    state = {"user_query": "what is happening with Microsoft stock today?", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] == "stock_research"
    assert res["ticker"] == "MSFT"
    
    # 'AAPL' as bare input
    state = {"user_query": "AAPL", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] == "stock_research"
    assert res["ticker"] == "AAPL"
    
    # 'research' alone (no target)
    state = {"user_query": "research", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] == "clarify"
    
    # 'analyze stock' (no company)
    state = {"user_query": "analyze stock", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] in ["clarify", "general_chat"]


def test_edge_cases():
    # Very long message (500 chars of gibberish)
    state = {"user_query": "a" * 500, "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] in ["general_chat", "clarify", "stock_research"]
    
    # Unicode characters: 'research 日本語'
    state = {"user_query": "research 日本語", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] in ["clarify", "stock_research", "general_chat"]
    
    # Repeated whitespace: '  research    NVDA   '
    state = {"user_query": "  research    NVDA   ", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] == "stock_research"
    assert res["ticker"] == "NVDA"
    
    # Numbers only: '12345'
    state = {"user_query": "12345", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] in ["general_chat", "clarify", "stock_research"]
    
    # Empty-ish after cleaning: all stop words
    state = {"user_query": "stock market", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] in ["general_chat", "clarify"]
    
    # Message with special chars: 'research $TSLA!!!'
    state = {"user_query": "research $TSLA!!!", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["ticker"] == "TSLA"
    
    # 'is it a good time to buy?' without session
    state = {"user_query": "is it a good time to buy?", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] == "clarify"
    
    # Period extraction in router: 'research TSLA for 1 year' should have period='1y' in result
    state = {"user_query": "research TSLA for 1 year", "session_id": "test-session", "use_llm": False}
    res = route_message(state)
    assert res["intent"] == "stock_research"
    assert res["period"] == "1y"


def test_clean_topic_function():
    assert _clean_topic("research analyze NVDA stock ticker") == "NVDA"
    assert _clean_topic("what is happening with Toyota stock today?") == "Toyota"
    assert _clean_topic("") == ""
    assert _clean_topic("research stock market") in ["", "market"]
