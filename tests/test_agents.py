from src.agents.common import extract_ticker_and_topic, keyword_sentiment, resolve_stock_identity
from src.agents.market_agent import market_trend, summarize_market
from src.agents.supervisor_agent import classify_divergence


def test_extract_known_ticker_from_company_name():
    topic, ticker = extract_ticker_and_topic("research Nvidia stock")

    assert topic == "Nvidia"
    assert ticker == "NVDA"


def test_extract_known_ticker_from_lowercase_symbol():
    topic, ticker = extract_ticker_and_topic("research aapl")

    assert topic == "Apple"
    assert ticker == "AAPL"


def test_generic_finance_word_does_not_resolve_as_ticker():
    result = resolve_stock_identity("divergence")

    assert result["ticker"] is None


def test_keyword_sentiment_labels_text():
    assert keyword_sentiment("strong growth and profit beat") == "bullish"
    assert keyword_sentiment("weak outlook and downgrade risk") == "bearish"


def test_market_trend_from_rows():
    assert market_trend([{"close": 100}, {"close": 104}]) == "up"
    assert market_trend([{"close": 100}, {"close": 96}]) == "down"
    assert market_trend([{"close": 100}, {"close": 100.4}]) == "flat"
    assert market_trend([]) == "unavailable"


def test_market_summary_without_llm():
    result = summarize_market({"ticker": "AAPL", "rows": [{"close": 100}, {"close": 103}], "error": None})

    assert result["trend"] == "up"
    assert "AAPL" in result["summary"]


def test_supervisor_detects_divergence():
    assert classify_divergence("bullish", "down", "neutral") == "divergent"
