import pytest
from src.agents.common import (
    extract_ticker_and_topic,
    extract_time_period,
    keyword_sentiment,
    parse_json_object,
    resolve_stock_identity,
    trim,
    KNOWN_TICKERS,
    ticker_to_topic
)
from src.agents.market_agent import fetch_market, market_trend, summarize_market
from src.agents.supervisor_agent import classify_divergence, build_final_report
from src.agents.news_agent import summarize_news
from src.agents.social_agent import summarize_social

# ---------------------------------------------------------------------------
# 1. Ticker resolution edge cases
# ---------------------------------------------------------------------------
def test_ticker_international():
    assert extract_ticker_and_topic('research Reliance') == ('Reliance Industries', 'RELIANCE.NS')
    assert extract_ticker_and_topic('research Samsung') == ('Samsung Electronics', '005930.KS')
    assert extract_ticker_and_topic('research BYD') == ('BYD', '1211.HK')
    assert extract_ticker_and_topic('research Toyota') == ('Toyota', 'TM')
    assert extract_ticker_and_topic('research LVMH') == ('LVMH', 'MC.PA')

def test_ticker_case_variations():
    assert extract_ticker_and_topic('nvda') == ('Nvidia', 'NVDA')
    assert extract_ticker_and_topic('Nvda') == ('Nvidia', 'NVDA')
    assert extract_ticker_and_topic('NVDA') == ('Nvidia', 'NVDA')
    assert extract_ticker_and_topic('NvDa') == ('Nvidia', 'NVDA')

def test_ticker_dotted():
    assert extract_ticker_and_topic('RELIANCE.NS') == ('Reliance Industries', 'RELIANCE.NS')

def test_ticker_multi_word():
    assert extract_ticker_and_topic('Tata Motors') == ('Tata Motors', 'TATAMOTORS.NS')
    assert extract_ticker_and_topic('Samsung Electronics') == ('Samsung Electronics', '005930.KS')

def test_ticker_name_vs_ticker():
    assert extract_ticker_and_topic('research Tesla')[1] == 'TSLA'
    assert extract_ticker_and_topic('research TSLA')[1] == 'TSLA'

def test_ticker_generic_words():
    words = ['stock', 'market', 'price', 'investment', 'buy', 'sell', 'hold', 'signals', 'divergence']
    for word in words:
        assert resolve_stock_identity(word)['ticker'] is None

def test_ticker_empty_string():
    assert extract_ticker_and_topic('') == ('', None)

def test_ticker_whitespace():
    assert extract_ticker_and_topic('   ') == ('', None)

def test_ticker_special_chars():
    assert extract_ticker_and_topic('$TSLA') == ('Tesla', 'TSLA')

def test_ticker_all_known():
    for name, ticker in KNOWN_TICKERS.items():
        assert extract_ticker_and_topic(name) == (ticker_to_topic(ticker), ticker)

# ---------------------------------------------------------------------------
# 2. Time period extraction
# ---------------------------------------------------------------------------
def test_time_period_cases():
    assert extract_time_period('research TSLA for 1 year') == ('1y', '1wk')
    assert extract_time_period('show me 6 months of NVDA data') == ('6mo', '1wk')
    assert extract_time_period('10 days of AAPL') == ('1mo', '1d')
    assert extract_time_period('3 months of MSFT') == ('3mo', '1d')
    assert extract_time_period('2 years of data') == ('2y', '1wk')
    assert extract_time_period('5y') == ('5y', '1mo')
    assert extract_time_period('ytd') == ('ytd', '1d')
    assert extract_time_period('year to date') == ('ytd', '1d')
    assert extract_time_period('max data') == ('max', '1mo')
    assert extract_time_period('all time') == ('max', '1mo')
    assert extract_time_period('research TSLA') == ('5d', '1d')
    assert extract_time_period('1 day') == ('1d', '1h')
    assert extract_time_period('1d') == ('1d', '1h')
    assert extract_time_period('2 weeks') == ('1mo', '1d')

# ---------------------------------------------------------------------------
# 3. Sentiment edge cases
# ---------------------------------------------------------------------------
def test_sentiment_bullish():
    assert keyword_sentiment("beat gains surge rally strong bullish growth record upgrade profit optimistic") == "bullish"

def test_sentiment_bearish():
    assert keyword_sentiment("miss fall drop downgrade weak loss bearish misses falls drops lawsuit probe concern risk") == "bearish"

def test_sentiment_neutral():
    assert keyword_sentiment("strong gain but weak drop") == "neutral"

def test_sentiment_empty():
    assert keyword_sentiment("") == "neutral"

def test_sentiment_unrelated():
    assert keyword_sentiment("this is just a random text with no sentiment") == "neutral"

def test_sentiment_case_insensitive():
    assert keyword_sentiment("BULLISH GROWTH SURGE") == "bullish"

# ---------------------------------------------------------------------------
# 4. Market agent edge cases
# ---------------------------------------------------------------------------
def test_fetch_market_none():
    res = fetch_market(None)
    assert res['error'] == 'No ticker detected.'

def test_market_trend_empty():
    assert market_trend([]) == 'unavailable'

def test_market_trend_single():
    assert market_trend([{'close': 100}]) == 'unavailable'

def test_market_trend_boundary_up_flat():
    assert market_trend([{'close': 100}, {'close': 101}]) == 'flat'
    assert market_trend([{'close': 100}, {'close': 101.01}]) == 'up'

def test_market_trend_boundary_down_flat():
    assert market_trend([{'close': 100}, {'close': 99}]) == 'flat'
    assert market_trend([{'close': 100}, {'close': 98.99}]) == 'down'

def test_market_trend_zero_first():
    assert market_trend([{'close': 0}, {'close': 100}]) == 'flat'

def test_summarize_market_empty():
    res = summarize_market({"error": "Empty", "rows": [], "ticker": "TSLA"}, None)
    assert "unavailable" in res['summary']

def test_summarize_market_data():
    res = summarize_market({"ticker": "TSLA", "rows": [{"close": 100}, {"close": 105}]}, None)
    assert "TSLA" in res['summary']

# ---------------------------------------------------------------------------
# 5. Supervisor divergence matrix
# ---------------------------------------------------------------------------
def test_classify_divergence_aligned():
    assert classify_divergence("bullish", "up", "bullish") == "aligned"
    assert classify_divergence("bearish", "down", "bearish") == "aligned"
    assert classify_divergence("neutral", "flat", "neutral") == "aligned"

def test_classify_divergence_divergent():
    assert classify_divergence("bullish", "down", "neutral") == "divergent"
    assert classify_divergence("bearish", "up", "bullish") == "divergent"
    assert classify_divergence("bullish", "unavailable", "bearish") == "divergent"

def test_classify_divergence_mixed():
    assert classify_divergence("bullish", "up", "neutral") == "mixed"
    assert classify_divergence("bullish", "unavailable", "bullish") == "mixed"
    assert classify_divergence("neutral", "unavailable", "neutral") == "mixed"

def test_build_final_report_no_llm():
    res = build_final_report(
        topic="Tesla", ticker="TSLA", news_summary="N", news_sentiment="neutral",
        market_summary="M", market_trend="flat", social_summary="S", social_sentiment="neutral"
    )
    assert "Tesla" in res['final_report']
    assert res['divergence_verdict'] == "aligned"

# ---------------------------------------------------------------------------
# 6. News and Social summarize edge cases
# ---------------------------------------------------------------------------
def test_summarize_news_empty():
    res = summarize_news({"topic": "TSLA", "items": [], "error": "err"}, None)
    assert "unavailable" in res['summary']

def test_summarize_news_data():
    res = summarize_news({"topic": "TSLA", "items": [{"title": "bullish surge"}]}, None)
    assert "TSLA" in res['summary']
    assert res['sentiment'] == "bullish"

def test_summarize_social_empty():
    res = summarize_social({"topic": "TSLA", "items": [], "error": "err"}, None)
    assert "unavailable" in res['summary']

def test_summarize_social_data():
    res = summarize_social({"topic": "TSLA", "items": [{"title": "bearish drop miss"}]}, None)
    assert "TSLA" in res['summary']
    assert res['sentiment'] == "bearish"

# ---------------------------------------------------------------------------
# 7. Utility edge cases
# ---------------------------------------------------------------------------
def test_parse_json_object():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('garbage text') == {}
    assert parse_json_object('here is json: {"b": 2}') == {"b": 2}

def test_trim():
    assert trim("short text", 100) == "short text"
    assert trim("very long text indeed", 10) == "very lo..."
