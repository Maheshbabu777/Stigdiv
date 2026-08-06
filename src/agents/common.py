from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)


# Fast-path lookup for common symbols across US, Indian (NSE), and Global exchanges
KNOWN_TICKERS = {
    # US Tech & Mega-caps
    "TESLA": "TSLA",
    "TSLA": "TSLA",
    "NVIDIA": "NVDA",
    "NVDA": "NVDA",
    "APPLE": "AAPL",
    "AAPL": "AAPL",
    "MICROSOFT": "MSFT",
    "MSFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "GOOGL": "GOOGL",
    "AMAZON": "AMZN",
    "AMZN": "AMZN",
    "META": "META",
    "AMD": "AMD",
    "PALANTIR": "PLTR",
    "PLTR": "PLTR",
    "NETFLIX": "NFLX",
    "NFLX": "NFLX",
    "INTEL": "INTC",
    "INTC": "INTC",
    "GAMESTOP": "GME",
    "GME": "GME",

    # Indian Tech, Auto & Mega-caps (NSE)
    "OLA": "OLAELEC.NS",
    "OLA ELECTRIC": "OLAELEC.NS",
    "OLA CABS": "OLAELEC.NS",
    "OLA MOTORS": "OLAELEC.NS",
    "OLAELEC": "OLAELEC.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "MOTORS": "TATAMOTORS.NS",
    "TATA": "TATAMOTORS.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "TATA POWER": "TATAPOWER.NS",
    "TCS": "TCS.NS",
    "RELIANCE": "RELIANCE.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "INFOSYS": "INFY",
    "INFY": "INFY",
    "WIPRO": "WIPRO.NS",
    "ZOMATO": "ZOMATO.NS",
    "SWIGGY": "SWIGGY.NS",
    "PAYTM": "PAYTM.NS",
    "JIO": "JIOFIN.NS",
    "JIO FINANCIAL": "JIOFIN.NS",
    "HDFC": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "LARSEN": "LT.NS",
    "L&T": "LT.NS",
    "MARUTI": "MARUTI.NS",
    "MARUTI SUZUKI": "MARUTI.NS",
    "BAJAJ AUTO": "BAJAJ-AUTO.NS",
    "MAHINDRA": "M&M.NS",
    "M&M": "M&M.NS",
    "ADANI": "ADANIENT.NS",
    "ADANI ENTERPRISES": "ADANIENT.NS",
    "ADANI PORTS": "ADANIPORTS.NS",
    "TITAN": "TITAN.NS",
    "KALYAN JEWELLERS": "KALYANKJIL.NS",
    "MUTHOOT": "MUTHOOTFIN.NS",

    # International
    "TOYOTA": "TM",
    "SAMSUNG": "005930.KS",
    "SAMSUNG ELECTRONICS": "005930.KS",
    "SONY": "SONY",
    "LVMH": "MC.PA",
    "SHOPIFY": "SHOP",
    "MERCADOLIBRE": "MELI",
    "BYD": "1211.HK",
    "ALIBABA": "BABA",
}

KNOWN_UNLISTED_COMPANIES = {
    "OPENAI": "OpenAI is a privately held artificial intelligence research company and is not publicly traded on any stock exchange.",
    "SPACEX": "SpaceX (Space Exploration Technologies Corp.) is a privately held aerospace manufacturer and space exploration company, not publicly traded.",
    "STRIPE": "Stripe is a privately held financial technology and payment processing company, not publicly listed.",
    "BYTEDANCE": "ByteDance (the parent company of TikTok) is a privately held technology company, not publicly listed.",
    "ANTHROPIC": "Anthropic is a privately held AI safety and research company, not publicly listed.",
    "DATABRICKS": "Databricks is a privately held data architecture and AI platform company, not publicly listed.",
    "CANVA": "Canva is a privately held graphic design and visual collaboration platform, not publicly listed.",
    "EPIC GAMES": "Epic Games is a privately held video game and software developer (creator of Unreal Engine and Fortnite), not publicly listed.",
    "DISCORD": "Discord is a privately held voice, video, and text communication service, not publicly listed.",
    "VALVE": "Valve Corporation is a privately held video game developer and digital distribution company (Steam), not publicly listed.",
}

POSITIVE_WORDS = {
    "beat", "beats", "bullish", "growth", "gain", "gains", "surge", "rally",
    "record", "upgrade", "strong", "profit", "optimistic",
}

NEGATIVE_WORDS = {
    "miss", "misses", "bearish", "fall", "falls", "drop", "drops", "lawsuit",
    "probe", "downgrade", "weak", "loss", "concern", "risk",
}


def extract_ticker_and_topic(message: str) -> tuple[str, str | None]:
    """Dynamically extract stock ticker or company topic from user message."""
    cleaned = re.sub(r"\s+", " ", message).strip()
    upper = cleaned.upper()

    # Fast-path for multi-word or exact known company names
    for name in sorted(KNOWN_TICKERS.keys(), key=lambda x: -len(x)):
        if re.search(rf"\b{re.escape(name)}\b", upper):
            ticker = KNOWN_TICKERS[name]
            return ticker_to_topic(ticker), ticker

    candidates = re.findall(r"\b[A-Za-z.]{2,12}\b", message)
    ignored = {"THE", "AND", "FOR", "WITH", "THIS", "THAT", "WHY", "HOW", "NOW", "WHAT", "SHOW", "TELL"}
    for candidate in candidates:
        upper_candidate = candidate.upper()
        if upper_candidate not in ignored and upper_candidate in KNOWN_TICKERS:
            ticker = KNOWN_TICKERS[upper_candidate]
            return ticker_to_topic(ticker), ticker

    # Clean research phrases to extract raw topic/company name for dynamic lookup
    topic = re.sub(
        r"\b(research|analyze|analyse|stock|ticker|share|shares|signal|divergence|for|about|now|please|price|quote|today|do a|do|check)\b",
        "",
        cleaned,
        flags=re.I,
    )
    topic = re.sub(r"\s+", " ", topic).strip(" ?.,") or cleaned
    return topic, None


def resolve_stock_identity(message: str) -> dict[str, Any]:
    """Universally resolve any company or ticker dynamically via live APIs and search."""
    topic, ticker = extract_ticker_and_topic(message)
    if ticker:
        return {"topic": topic, "ticker": ticker, "is_listed": True, "note": None}

    # Check known unlisted companies fast-path
    upper_query = topic.upper().strip()
    for name, reason in KNOWN_UNLISTED_COMPANIES.items():
        if re.search(rf"\b{re.escape(name)}\b", upper_query):
            return {
                "topic": topic,
                "ticker": None,
                "is_listed": False,
                "note": reason,
            }

    # 1. Dynamic symbol resolution via yfinance Search API
    resolved = yahoo_resolve_symbol(topic)
    if resolved and resolved.get("ticker"):
        return {
            "topic": resolved.get("topic") or topic,
            "ticker": resolved.get("ticker"),
            "is_listed": True,
            "note": None,
        }

    # 2. Dynamic web search & Firecrawl on-demand investigation
    if topic and not _is_generic_lookup_query(topic):
        web_info = deep_web_company_search(topic)
        if web_info.get("ticker"):
            return {
                "topic": web_info.get("topic") or topic,
                "ticker": web_info.get("ticker"),
                "is_listed": True,
                "note": None,
            }
        if web_info.get("note"):
            return {
                "topic": topic,
                "ticker": None,
                "is_listed": False,
                "note": web_info.get("note"),
            }

    return {"topic": topic, "ticker": None, "is_listed": False, "note": None}


def yahoo_resolve_symbol(query: str) -> dict[str, str | None] | None:
    """Dynamically search Yahoo Finance for any publicly traded security across global exchanges."""
    if not query or _is_generic_lookup_query(query):
        return None
    try:
        import yfinance as yf

        search = yf.Search(query, max_results=10, news_count=0, lists_count=0, enable_fuzzy_query=True)
        quotes = getattr(search, "quotes", []) or []
        for quote in quotes:
            symbol = quote.get("symbol")
            quote_type = quote.get("quoteType") or quote.get("typeDisp")
            if symbol and str(quote_type).upper() in {"EQUITY", "ETF"} and _quote_matches_query(query, quote):
                name = quote.get("shortname") or quote.get("longname") or query
                return {"topic": name, "ticker": symbol}
    except Exception as exc:
        logger.debug("yfinance symbol resolution failed for '%s': %s", query, exc)
        return None
    return None


def deep_web_company_search(query: str) -> dict[str, Any]:
    """Perform a web search and on-demand Firecrawl crawl to inspect unlisted or obscure companies."""
    cleaned = _normalize_lookup_text(query)
    if not cleaned or _is_generic_lookup_query(cleaned):
        return {}

    # 1. Live Yahoo Finance HTTP search endpoint
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=8&newsCount=0"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=3.5)
        if resp.status_code == 200:
            data = resp.json()
            quotes = data.get("quotes", [])
            for q in quotes:
                symbol = q.get("symbol")
                qtype = (q.get("quoteType") or q.get("typeDisp") or "").upper()
                if symbol and qtype in {"EQUITY", "ETF"} and _quote_matches_query(query, q):
                    name = q.get("shortname") or q.get("longname") or query
                    return {"topic": name, "ticker": symbol, "is_listed": True}
    except Exception as exc:
        logger.debug("Web finance search error for '%s': %s", query, exc)

    # 2. On-demand Firecrawl search if configured
    try:
        from src.tools.firecrawl_client import get_firecrawl
        firecrawl = get_firecrawl()
        if firecrawl.is_available:
            results = firecrawl.search(f"{query} company public stock or private valuation", limit=2)
            if results:
                combined_text = " ".join([r.get("markdown", "") + " " + r.get("title", "") for r in results]).lower()
                if any(w in combined_text for w in ("privately held", "private company", "unlisted", "venture-backed")):
                    return {
                        "topic": query,
                        "ticker": None,
                        "is_listed": False,
                        "note": f"{query} is a privately held company and is not listed on public stock exchanges.",
                    }
    except Exception as exc:
        logger.debug("Firecrawl company search error for '%s': %s", query, exc)

    return {
        "topic": query,
        "ticker": None,
        "is_listed": False,
        "note": f"I performed a search for '{query}', but no publicly traded stock or listing symbol was found. It may be a private company or unlisted entity.",
    }


def _is_generic_lookup_query(query: str) -> bool:
    generic = {
        "divergence", "divergent", "signal", "signals", "market", "markets",
        "stock", "stocks", "share", "shares", "price", "prices", "buy", "sell",
        "hold", "invest", "investment", "gold", "silver", "crypto", "oil",
        "india", "world", "trade", "trading",
    }
    normalized = _normalize_lookup_text(query)
    return normalized in generic


def _quote_matches_query(query: str, quote: dict[str, Any]) -> bool:
    normalized_query = _normalize_lookup_text(query)
    if not normalized_query:
        return False
    raw_symbol = str(quote.get("symbol", "")).upper()
    symbol = _normalize_lookup_text(raw_symbol)
    symbol_base = _normalize_lookup_text(raw_symbol.split(".")[0])
    shortname = _normalize_lookup_text(str(quote.get("shortname", "")))
    longname = _normalize_lookup_text(str(quote.get("longname", "")))

    # Exact symbol or base symbol match (e.g. "AAPL" == "AAPL" or "RELIANCE" == "RELIANCE.NS")
    if normalized_query == symbol or normalized_query == symbol_base:
        return True

    query_tokens = set(normalized_query.split())
    name_tokens = set((shortname + " " + longname).split())
    if not query_tokens:
        return False

    overlap = query_tokens & name_tokens
    if len(query_tokens) == 1:
        token = next(iter(query_tokens))
        if len(token) <= 2:
            return token == symbol or token == symbol_base
        # Token must be in company name tokens, or shortname/longname starts with token
        return token in name_tokens or shortname.startswith(token) or longname.startswith(token)

    return len(overlap) >= max(1, min(2, len(query_tokens)))


def _normalize_lookup_text(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9. ]+", " ", value).lower()
    return re.sub(r"\s+", " ", value).strip()


def ticker_to_topic(ticker: str) -> str:
    return {
        "TSLA": "Tesla",
        "NVDA": "Nvidia",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet",
        "AMZN": "Amazon",
        "META": "Meta",
        "AMD": "AMD",
        "PLTR": "Palantir",
        "NFLX": "Netflix",
        "INTC": "Intel",
        "OLAELEC.NS": "Ola Electric",
        "TATAMOTORS.NS": "Tata Motors",
        "TATASTEEL.NS": "Tata Steel",
        "TATAPOWER.NS": "Tata Power",
        "TCS.NS": "Tata Consultancy Services",
        "RELIANCE.NS": "Reliance Industries",
        "INFY": "Infosys",
        "WIPRO.NS": "Wipro",
        "ZOMATO.NS": "Zomato",
        "SWIGGY.NS": "Swiggy",
        "PAYTM.NS": "Paytm",
        "JIOFIN.NS": "Jio Financial Services",
        "HDFCBANK.NS": "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank",
        "SBIN.NS": "State Bank of India",
        "BHARTIARTL.NS": "Bharti Airtel",
        "ITC.NS": "ITC Limited",
        "LT.NS": "Larsen & Toubro",
        "MARUTI.NS": "Maruti Suzuki",
        "BAJAJ-AUTO.NS": "Bajaj Auto",
        "M&M.NS": "Mahindra & Mahindra",
        "ADANIENT.NS": "Adani Enterprises",
        "ADANIPORTS.NS": "Adani Ports",
        "TITAN.NS": "Titan Company",
        "KALYANKJIL.NS": "Kalyan Jewellers",
        "MUTHOOTFIN.NS": "Muthoot Finance",
        "TM": "Toyota",
        "005930.KS": "Samsung Electronics",
        "SONY": "Sony",
        "MC.PA": "LVMH",
        "SHOP": "Shopify",
        "MELI": "MercadoLibre",
        "1211.HK": "BYD",
        "BABA": "Alibaba",
    }.get(ticker.upper(), ticker.upper())


def keyword_sentiment(text: str) -> str:
    lower = text.lower()
    positive = sum(1 for word in POSITIVE_WORDS if word in lower)
    negative = sum(1 for word in NEGATIVE_WORDS if word in lower)
    if positive > negative:
        return "bullish"
    if negative > positive:
        return "bearish"
    return "neutral"


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def trim(text: str, limit: int = 2400) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Time-period extraction for flexible market-data windows
# ---------------------------------------------------------------------------

_VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}


def extract_time_period(message: str) -> tuple[str, str]:
    """Parse a user message for a time range and return *(yfinance_period, interval)*."""
    lower = message.lower()

    # 1. Lifetime, IPO, Inception, All-time, or Since Listing
    if any(
        p in lower
        for p in (
            "from the time that company listed",
            "from the time the company listed",
            "from the time it listed",
            "from the time of listing",
            "time that company listed",
            "time company listed",
            "since listing",
            "since listed",
            "since it listed",
            "since it was listed",
            "from listing",
            "from its listing",
            "since being listed",
            "since ipo",
            "from ipo",
            "since its ipo",
            "post ipo",
            "since inception",
            "from inception",
            "inception to date",
            "since the beginning",
            "from the beginning",
            "since the start",
            "from the start",
            "since start",
            "from start",
            "since day 1",
            "since day one",
            "from day 1",
            "from day one",
            "all time",
            "all-time",
            "alltime",
            "entire lifetime",
            "full lifetime",
            "entire history",
            "full history",
            "complete history",
            "all price data",
            "entire price data",
            "full price data",
            "max",
            "maximum",
        )
    ) or re.search(r"\b(listed in the market|since (it )?got listed|listed on (the )?exchange|listed on market|all[\s-]?time|inception)\b", lower):
        return ("max", "1mo")

    # 2. Multi-year lookbacks
    if any(p in lower for p in ("10 years", "10 year", "10 yrs", "10yr", "past 10 years", "last 10 years", "decade")):
        return ("10y", "1mo")

    if any(p in lower for p in ("5 years", "5 year", "5 yrs", "5yr", "past 5 years", "last 5 years")):
        return ("5y", "1mo")

    if any(p in lower for p in ("2 years", "2 year", "2 yrs", "2yr", "past 2 years", "last 2 years")):
        return ("2y", "1wk")

    if any(p in lower for p in ("last year's to till date", "last year to till date", "last year to date", "last year till date", "last year to now", "last year till now")):
        return ("2y", "1wk")

    if any(p in lower for p in ("ytd", "year to date", "till date", "to date", "from start of year")):
        return ("ytd", "1d")

    if any(p in lower for p in ("full year", "full years", "last year", "past year", "previous year", "one year", "1 year", "1 yr", "1yr", "yearly", "annual", "entire year", "whole year", "12 months", "12 month", "past 12 months", "last 12 months")):
        return ("1y", "1wk")

    if any(p in lower for p in ("last 6 months", "past 6 months", "six months", "6 months", "half year", "6mo")):
        return ("6mo", "1wk")

    if any(p in lower for p in ("last 3 months", "past 3 months", "three months", "past quarter", "last quarter", "3 months", "3mo", "quarterly")):
        return ("3mo", "1d")

    if any(p in lower for p in ("last month", "past month", "previous month", "one month", "1 month", "1mo", "monthly")):
        return ("1mo", "1d")

    if any(p in lower for p in ("last week", "past week", "previous week", "one week", "1 week", "weekly", "5d", "5 days")):
        return ("5d", "1d")

    match = re.search(r"(\d+)\s*(days?|d|weeks?|w|months?|mo|m|years?|yr|y)\b", lower)
    if match:
        num = int(match.group(1))
        unit = match.group(2)

        if unit.startswith("day") or unit == "d":
            return _period_for_days(num)
        if unit.startswith("week") or unit == "w":
            return _period_for_days(num * 7)
        if unit.startswith("month") or unit == "mo" or unit == "m":
            return _period_for_months(num)
        if unit.startswith("year") or unit.startswith("yr") or unit == "y":
            return _period_for_years(num)

    return ("5d", "1d")


def _period_for_days(n: int) -> tuple[str, str]:
    if n <= 1:
        return ("1d", "1h")
    if n <= 5:
        return ("5d", "1d")
    if n <= 30:
        return ("1mo", "1d")
    if n <= 90:
        return ("3mo", "1d")
    if n <= 180:
        return ("6mo", "1wk")
    if n <= 365:
        return ("1y", "1wk")
    return ("2y", "1wk")


def _period_for_months(n: int) -> tuple[str, str]:
    if n <= 1:
        return ("1mo", "1d")
    if n <= 3:
        return ("3mo", "1d")
    if n <= 6:
        return ("6mo", "1wk")
    if n <= 12:
        return ("1y", "1wk")
    if n <= 24:
        return ("2y", "1wk")
    return ("5y", "1mo")


def _period_for_years(n: int) -> tuple[str, str]:
    if n <= 1:
        return ("1y", "1wk")
    if n <= 2:
        return ("2y", "1wk")
    if n <= 5:
        return ("5y", "1mo")
    return ("max", "1mo")
