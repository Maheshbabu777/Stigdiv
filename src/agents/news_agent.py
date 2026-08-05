from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote_plus

from src.agents.common import keyword_sentiment, trim
from src.llm.client import LLMClient
from src.tools.firecrawl_client import get_firecrawl

logger = logging.getLogger(__name__)


def fetch_news(topic: str, ticker: str | None = None, limit: int = 12) -> dict[str, Any]:
    """Extensively fetch financial news from multiple institutional and media sources."""
    items: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    error_msgs: list[str] = []

    def _add_item(item: dict[str, Any]) -> None:
        title = (item.get("title") or "").strip()
        if not title:
            return
        # Normalize title for deduplication
        norm_title = "".join(c.lower() for c in title if c.isalnum())[:40]
        if norm_title and norm_title not in seen_titles:
            seen_titles.add(norm_title)
            items.append(item)

    # 1. Institutional Ticker News via yfinance
    if ticker:
        try:
            import yfinance as yf
            ticker_obj = yf.Ticker(ticker)
            raw_news = getattr(ticker_obj, "news", []) or []
            for entry in raw_news:
                title = entry.get("title") or (entry.get("content", {}).get("title") if isinstance(entry.get("content"), dict) else None)
                if title:
                    publisher = entry.get("publisher") or "Yahoo Finance"
                    link = entry.get("link") or (entry.get("content", {}).get("canonicalUrl", {}).get("url") if isinstance(entry.get("content"), dict) else f"https://finance.yahoo.com/quote/{ticker}/news")
                    summary = entry.get("summary") or (entry.get("content", {}).get("summary") if isinstance(entry.get("content"), dict) else "")
                    _add_item({
                        "title": title,
                        "link": link or f"https://finance.yahoo.com/quote/{ticker}/news",
                        "publisher": publisher,
                        "source": publisher,
                        "snippet": (summary or title)[:350],
                        "published": "Recent Institutional",
                        "crawled_via": "yfinance_institutional",
                    })
        except Exception as exc:
            logger.debug("yfinance news fetch error for %s: %s", ticker, exc)
            error_msgs.append(f"Institutional news error: {exc}")

    # 2. Deep Web Crawl: Firecrawl Financial Search (when configured)
    firecrawl = get_firecrawl()
    if firecrawl.is_available:
        try:
            search_query = f"{topic} {ticker or ''} stock financial news earnings catalysts analysis"
            crawled = firecrawl.search(search_query, limit=6)
            for entry in crawled:
                if entry.get("title"):
                    raw_url = entry.get("url", "")
                    domain = "Financial Web"
                    if raw_url and "://" in raw_url:
                        domain = raw_url.split("/")[2].replace("www.", "")
                    _add_item({
                        "title": entry["title"],
                        "link": raw_url or f"https://finance.yahoo.com/quote/{ticker or topic}",
                        "publisher": domain,
                        "source": domain,
                        "snippet": (entry.get("markdown", "") or entry.get("description", ""))[:450],
                        "published": "Live Web Crawl",
                        "crawled_via": "firecrawl",
                    })
        except Exception as exc:
            logger.debug("Firecrawl search error: %s", exc)
            error_msgs.append(f"Firecrawl error: {exc}")

    # 3. Comprehensive Google News Financial RSS Feeds
    try:
        import feedparser

        search_terms = [
            f"{topic} {ticker or ''} stock market news",
            f"{topic} earnings financial results valuation",
        ]
        for term in search_terms:
            if len(items) >= limit:
                break
            url = f"https://news.google.com/rss/search?q={quote_plus(term.strip())}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:8]:
                title = getattr(entry, "title", "")
                link = getattr(entry, "link", "")
                source_name = "Google News"
                if hasattr(entry, "source") and isinstance(entry.source, dict) and entry.source.get("title"):
                    source_name = entry.source.get("title")
                elif " - " in title:
                    parts = title.rsplit(" - ", 1)
                    source_name = parts[1].strip()
                    title = parts[0].strip()

                snippet = getattr(entry, "summary", "") or getattr(entry, "description", "")
                _add_item({
                    "title": title,
                    "link": link or f"https://news.google.com/search?q={quote_plus(topic)}",
                    "publisher": source_name,
                    "source": source_name,
                    "snippet": snippet[:350] if snippet else "",
                    "published": getattr(entry, "published", "Recent"),
                    "crawled_via": "rss",
                })
    except Exception as exc:
        logger.debug("RSS news parse error: %s", exc)
        error_msgs.append(f"RSS error: {exc}")

    return {
        "topic": topic,
        "ticker": ticker,
        "items": items[:limit],
        "error": None if items else ("; ".join(error_msgs) or "No news items returned."),
    }


def summarize_news(data: dict[str, Any], llm: LLMClient | None = None) -> dict[str, Any]:
    """Synthesize news articles with deep factual catalysts and sentiment analysis."""
    items = data.get("items", [])
    if not items:
        return {
            "summary": f"News data unavailable: {data.get('error') or 'no items returned'}.",
            "sentiment": "neutral",
            "raw": data,
        }

    titles = [item["title"] for item in items if item.get("title")]
    snippets = [
        f"[{item.get('publisher') or item.get('source', 'Media')}] {item['title']}: {item.get('snippet', '')}".strip()
        for item in items
        if item.get("title")
    ]
    joined = " | ".join(titles)
    sentiment = keyword_sentiment(joined)
    deterministic = f"Recent institutional headlines for {data['topic']} indicate a {sentiment} backdrop across {len(items)} sources: " + "; ".join(titles[:4])
    if llm is None:
        return {"summary": trim(deterministic, 1200), "sentiment": sentiment, "raw": data}

    content_bullets = "\n- ".join(snippets[:10] if snippets else titles[:10])
    prompt = (
        f"You are a Senior Financial Intelligence Analyst analyzing deep news coverage for {data['topic']}"
        + (f" ({data.get('ticker')})" if data.get("ticker") else "") + ".\n\n"
        f"Extensive Media & Institutional News Excerpts ({len(snippets)} reports):\n- {content_bullets}\n\n"
        "Instructions:\n"
        "1. Provide a comprehensive, high-signal financial synthesis (3-4 concise paragraphs or bullet groups).\n"
        "2. Detail specific operational catalysts, earnings data, product launches, analyst rating changes, or regulatory matters.\n"
        "3. Evaluate the net institutional sentiment as Bullish, Bearish, or Neutral with explicit factual drivers.\n"
        "4. Write in crisp, professional financial English without preamble or filler."
    )
    result = llm.generate(prompt, task="news_summary")
    return {"summary": result.text or deterministic, "sentiment": sentiment, "raw": data}
