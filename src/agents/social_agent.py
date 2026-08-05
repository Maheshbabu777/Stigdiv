from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote_plus

import requests

from src.agents.common import keyword_sentiment, trim
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


def fetch_social(topic: str, ticker: str | None = None, limit: int = 15) -> dict[str, Any]:
    """Extensively fetch retail and community sentiment across StockTwits, Reddit, and Hacker News."""
    items: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    errors: list[str] = []

    def _add_item(item: dict[str, Any]) -> None:
        title = (item.get("title") or "").strip()
        if not title:
            return
        norm = "".join(c.lower() for c in title if c.isalnum())[:35]
        if norm and norm not in seen_titles:
            seen_titles.add(norm)
            items.append(item)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SignalDivergence/2.0"}

    # 1. StockTwits Live Real-Time Trader Feed
    stocktwits_ticker = ticker or topic.upper().split()[0]
    try:
        st_url = f"https://api.stocktwits.com/api/2/streams/symbol/{stocktwits_ticker}.json"
        resp = requests.get(st_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            st_data = resp.json()
            messages = st_data.get("messages", [])
            for msg in messages[:8]:
                body = msg.get("body", "").replace("\n", " ").strip()
                sentiment_info = msg.get("entities", {}).get("sentiment", {})
                sentiment_label = sentiment_info.get("basic") if isinstance(sentiment_info, dict) else None
                user_name = msg.get("user", {}).get("username", "Trader")
                msg_id = msg.get("id")
                link = f"https://stocktwits.com/{user_name}/message/{msg_id}" if msg_id else f"https://stocktwits.com/symbol/{stocktwits_ticker}"
                _add_item({
                    "source": "StockTwits",
                    "publisher": f"StockTwits (@{user_name})",
                    "title": body[:200],
                    "sentiment": sentiment_label or "Neutral",
                    "link": link,
                })
    except Exception as exc:
        logger.debug("StockTwits fetch error: %s", exc)
        errors.append(f"StockTwits: {exc}")

    # 2. Reddit Multi-Subreddit Search (r/wallstreetbets, r/stocks, r/investing)
    try:
        reddit_query = f"{ticker or topic} stock"
        reddit_url = f"https://www.reddit.com/r/wallstreetbets+stocks+investing+options/search.json?q={quote_plus(reddit_query)}&sort=new&limit=10"
        resp = requests.get(reddit_url, headers=headers, timeout=6)
        if resp.status_code == 200:
            children = resp.json().get("data", {}).get("children", [])
            for child in children:
                data = child.get("data", {})
                sub = data.get("subreddit", "Reddit")
                title = data.get("title", "").strip()
                permalink = data.get("permalink", "")
                ups = data.get("ups", 0)
                num_comments = data.get("num_comments", 0)
                if title:
                    _add_item({
                        "source": f"Reddit (r/{sub})",
                        "publisher": f"r/{sub}",
                        "title": title,
                        "sentiment": "neutral",
                        "link": f"https://reddit.com{permalink}" if permalink else f"https://reddit.com/r/{sub}",
                        "stats": f"{ups} upvotes, {num_comments} comments",
                    })
    except Exception as exc:
        logger.debug("Reddit fetch error: %s", exc)
        errors.append(f"Reddit: {exc}")

    # 3. Hacker News Tech & Market Discussions
    try:
        hn_url = f"https://hn.algolia.com/api/v1/search?query={quote_plus(topic)}&tags=story&hitsPerPage=8"
        resp = requests.get(hn_url, timeout=5)
        if resp.status_code == 200:
            for hit in resp.json().get("hits", []):
                title = hit.get("title") or hit.get("story_title") or ""
                object_id = hit.get("objectID")
                url = hit.get("url") or (f"https://news.ycombinator.com/item?id={object_id}" if object_id else "")
                points = hit.get("points", 0)
                if title:
                    _add_item({
                        "source": "Hacker News",
                        "publisher": "Hacker News",
                        "title": title,
                        "sentiment": "neutral",
                        "link": url or f"https://news.ycombinator.com/item?id={object_id}",
                        "stats": f"{points} points",
                    })
    except Exception as exc:
        logger.debug("HN fetch error: %s", exc)
        errors.append(f"HN: {exc}")

    return {
        "topic": topic,
        "ticker": ticker,
        "items": items[:limit],
        "error": None if items else ("; ".join(errors) or "No social signals returned."),
    }


def summarize_social(data: dict[str, Any], llm: LLMClient | None = None) -> dict[str, Any]:
    """Synthesize retail trader community discussions and sentiment distribution."""
    items = data.get("items", [])
    if not items:
        return {
            "summary": f"Social signals unavailable: {data.get('error') or 'no items returned'}.",
            "sentiment": "neutral",
            "raw": data,
        }

    titles = [item["title"] for item in items if item.get("title")]
    stocktwits_bulls = sum(1 for item in items if item.get("source") == "StockTwits" and str(item.get("sentiment")).lower() == "bullish")
    stocktwits_bears = sum(1 for item in items if item.get("source") == "StockTwits" and str(item.get("sentiment")).lower() == "bearish")

    joined = " | ".join(titles)
    sentiment = keyword_sentiment(joined)
    if stocktwits_bulls > stocktwits_bears:
        sentiment = "bullish"
    elif stocktwits_bears > stocktwits_bulls:
        sentiment = "bearish"

    deterministic = (
        f"Retail sentiment for {data['topic']} across {len(items)} discussion threads (StockTwits, Reddit, HN) leans {sentiment}. "
        f"Key topics: " + "; ".join(titles[:4])
    )
    if llm is None:
        return {"summary": trim(deterministic, 1000), "sentiment": sentiment, "raw": data}

    posts_formatted = []
    for item in items[:12]:
        src = item.get("publisher") or item.get("source", "Community")
        posts_formatted.append(f"[{src}] {item['title']}")

    prompt = (
        f"You are a Retail Sentiment & Flow Analyst evaluating market community chatter for {data['topic']}"
        + (f" ({data.get('ticker')})" if data.get("ticker") else "") + ".\n\n"
        f"Gathered Community Discussions ({len(items)} signals across StockTwits, Reddit, Hacker News):\n- "
        + "\n- ".join(posts_formatted) + "\n\n"
        "Instructions:\n"
        "1. Synthesize the main retail narrative, sentiment momentum, and retail positioning.\n"
        "2. Contrast bullish arguments vs bearish concerns raised by traders and retail investors.\n"
        "3. Conclude with an overall retail sentiment assessment (Bullish, Bearish, or Mixed/Neutral).\n"
        "4. Write in crisp, professional financial English without filler."
    )
    result = llm.generate(prompt, task="social_summary")
    return {"summary": result.text or deterministic, "sentiment": sentiment, "raw": data}
