from __future__ import annotations

import re

from src.agents.market_agent import fetch_market, summarize_market
from src.agents.news_agent import fetch_news, summarize_news
from src.agents.social_agent import fetch_social, summarize_social
from src.agents.supervisor_agent import build_final_report
from src.llm.client import get_llm
from src.storage.session_store import (
    add_message,
    build_compression_text,
    compress_chat_history,
    find_report_for_query,
    format_all_reports_for_prompt,
    format_chat_for_prompt,
    get_latest_report,
    needs_compression,
    save_report,
)


def _llm_or_none(state: dict):
    return get_llm() if state.get("use_llm", True) else None


def _record_turn_and_compress(session_id: str, query: str, response: str, llm=None) -> None:
    """Record user and assistant messages in session history and compress if exceeding threshold."""
    if not session_id:
        return
    add_message(session_id, "user", query)
    add_message(session_id, "assistant", response)
    if needs_compression(session_id):
        old_text = build_compression_text(session_id)
        if old_text:
            if llm:
                summary = llm.generate(
                    "Summarize these earlier conversation turns into 2-3 key bullet points preserving stock tickers and findings:\n\n"
                    f"{old_text}",
                    task="general",
                ).text
            else:
                summary = "Earlier discussion covered previous stock queries and signal divergence analysis."
            compress_chat_history(session_id, summary)


def _is_simple_identity_or_greeting(query: str) -> bool:
    lower = query.lower().strip(" ?.,!").strip()
    greeting_words = ("hi", "hello", "hey", "howdy", "greetings", "good morning", "good evening", "good afternoon")
    identity_phrases = (
        "who are you",
        "what are you",
        "what is your name",
        "what can you do",
        "what do you do",
        "help",
        "who created you",
        "your capabilities",
    )
    if any(lower.startswith(g) or lower == g for g in greeting_words):
        return True
    if any(phrase in lower for phrase in identity_phrases):
        return True
    return False


def _extract_clean_search_topic(query: str) -> str:
    cleaned = re.sub(
        r"\b(should i|can i|is it good to|how to|what is|what are|tell me about|explain|recommend|buy|sell|invest in|go for|any|please|currently|today)\b",
        "",
        query,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.,")
    return cleaned or query


def general_node(state: dict) -> dict:
    note = state.get("router_note")
    query = state["user_query"]
    session_id = state.get("session_id", "demo")
    llm = _llm_or_none(state)
    sources = {}

    # Deploy live News & Web Intelligence agent dynamically on topical/macro queries
    live_news_context = ""
    is_identity = _is_simple_identity_or_greeting(query)

    if not note and not is_identity and state.get("use_live_data", True):
        topic = _extract_clean_search_topic(query)
        if topic and len(topic) >= 3:
            news_data = fetch_news(topic, limit=4)
            items = news_data.get("items", [])
            if items:
                sources["news"] = items
                live_snippets = [
                    f"- [{item.get('source', 'Web')}] {item['title']}: {item.get('snippet', '')}".strip()
                    for item in items[:4]
                    if item.get("title")
                ]
                if live_snippets:
                    live_news_context = "Current Live Financial Intelligence & Market News:\n" + "\n".join(live_snippets)

    if note:
        response = (
            f"{note} Tell me the company or ticker you want me to check, and I can run the news, market, "
            "and social-signal agents."
        )
    elif state.get("use_llm", True):
        history_context = format_chat_for_prompt(session_id, limit=15)
        all_reports = format_all_reports_for_prompt(session_id)

        prompt = (
            "You are an elite AI financial intelligence and multi-agent market research assistant.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. NEVER start responses with greetings like 'Hello again', 'Hello, I am the Signal Divergence Agent', 'Welcome', or any preamble.\n"
            "2. Start your response directly with the core insight and analysis.\n"
            "3. If asked who you are or what you do, concisely explain that you are the Signal Divergence Agent equipped with multi-agent market, news, and social intelligence tools.\n"
            "4. For financial, macroeconomic, asset class, or bond/gold/crypto topics, provide a well-structured, comprehensive, and balanced analysis.\n"
            "5. Use the live news and intelligence context provided below to reference current yields, rates, or market developments where relevant.\n"
            "6. Use clear Markdown formatting with headers, bullet points, pros/cons, and risks.\n\n"
            f"{live_news_context}\n\n"
            f"Active Session Previous Reports:\n{all_reports}\n\n"
            f"Conversation History:\n{history_context}\n\n"
            f"User Question: {query}"
        )
        response = get_llm().generate(prompt, task="general_chat").text
    else:
        if is_identity:
            response = (
                "I am the **Signal Divergence Agent**, a specialized multi-agent stock research assistant. "
                "I analyze real-time market data, financial news, and social sentiment to detect signal divergence. "
                "You can ask me to research any listed company (e.g., 'research AAPL for 1 year', 'analyze NVDA'), "
                "or ask follow-up questions like 'why was it divergent?'."
            )
        else:
            response = (
                f"Analysis for: **{query}**\n\n"
                "When evaluating this financial decision, consider key factors such as risk tolerance, "
                "yield expectations, inflation hedging, and liquidity needs. "
                "For listed equities and multi-agent divergence intelligence, you can also ask for specific tickers."
            )

    _record_turn_and_compress(session_id, query, response, llm=llm)

    return {
        "intent": state.get("intent", "general_chat"),
        "topic": None,
        "ticker": None,
        "divergence_verdict": None,
        "response": response,
        "sources": sources,
    }


def news_node(state: dict) -> dict:
    if state.get("use_live_data", True):
        data = fetch_news(state["topic"], ticker=state.get("ticker"))
    else:
        data = {
            "topic": state["topic"],
            "ticker": state.get("ticker"),
            "items": [{"title": f"{state['topic']} shares gain after strong demand", "link": "", "published": ""}],
            "error": None,
        }
    result = summarize_news(data, llm=_llm_or_none(state))
    return {
        "news_summary": result["summary"],
        "news_sentiment": result["sentiment"],
        "news_sources": result["raw"].get("items", []),
    }


def market_node(state: dict) -> dict:
    period = state.get("period", "5d")
    interval = state.get("interval", "1d")
    if state.get("use_live_data", True):
        data = fetch_market(state.get("ticker"), period=period, interval=interval)
    else:
        data = {
            "ticker": state.get("ticker") or "TEST",
            "rows": [
                {"date": "2025-08-01", "open": 100, "high": 102, "low": 97, "close": 100, "volume": 100000},
                {"date": "2025-08-02", "open": 100, "high": 101, "low": 95, "close": 96, "volume": 120000},
            ],
            "period": period,
            "interval": interval,
            "error": None,
        }
    result = summarize_market(data, llm=_llm_or_none(state))
    return {
        "market_summary": result["summary"],
        "market_trend": result["trend"],
        "market_sources": result["raw"].get("rows", []),
    }


def social_node(state: dict) -> dict:
    if state.get("use_live_data", True):
        data = fetch_social(state["topic"], ticker=state.get("ticker"))
    else:
        data = {
            "topic": state["topic"],
            "ticker": state.get("ticker"),
            "items": [{"source": "stub", "title": f"{state['topic']} holders worry about near-term risk", "link": ""}],
            "error": None,
        }
    result = summarize_social(data, llm=_llm_or_none(state))
    return {
        "social_summary": result["summary"],
        "social_sentiment": result["sentiment"],
        "social_sources": result["raw"].get("items", []),
    }


def supervisor_node(state: dict) -> dict:
    result = build_final_report(
        topic=state["topic"],
        ticker=state.get("ticker"),
        news_summary=state.get("news_summary", "News signal unavailable."),
        news_sentiment=state.get("news_sentiment", "neutral"),
        market_summary=state.get("market_summary", "Market signal unavailable."),
        market_trend=state.get("market_trend", "flat"),
        social_summary=state.get("social_summary", "Social signal unavailable."),
        social_sentiment=state.get("social_sentiment", "neutral"),
        llm=_llm_or_none(state),
    )
    return {
        "divergence_verdict": result["divergence_verdict"],
        "final_report": result["final_report"],
        "response": result["final_report"],
    }


def save_node(state: dict) -> dict:
    sources = {
        "news": state.get("news_sources", []),
        "market": state.get("market_sources", []),
        "social": state.get("social_sources", []),
    }
    chart_data = {
        "ticker": state.get("ticker"),
        "period": state.get("period", "5d"),
        "interval": state.get("interval", "1d"),
        "rows": state.get("market_sources", []),
    }
    report = {
        "topic": state["topic"],
        "ticker": state.get("ticker"),
        "news_summary": state.get("news_summary"),
        "news_sentiment": state.get("news_sentiment"),
        "market_summary": state.get("market_summary"),
        "market_trend": state.get("market_trend"),
        "social_summary": state.get("social_summary"),
        "social_sentiment": state.get("social_sentiment"),
        "divergence_verdict": state.get("divergence_verdict"),
        "final_report": state.get("final_report"),
        "sources": sources,
        "chart_data": chart_data,
    }
    save_report(state["session_id"], report)
    _record_turn_and_compress(
        state["session_id"],
        state["user_query"],
        state.get("final_report", ""),
        llm=_llm_or_none(state),
    )
    return {"sources": sources, "chart_data": chart_data}


def recall_node(state: dict) -> dict:
    session_id = state["session_id"]
    question = state["user_query"]
    target_report = find_report_for_query(session_id, question) or get_latest_report(session_id)
    if target_report is None:
        resp = "No report is stored in this session yet. Ask me to research a company or ticker first."
        _record_turn_and_compress(session_id, question, resp, llm=_llm_or_none(state))
        return {
            "topic": None,
            "ticker": None,
            "divergence_verdict": None,
            "response": resp,
            "sources": {},
        }

    base = target_report.get("final_report", "")
    all_reports_context = format_all_reports_for_prompt(session_id)
    advice_note = ""
    if _asks_for_buy_or_sell(question):
        advice_note = (
            " I cannot tell you whether to buy or sell, but I can explain what the saved signals imply."
        )
    if state.get("use_llm", True):
        history_context = format_chat_for_prompt(session_id, limit=15)
        prompt = (
            "You are the Signal Divergence Agent answering a conversational follow-up question.\n"
            "Use the stored research report, gathered market/news/social data, and recent conversation history as your ground-truth context.\n\n"
            "CRITICAL FORMATTING RULES:\n"
            "- NEVER begin with 'Hello', 'Hello again', or boilerplate intros.\n"
            "- Dive straight into answering the user's question directly and conversationally.\n\n"
            f"All Researched Reports in this Session:\n{all_reports_context}\n\n"
            f"Primary Target Stored Report ({target_report.get('topic')} / {target_report.get('ticker')}):\n{base}\n\n"
            f"Detailed Stored Context Signals for {target_report.get('topic')}:\n"
            f"- Market Movement [{target_report.get('market_trend')}]: {target_report.get('market_summary')}\n"
            f"- News Sentiment [{target_report.get('news_sentiment')}]: {target_report.get('news_summary')}\n"
            f"- Social Sentiment [{target_report.get('social_sentiment')}]: {target_report.get('social_summary')}\n"
            f"- Divergence Verdict: {target_report.get('divergence_verdict')}\n\n"
            f"Recent Conversation History:\n{history_context}\n\n"
            f"User Follow-up: {question}\n\n"
            "Guidelines:\n"
            "- Answer the user's question directly, insightfully, and conversationally.\n"
            "- If the user asks about a specific company analyzed previously in this session, use its report.\n"
            "- Reference the specific context from the stored research and conversation.\n"
            "- Do not give financial advice."
        )
        response = get_llm().generate(prompt, task="recall").text
    else:
        response = (
            f"From the current session report, {target_report.get('topic', 'the last topic')} "
            f"had a {target_report.get('divergence_verdict', 'mixed')} verdict.{advice_note} "
            f"News was labeled {target_report.get('news_sentiment', 'neutral')}, "
            f"market was labeled {target_report.get('market_trend', 'unavailable')}, "
            f"and social was labeled {target_report.get('social_sentiment', 'neutral')}. "
            "That is why the saved report did not treat all signals as fully aligned."
        )

    _record_turn_and_compress(session_id, question, response, llm=_llm_or_none(state))

    return {
        "topic": target_report.get("topic"),
        "ticker": target_report.get("ticker"),
        "divergence_verdict": target_report.get("divergence_verdict"),
        "response": response,
        "sources": target_report.get("sources", {}),
    }


def _asks_for_buy_or_sell(question: str) -> bool:
    lower = question.lower()
    return any(phrase in lower for phrase in ("buy", "sell", "hold", "good time", "bad time", "invest"))
