from __future__ import annotations

import re
from typing import Literal

from src.agents.common import extract_time_period, parse_json_object, resolve_stock_identity
from src.llm.client import get_llm
from src.storage.session_store import find_report_for_query, get_latest_report


def route_message(state: dict) -> dict:
    query = state["user_query"]
    session_id = state["session_id"]
    latest = get_latest_report(session_id)
    use_llm = state.get("use_llm", True)

    # 1. First check if it is a macro / commodity / asset class question (e.g. gold in india)
    if _is_macro_or_commodity_question(query):
        return {"intent": "general_chat", "topic": None, "ticker": None}

    # 2. Check if it is a clear recall question for existing reports
    if latest and _looks_like_recall_question(query):
        matching = find_report_for_query(session_id, query) or latest
        return {
            "intent": "recall",
            "topic": matching.get("topic"),
            "ticker": matching.get("ticker"),
        }

    # 3. Use LLM or rule-based tool decision
    if use_llm:
        decision = _llm_tool_decision(query, latest)
    else:
        decision = _fallback_tool_decision(query, latest)

    action = decision.get("action", "chat")
    target = decision.get("target") or query
    agents = decision.get("agents") or ["news", "market", "social"]

    if action == "recall":
        if latest:
            matching = find_report_for_query(session_id, query) or latest
            return {
                "intent": "recall",
                "topic": matching.get("topic"),
                "ticker": matching.get("ticker"),
            }
        return {
            "intent": "clarify",
            "topic": None,
            "ticker": None,
            "router_note": "I do not have a previous research report in this session yet.",
        }

    if action == "stock_research":
        if _is_generic_non_company_target(str(target)):
            return {"intent": "general_chat", "topic": None, "ticker": None}
        identity = resolve_stock_identity(str(target))
        topic = identity["topic"] or str(target)
        ticker = identity["ticker"]
        period, interval = extract_time_period(query)
        if ticker:
            return {
                "intent": "stock_research",
                "topic": topic,
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "agents": agents,
            }
        # If no ticker found and it is not explicitly marked unlisted, check if it's a general question
        if not identity.get("note") and len(str(target).split()) > 2:
            return {"intent": "general_chat", "topic": None, "ticker": None}

        note = identity.get("note") or f"I could not identify a listed company or ticker from '{target}'."
        return {
            "intent": "clarify",
            "topic": None,
            "ticker": None,
            "router_note": note,
        }

    if action == "clarify":
        return {"intent": "clarify", "topic": None, "ticker": None, "router_note": decision.get("reason")}

    return {"intent": "general_chat", "topic": None, "ticker": None}


def next_nodes(state: dict) -> Literal["recall", "general"] | list[str]:
    intent = state.get("intent")
    if intent == "recall":
        return "recall"
    if intent in {"general_chat", "clarify"}:
        return "general"
    agents = state.get("agents")
    if agents and isinstance(agents, list) and len(agents) > 0:
        valid_agents = [a for a in agents if a in {"news", "market", "social"}]
        if valid_agents:
            return valid_agents
    return ["news", "market", "social"]


def _is_macro_or_commodity_question(query: str) -> bool:
    lower = query.lower().strip()
    commodities = (
        "gold", "silver", "crude oil", "commodity", "commodities", "crypto",
        "cryptocurrency", "bitcoin", "btc", "ethereum", "eth", "real estate",
        "mutual fund", "mutual funds", "sgb", "sovereign gold bond", "fixed deposit",
        "fd", "index fund", "index funds", "ppf", "nps", "bonds", "gold etf",
    )
    # If the user specifically mentions a ticker or listed company alongside, let stock research handle it
    if re.search(r"\b(titan|kalyan|muthoot|barrick|newmont)\b", lower):
        return False
    return any(re.search(rf"\b{re.escape(c)}\b", lower) for c in commodities)


def _llm_tool_decision(query: str, latest: dict | None) -> dict:
    prompt = (
        "You are the Intelligent Orchestrator for a multi-agent financial intelligence assistant.\n"
        "You have the following specialized agents and capabilities:\n"
        "- news: Live news crawler (Firecrawl & financial media) for breaking news, earnings catalysts, announcements, and sector headlines.\n"
        "- market: Real-time price action & historical OHLCV chart analysis (yfinance) for price movements, technical trends, and charts.\n"
        "- social: Retail investor sentiment analyzer (Reddit & StockTwits) for community buzz, hype, and social sentiment.\n"
        "- recall: Session memory recall for answering follow-up questions about previously analyzed reports in this session.\n"
        "- chat: Direct conversational analysis for macroeconomic questions, asset classes (gold, bonds, crypto), educational questions, or greetings.\n"
        "- clarify: Asks user for clarification if request is ambiguous or company is not found.\n\n"
        "Task: Analyze the user's message and select the exact action, target, and agent(s) to deploy.\n\n"
        "Decision Guidelines:\n"
        "1. Targeted Requests:\n"
        "   - If user asks specifically for news/catalysts/announcements (e.g. 'What is the news on Tesla?', 'Any announcements from Tata Motors?'): action='stock_research', agents=['news'].\n"
        "   - If user asks specifically for stock price/chart/performance (e.g. 'Show TSLA price for 1 month', 'How did Apple perform today?'): action='stock_research', agents=['market'].\n"
        "   - If user asks specifically for social sentiment/Reddit/retail chatter (e.g. 'What are people on Reddit saying about GameStop?'): action='stock_research', agents=['social'].\n"
        "2. Comprehensive Stock Research & Divergence Analysis:\n"
        "   - If user asks to research/analyze a company or check signal divergence (e.g. 'Research TSLA', 'Analyze Ola Electric', 'Tata Motors'): action='stock_research', agents=['news', 'market', 'social'].\n"
        "3. Session Follow-up Questions:\n"
        "   - If user asks follow-up questions referencing previous analysis (e.g. 'Why was it divergent?', 'Explain the news score'): action='recall'.\n"
        "   - If user asks hypothetical investment, profit, or backtest questions on the previous stock (e.g. 'if i have invested in this around 2016 with $500 how much it would be now?', 'what if i invested 1000 in 2020?'): action='recall'.\n"
        "4. Macro / Commodity / Conceptual Questions:\n"
        "   - If user asks about gold, bonds, crypto, real estate, macroeconomics, or general investing: action='chat'.\n"
        "5. Session Timeframe / Follow-up Data Requests:\n"
        "   - If user asks to pull, fetch, or show data/charts for a different timeframe (e.g. 'can you pull last year's to till date data', 'show 1 year chart', 'pull 6 month data') without specifying a new company, and a previous report exists: action='stock_research', target=previous report topic/ticker, agents=['market'].\n\n"
        "Return JSON only with keys: action, target, agents, reason.\n"
        f"Has previous report: {bool(latest)}\n"
        f"Previous report topic: {latest.get('topic') if latest else ''}\n"
        f"User message: {query}"
    )
    result = get_llm().generate(prompt, task="tool_decision")
    parsed = parse_json_object(result.text)
    if parsed.get("action") in {"chat", "recall", "stock_research", "clarify"}:
        if not parsed.get("agents"):
            parsed["agents"] = ["news", "market", "social"]
        return parsed
    return _fallback_tool_decision(query, latest)


def _fallback_tool_decision(query: str, latest: dict | None) -> dict:
    if _is_macro_or_commodity_question(query):
        return {"action": "chat", "target": None, "reason": "macro/commodity question"}

    if latest and _looks_like_timeframe_or_pull_request(query):
        target = latest.get("ticker") or latest.get("topic")
        return {
            "action": "stock_research",
            "target": target,
            "agents": ["market"],
            "reason": "timeframe/data request referencing previous session stock",
        }

    if latest and _looks_like_recall_question(query):
        return {"action": "recall", "target": latest.get("topic"), "reason": "follow-up question"}

    if _looks_like_general_question(query):
        return {"action": "chat", "target": None, "reason": "general question"}

    # Targeted single-agent requests
    lower = query.lower()
    if _looks_like_news_only_request(query):
        target = _clean_topic(query)
        if target:
            return {"action": "stock_research", "target": target, "agents": ["news"], "reason": "news-only request"}

    if _looks_like_market_only_request(query):
        target = _clean_topic(query)
        if target:
            return {"action": "stock_research", "target": target, "agents": ["market"], "reason": "market-price request"}

    if _looks_like_social_only_request(query):
        target = _clean_topic(query)
        if target:
            return {"action": "stock_research", "target": target, "agents": ["social"], "reason": "social-sentiment request"}

    if _looks_like_market_request(query):
        target = _clean_topic(query)
        if target:
            return {"action": "stock_research", "target": target, "agents": ["news", "market", "social"], "reason": "market research request"}
        return {"action": "clarify", "target": None, "reason": "market research requested without a clear company"}

    if _looks_like_possible_company_or_ticker(query):
        return {"action": "stock_research", "target": query, "agents": ["news", "market", "social"], "reason": "short company or ticker-like message"}

    if lower.startswith(("why ", "how ", "explain ", "should i ", "is it ")):
        return {"action": "clarify", "target": None, "reason": "question needs context or a company first"}

    return {"action": "chat", "target": None, "reason": "ordinary chat"}


def _looks_like_timeframe_or_pull_request(query: str) -> bool:
    lower = query.lower()
    time_terms = (
        "last year", "past year", "till date", "to date", "ytd", "last month", "past month",
        "last week", "past week", "6 month", "3 month", "1 year", "2 year", "5 year",
        "pull", "fetch", "chart", "timeline", "timeframe", "historical"
    )
    data_words = ("data", "chart", "pull", "fetch", "show", "prices", "price")
    has_time = any(t in lower for t in time_terms)
    has_data = any(w in lower for w in data_words)
    return has_time and has_data


def _looks_like_news_only_request(query: str) -> bool:
    lower = query.lower()
    return any(p in lower for p in ("news on", "headlines for", "announcements from", "latest news", "breaking news", "press release"))


def _looks_like_market_only_request(query: str) -> bool:
    lower = query.lower()
    return any(p in lower for p in ("price of", "stock price", "chart for", "price trend", "performance for", "how did it perform"))


def _looks_like_social_only_request(query: str) -> bool:
    lower = query.lower()
    return any(p in lower for p in ("reddit", "stocktwits", "social sentiment", "retail sentiment", "what are people saying", "what is retail saying"))


def _looks_like_investment_calculation(query: str) -> tuple[int | None, float | None]:
    lower = query.lower()
    has_invest = any(w in lower for w in ("invested", "bought", "invest", "buy", "holding", "put in", "allocated", "investment"))
    has_return = any(w in lower for w in ("how much", "what would", "value now", "worth now", "be now", "return", "returns", "profit", "grew to", "growth", "become"))
    if not (has_invest and has_return):
        return None, None

    year_match = re.search(r"\b(19\d\d|20[0-2]\d)\b", query)
    year = int(year_match.group(1)) if year_match else None

    # Amount extraction
    amount = 500.0
    amount_match = re.search(r"(\$|₹|rs\.?|usd)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(\$|₹|rs\.?|usd)?", query, re.I)
    if amount_match:
        raw_val = amount_match.group(2).replace(",", "")
        try:
            val = float(raw_val)
            if val > 0 and (val != year or not year_match):
                amount = val
        except ValueError:
            pass

    return year, amount


def _looks_like_recall_question(query: str) -> bool:
    lower = query.lower().strip()
    followup_words = (
        "previous",
        "earlier",
        "last report",
        "you said",
        "that report",
        "this report",
        "in this",
        "about this",
        "for this",
        "of this",
        "invested in this",
        "if i invested",
        "if i had invested",
        "if i have invested",
        "if i bought",
        "if i had bought",
        "how much would it be",
        "how much it would be",
        "what would it be",
        "what would it be worth",
        "worth now",
        "why was it",
        "why is it",
        "why is there",
        "why there is",
        "why was that",
        "why is that",
        "why is there a divergence",
        "why is there divergence",
        "why the divergence",
        "why divergent",
        "why was it divergent",
        "why is it divergent",
        "explain that",
        "explain it",
        "explain the divergence",
        "should i buy",
        "should i sell",
        "should i hold",
        "good time to buy",
        "bad time to buy",
    )
    research_words = (
        "research", "analyze", "analyse", "look up", "fetch", "pull", "get data", "pull data",
        "show data", "show chart", "chart", "last year", "past year", "till date", "to date", "ytd",
        "1 year", "6 month", "3 month", "1 month", "now research"
    )
    if any(word in lower for word in research_words):
        return False
    if any(word in lower for word in followup_words):
        return True
    
    # Check if hypothetical investment query
    year, _ = _looks_like_investment_calculation(query)
    if year is not None:
        return True

    if lower.startswith(("why ", "how ", "explain ")) and any(
        word in lower for word in ("divergence", "divergent", "signal", "signals", "buy", "sell", "hold", "much", "worth")
    ):
        return True
    return False


def _looks_like_general_question(query: str) -> bool:
    lower = query.lower().strip()
    general_starts = (
        "hi",
        "hello",
        "hey",
        "what can you do",
        "who are you",
        "what is your name",
        "help",
        "thanks",
        "thank you",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "greetings",
    )
    if lower in general_starts or any(lower == g or lower.startswith(f"{g} ") for g in general_starts) or lower.startswith(("what is divergence", "explain divergence")):
        return True
    return False


def _looks_like_market_request(query: str) -> bool:
    lower = query.lower()
    tool_words = (
        "research",
        "analyze",
        "analyse",
        "stock",
        "ticker",
        "share",
        "shares",
        "market",
        "price",
        "news",
        "social sentiment",
        "signals",
        "divergence report",
        "earnings",
        "financials",
        "valuation",
        "latest on",
        "what is happening with",
    )
    return any(word in lower for word in tool_words)


def _looks_like_possible_company_or_ticker(query: str) -> bool:
    cleaned = _clean_topic(query)
    if not cleaned:
        return False
    words = cleaned.split()
    if len(words) > 3:
        return False
    generic = {
        "divergence",
        "market",
        "stock",
        "stocks",
        "signal",
        "signals",
        "price",
        "buy",
        "sell",
        "hello",
        "help",
    }
    return cleaned.lower() not in generic


def _is_generic_non_company_target(target: str) -> bool:
    return _clean_topic(target).lower() in {
        "",
        "divergence",
        "divergent",
        "signal",
        "signals",
        "market",
        "markets",
        "stock",
        "stocks",
        "price",
        "prices",
        "buy",
        "sell",
        "hold",
        "invest",
        "investment",
    }


def _clean_topic(query: str) -> str:
    phrases = [
        "what is happening with",
        "what's happening with",
        "social sentiment on",
        "retail sentiment on",
        "announcements from",
        "tell me about",
        "headlines for",
        "what are the",
        "what is the",
        "price trend",
        "sentiment on",
        "divergence",
        "stocktwits",
        "currently",
        "sentiment",
        "latest on",
        "chart for",
        "price of",
        "what are",
        "what is",
        "analyze",
        "analyse",
        "research",
        "news on",
        "current",
        "tell me",
        "show me",
        "signals",
        "stocks",
        "shares",
        "ticker",
        "reddit",
        "report",
        "recent",
        "please",
        "signal",
        "how is",
        "latest",
        "shares",
        "today",
        "check",
        "about",
        "stock",
        "do a",
        "now",
        "for",
        "do",
    ]
    pattern = r"\b(" + "|".join(re.escape(p) for p in phrases) + r")\b"
    cleaned = re.sub(pattern, "", query, flags=re.I)
    return re.sub(r"\s+", " ", cleaned).strip(" ?.,")
