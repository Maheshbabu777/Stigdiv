from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class BriefingState(TypedDict):
    user_query: str
    session_id: str
    intent: NotRequired[str]
    topic: NotRequired[str]
    ticker: NotRequired[str | None]
    agents: NotRequired[list[str]]
    use_llm: NotRequired[bool]
    use_live_data: NotRequired[bool]
    period: NotRequired[str]
    interval: NotRequired[str]
    news_summary: NotRequired[str]
    news_sentiment: NotRequired[str]
    news_sources: NotRequired[list[dict[str, Any]]]
    market_summary: NotRequired[str]
    market_trend: NotRequired[str]
    market_sources: NotRequired[list[dict[str, Any]]]
    social_summary: NotRequired[str]
    social_sentiment: NotRequired[str]
    social_sources: NotRequired[list[dict[str, Any]]]
    divergence_verdict: NotRequired[str]
    final_report: NotRequired[str]
    response: NotRequired[str]
    sources: NotRequired[dict[str, Any]]
    chart_data: NotRequired[dict[str, Any]]
    router_note: NotRequired[str | None]
    provider_used: NotRequired[str]
