from __future__ import annotations

from src.llm.client import LLMClient


def classify_divergence(news_sentiment: str, market_trend: str, social_sentiment: str) -> str:
    if market_trend == "unavailable":
        if news_sentiment == social_sentiment:
            return "mixed"
        if "bullish" in {news_sentiment, social_sentiment} and "bearish" in {news_sentiment, social_sentiment}:
            return "divergent"
        return "mixed"
    market_signal = {"up": "bullish", "down": "bearish", "flat": "neutral"}.get(market_trend, "neutral")
    signals = [news_sentiment, market_signal, social_sentiment]
    if len(set(signals)) == 1:
        return "aligned"
    if "bullish" in signals and "bearish" in signals:
        return "divergent"
    return "mixed"


def build_final_report(
    *,
    topic: str,
    ticker: str | None,
    news_summary: str = "",
    news_sentiment: str = "neutral",
    market_summary: str = "",
    market_trend: str = "unavailable",
    social_summary: str = "",
    social_sentiment: str = "neutral",
    llm: LLMClient | None = None,
) -> dict[str, str]:
    has_news = bool(news_summary and "unavailable" not in news_summary.lower())
    has_market = bool(market_summary and "unavailable" not in market_summary.lower() and market_trend != "unavailable")
    has_social = bool(social_summary and "unavailable" not in social_summary.lower())

    active_count = sum([has_news, has_market, has_social])

    # Specialized deterministic reports if LLM is unavailable
    if active_count <= 1:
        if has_news and not has_market and not has_social:
            verdict = "news_briefing"
            deterministic = (
                f"### News Intelligence Briefing: {topic}" + (f" (`{ticker}`)" if ticker else "") + "\n\n"
                f"- **Media & News Sentiment ({news_sentiment.title()}):** {news_summary}\n\n"
                "> *Note: Intelligence compiled from multi-source institutional and financial media.*"
            )
        elif has_market and not has_news and not has_social:
            verdict = "market_analysis"
            deterministic = (
                f"### Market Price Action & Valuation: {topic}" + (f" (`{ticker}`)" if ticker else "") + "\n\n"
                f"- **Price Action ({market_trend.title()}):** {market_summary}\n\n"
                "> *Note: Real-time price action and quantitative fundamentals.*"
            )
        elif has_social and not has_news and not has_market:
            verdict = "social_sentiment"
            deterministic = (
                f"### Community & Retail Sentiment: {topic}" + (f" (`{ticker}`)" if ticker else "") + "\n\n"
                f"- **Retail Sentiment ({social_sentiment.title()}):** {social_summary}\n\n"
                "> *Note: Real-time trader discussion from StockTwits, Reddit, and community feeds.*"
            )
        else:
            verdict = classify_divergence(news_sentiment, market_trend, social_sentiment)
            deterministic = (
                f"### Comprehensive Research: {topic}" + (f" (`{ticker}`)" if ticker else "") + "\n\n"
                f"- **Market & Valuation:** {market_summary}\n"
                f"- **News Catalysts:** {news_summary}\n"
                f"- **Community Sentiment:** {social_summary}\n"
            )
    else:
        verdict = classify_divergence(news_sentiment, market_trend, social_sentiment)
        deterministic = (
            f"### Multi-Source Signal Analysis: {topic}" + (f" (`{ticker}`)" if ticker else "") + "\n\n"
            f"**Signal Alignment:** `{verdict.upper()}`\n\n"
            f"- **📈 Market & Valuation ({market_trend.title()}):** {market_summary}\n"
            f"- **📰 Institutional News ({news_sentiment.title()}):** {news_summary}\n"
            f"- **💬 Retail & Community ({social_sentiment.title()}):** {social_summary}\n\n"
            f"**Assessment:** Price action exhibits a **{market_trend}** bias, with institutional news registering as **{news_sentiment}** "
            f"and retail chatter reflecting **{social_sentiment}** tone.\n\n"
            "> *Note: This multi-source research report is for informational purposes and does not constitute investment advice.*"
        )

    if llm is None:
        return {"divergence_verdict": verdict, "final_report": deterministic}

    active_sections = []
    if has_market:
        active_sections.append(f"### Market Price Action & Fundamentals [{market_trend}]:\n{market_summary}")
    if has_news:
        active_sections.append(f"### Institutional News & Catalysts [{news_sentiment}]:\n{news_summary}")
    if has_social:
        active_sections.append(f"### Retail Flow & Community Sentiment [{social_sentiment}]:\n{social_summary}")

    signals_text = "\n\n".join(active_sections) if active_sections else (
        f"Market Price Action:\n{market_summary}\n\nNews Catalysts:\n{news_summary}\n\nRetail Sentiment:\n{social_summary}"
    )

    prompt = (
        "You are an elite senior equity research analyst at a top-tier institutional investment firm.\n"
        "Conduct a deep, rigorous, multi-faceted equity research report in elegant, professional plain English.\n\n"
        f"Asset Under Analysis: {topic}" + (f" ({ticker})" if ticker else "") + "\n"
        f"Signal Divergence Assessment: {verdict.upper()}\n\n"
        f"GATHERED DEEP RESEARCH INTELLIGENCE:\n{signals_text}\n\n"
        "CRITICAL EDITORIAL GUIDELINES:\n"
        "1. Write in natural, authoritative, institutional-grade financial English.\n"
        "2. Structure your report into clear, well-formatted Markdown sections:\n"
        "   - **Executive Summary & Signal Alignment**: Synthesize whether price action, fundamentals, news, and retail flow are aligned or divergent.\n"
        "   - **Price Momentum & Valuation**: Analyze price levels, period performance, valuation multiples, and analyst consensus.\n"
        "   - **Key Catalysts & Institutional News**: Detail the core fundamental developments, earnings drivers, or regulatory events.\n"
        "   - **Retail Positioning & Sentiment**: Contrast retail trader positioning against institutional fundamentals.\n"
        "   - **Risk Factors & Strategic Takeaway**: Key upside catalysts and downside risks.\n"
        "3. NEVER use robotic internal jargon like 'Market Price Movement agent signal', 'agent reports', or 'Lead Supervisor'.\n"
        "4. DO NOT include greetings (such as 'Hello') or filler preamble. Jump immediately into the analysis.\n"
        "5. Conclude with a brief reminder that this is research intelligence and not financial advice."
    )
    result = llm.generate(prompt, task="supervisor")
    return {"divergence_verdict": verdict, "final_report": result.text or deterministic}
