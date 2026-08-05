from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


def fetch_market(ticker: str | None, period: str = "5d", interval: str = "1d") -> dict[str, Any]:
    """Fetch deep market OHLCV price action, moving averages, and fundamental valuation metrics."""
    if not ticker:
        return {
            "ticker": None,
            "rows": [],
            "period": period,
            "interval": interval,
            "fundamentals": {},
            "error": "No ticker detected.",
        }

    try:
        import yfinance as yf

        cache_dir = Path(".cache") / "yfinance"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))

        ticker_obj = yf.Ticker(ticker)
        history = ticker_obj.history(period=period, interval=interval)
        if history.empty:
            return {
                "ticker": ticker,
                "rows": [],
                "period": period,
                "interval": interval,
                "fundamentals": {},
                "error": "No market data returned.",
            }

        rows = []
        for index, row in history.iterrows():
            date_str = str(index.date()) if hasattr(index, "date") else str(index)
            rows.append(
                {
                    "date": date_str,
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                }
            )

        # Extract rich fundamentals and valuation snapshot
        fundamentals: dict[str, Any] = {}
        try:
            info = getattr(ticker_obj, "fast_info", None)
            if info:
                market_cap = getattr(info, "market_cap", None)
                if market_cap:
                    if market_cap >= 1e12:
                        fundamentals["market_cap"] = f"${market_cap / 1e12:.2f}T"
                    elif market_cap >= 1e9:
                        fundamentals["market_cap"] = f"${market_cap / 1e9:.2f}B"
                    else:
                        fundamentals["market_cap"] = f"${market_cap / 1e6:.2f}M"
                fundamentals["fifty_two_week_high"] = getattr(info, "year_high", None)
                fundamentals["fifty_two_week_low"] = getattr(info, "year_low", None)
                fundamentals["fifty_day_average"] = getattr(info, "fifty_day_average", None)
                fundamentals["two_hundred_day_average"] = getattr(info, "two_hundred_day_average", None)
                fundamentals["currency"] = getattr(info, "currency", "USD")

            # Deep info dictionary for PE and Analyst Target
            full_info = getattr(ticker_obj, "info", {})
            if isinstance(full_info, dict) and full_info:
                if full_info.get("trailingPE"):
                    fundamentals["pe_ratio"] = round(float(full_info["trailingPE"]), 2)
                if full_info.get("forwardPE"):
                    fundamentals["forward_pe"] = round(float(full_info["forwardPE"]), 2)
                if full_info.get("targetMeanPrice"):
                    fundamentals["analyst_target"] = round(float(full_info["targetMeanPrice"]), 2)
                if full_info.get("recommendationKey"):
                    fundamentals["analyst_consensus"] = str(full_info["recommendationKey"]).upper()
                if full_info.get("revenueGrowth"):
                    fundamentals["revenue_growth"] = f"{float(full_info['revenueGrowth']) * 100:.1f}%"
                if full_info.get("profitMargins"):
                    fundamentals["profit_margins"] = f"{float(full_info['profitMargins']) * 100:.1f}%"
        except Exception as info_exc:
            logger.debug("Fundamentals extraction error for %s: %s", ticker, info_exc)

        return {
            "ticker": ticker.upper(),
            "rows": rows,
            "period": period,
            "interval": interval,
            "fundamentals": fundamentals,
            "error": None,
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "rows": [],
            "period": period,
            "interval": interval,
            "fundamentals": {},
            "error": str(exc),
        }


def market_trend(rows: list[dict[str, Any]]) -> str:
    if len(rows) < 2:
        return "unavailable"
    first = float(rows[0]["close"])
    last = float(rows[-1]["close"])
    change_pct = ((last - first) / first) * 100 if first else 0
    if change_pct > 1:
        return "up"
    if change_pct < -1:
        return "down"
    return "flat"


def summarize_market(data: dict[str, Any], llm: LLMClient | None = None) -> dict[str, Any]:
    """Synthesize market performance, timeframe return, and valuation metrics."""
    rows = data.get("rows", [])
    trend = market_trend(rows)
    fundamentals = data.get("fundamentals", {})
    if not rows:
        return {
            "summary": f"Market data unavailable: {data.get('error') or 'no rows returned'}.",
            "trend": "unavailable",
            "raw": data,
        }

    first = float(rows[0].get("close", 0))
    last = float(rows[-1].get("close", 0))
    high = max(float(r.get("high", r.get("close", 0))) for r in rows)
    low = min(float(r.get("low", r.get("close", 0))) for r in rows)
    change_pct = ((last - first) / first) * 100 if first else 0
    period = str(data.get("period", "5d")).upper()

    fund_details = []
    if fundamentals.get("market_cap"):
        fund_details.append(f"Market Cap: {fundamentals['market_cap']}")
    if fundamentals.get("pe_ratio"):
        fund_details.append(f"P/E Ratio: {fundamentals['pe_ratio']}")
    if fundamentals.get("analyst_target"):
        fund_details.append(f"Analyst Consensus Target: ${fundamentals['analyst_target']}")
    if fundamentals.get("analyst_consensus"):
        fund_details.append(f"Consensus: {fundamentals['analyst_consensus']}")

    deterministic = (
        f"{data['ticker']} closed at ${last:.2f}, moving {change_pct:+.2f}% over the {period} window "
        f"(High: ${high:.2f}, Low: ${low:.2f}). Overall technical trend is {trend}. "
        + (" | ".join(fund_details) if fund_details else "")
    )
    if llm is None:
        return {"summary": deterministic, "trend": trend, "raw": data}

    prompt = (
        f"You are a Senior Quantitative & Market Analyst evaluating price action and fundamentals for {data['ticker']}.\n\n"
        f"Price Action Overview ({period} Window):\n"
        f"- Current Price: ${last:.2f}\n"
        f"- Timeframe Change: {change_pct:+.2f}%\n"
        f"- Period Range: High ${high:.2f} | Low ${low:.2f}\n"
        f"- Trend Signal: {trend.upper()}\n"
        + (f"Key Valuation & Fundamentals:\n- " + "\n- ".join(fund_details) + "\n\n" if fund_details else "\n")
        + "Instructions:\n"
        "1. Write a professional, data-dense price momentum and valuation summary (2-3 concise paragraphs).\n"
        "2. Analyze current price levels relative to moving averages and historical range.\n"
        "3. Highlight key support/resistance levels and analyst consensus.\n"
        "4. Write in crisp, professional financial English without filler or preamble."
    )
    result = llm.generate(prompt, task="market_summary")
    return {"summary": result.text or deterministic, "trend": trend, "raw": data}
