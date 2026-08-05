from __future__ import annotations

import re
from datetime import datetime, timezone
from threading import Lock
from typing import Any

MAX_HISTORY = 20  # full messages to keep before triggering compression
COMPRESS_KEEP = 15  # messages retained after compression

_store: dict[str, list[dict[str, Any]]] = {}
_chat: dict[str, list[dict[str, Any]]] = {}
_lock = Lock()


# ---------------------------------------------------------------------------
# Report storage
# ---------------------------------------------------------------------------

def save_report(session_id: str, report: dict[str, Any]) -> dict[str, Any]:
    sid = session_id.strip()
    stored = dict(report)
    stored["created_at"] = datetime.now(timezone.utc).isoformat()
    with _lock:
        _store.setdefault(sid, []).append(stored)
    return stored


def get_reports(session_id: str) -> list[dict[str, Any]]:
    sid = session_id.strip()
    with _lock:
        return list(_store.get(sid, []))


def get_latest_report(session_id: str) -> dict[str, Any] | None:
    sid = session_id.strip()
    with _lock:
        reports = _store.get(sid, [])
        if not reports:
            return None
        return dict(reports[-1])


def find_report_for_query(session_id: str, query: str) -> dict[str, Any] | None:
    """Find the most relevant report in session matching any mentioned ticker/company name.
    
    Defaults to the latest report if no specific company is mentioned.
    """
    sid = session_id.strip()
    reports = get_reports(sid)
    if not reports:
        return None

    query_lower = query.lower()
    # Search backwards from newest to oldest for a specific topic or ticker match
    for report in reversed(reports):
        topic = str(report.get("topic") or "").lower()
        ticker = str(report.get("ticker") or "").lower()
        ticker_base = ticker.split(".")[0] if ticker else ""

        if ticker and (ticker in query_lower or (len(ticker_base) >= 2 and ticker_base in query_lower)):
            return dict(report)
        if topic and any(token in query_lower for token in topic.split() if len(token) >= 3):
            return dict(report)

    # If no specific company was referenced, return the most recent report
    return dict(reports[-1])


def format_all_reports_for_prompt(session_id: str) -> str:
    """Format all stored reports in this session as a concise context block for the LLM."""
    sid = session_id.strip()
    reports = get_reports(sid)
    if not reports:
        return "No previous reports in this session."

    lines: list[str] = []
    for idx, r in enumerate(reports, 1):
        topic = r.get("topic", "Unknown")
        ticker = r.get("ticker", "N/A")
        verdict = r.get("divergence_verdict", "N/A")
        m_trend = r.get("market_trend", "N/A")
        n_sent = r.get("news_sentiment", "N/A")
        s_sent = r.get("social_sentiment", "N/A")
        summary = r.get("final_report", "")[:250].replace("\n", " ")
        lines.append(
            f"- Report {idx}: {topic} ({ticker}) | Verdict: {verdict.upper()} | "
            f"Market: {m_trend}, News: {n_sent}, Social: {s_sent} | Summary: {summary}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat history — 15-20 message recall with compression
# ---------------------------------------------------------------------------

def add_message(session_id: str, role: str, content: str) -> None:
    """Append a message to the session chat history."""
    sid = session_id.strip()
    with _lock:
        history = _chat.setdefault(sid, [])
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


def get_chat_history(session_id: str) -> list[dict[str, Any]]:
    """Return a *copy* of the full chat history for this session."""
    sid = session_id.strip()
    with _lock:
        return list(_chat.get(sid, []))


def chat_length(session_id: str) -> int:
    sid = session_id.strip()
    with _lock:
        return len(_chat.get(sid, []))


def needs_compression(session_id: str) -> bool:
    sid = session_id.strip()
    return chat_length(sid) > MAX_HISTORY


def compress_chat_history(session_id: str, summary: str) -> None:
    """Replace older messages with a single summary, keeping the last COMPRESS_KEEP."""
    sid = session_id.strip()
    with _lock:
        history = _chat.get(sid, [])
        if len(history) <= COMPRESS_KEEP:
            return
        recent = history[-COMPRESS_KEEP:]
        compressed = {
            "role": "system",
            "content": f"[Compressed context from earlier messages]\n{summary}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _chat[sid] = [compressed] + recent


def build_compression_text(session_id: str) -> str:
    """Return the text of messages that would be compressed (everything before the last COMPRESS_KEEP)."""
    sid = session_id.strip()
    with _lock:
        history = _chat.get(sid, [])
        if len(history) <= COMPRESS_KEEP:
            return ""
        old = history[:-COMPRESS_KEEP]
    parts: list[str] = []
    for msg in old:
        prefix = "User" if msg["role"] == "user" else "Assistant" if msg["role"] == "assistant" else "System"
        parts.append(f"{prefix}: {msg['content'][:300]}")
    return "\n".join(parts)


def format_chat_for_prompt(session_id: str, limit: int = MAX_HISTORY) -> str:
    """Format chat history as a prompt-friendly string for LLM context."""
    sid = session_id.strip()
    history = get_chat_history(sid)
    if not history:
        return ""
    recent = history[-limit:]
    parts: list[str] = []
    for msg in recent:
        if msg["role"] == "system":
            parts.append(msg["content"])
        elif msg["role"] == "user":
            parts.append(f"User: {msg['content']}")
        else:
            text = msg["content"]
            if len(text) > 400:
                text = text[:400] + "..."
            parts.append(f"Assistant: {text}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Session lifecycle & Deletion
# ---------------------------------------------------------------------------

def clear_session(session_id: str) -> int:
    """Completely delete and purge all reports and chat history for a session."""
    sid = session_id.strip()
    with _lock:
        removed_reports = len(_store.get(sid, []))
        _store.pop(sid, None)
        _chat.pop(sid, None)
    return removed_reports


def active_session_count() -> int:
    """Return total number of currently active sessions in memory."""
    with _lock:
        all_keys = set(_store.keys()) | set(_chat.keys())
        return len(all_keys)
