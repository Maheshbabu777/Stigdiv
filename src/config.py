from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    gemini_api_key: str | None
    openrouter_api_key: str | None
    provider_order: tuple[str, ...]
    groq_model: str
    gemini_model: str
    openrouter_model: str
    request_timeout_sec: int
    firecrawl_api_key: str | None = None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_settings() -> Settings:
    order = os.getenv("LLM_PROVIDER_ORDER", "groq,gemini,openrouter")
    return Settings(
        groq_api_key=_clean(os.getenv("GROQ_API_KEY")),
        gemini_api_key=_clean(os.getenv("GEMINI_API_KEY")),
        openrouter_api_key=_clean(os.getenv("OPENROUTER_API_KEY")),
        firecrawl_api_key=_clean(os.getenv("FIRECRAWL_API_KEY")),
        provider_order=tuple(part.strip().lower() for part in order.split(",") if part.strip()),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        request_timeout_sec=int(os.getenv("LLM_TIMEOUT_SEC", "30")),
    )
