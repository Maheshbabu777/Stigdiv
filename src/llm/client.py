from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from src.config import Settings, get_settings


logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    used_fallback: bool = False


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def generate(self, prompt: str, *, system: str = "", task: str = "general") -> LLMResult:
        errors: list[str] = []
        for provider in self.settings.provider_order:
            try:
                if provider == "groq" and self.settings.groq_api_key:
                    text = self._chat_completion(
                        url="https://api.groq.com/openai/v1/chat/completions",
                        api_key=self.settings.groq_api_key,
                        model=self.settings.groq_model,
                        prompt=prompt,
                        system=system,
                    )
                    return LLMResult(text=text, provider="groq", model=self.settings.groq_model)
                if provider == "gemini" and self.settings.gemini_api_key:
                    text = self._gemini(prompt=prompt, system=system)
                    return LLMResult(text=text, provider="gemini", model=self.settings.gemini_model)
                if provider == "openrouter" and self.settings.openrouter_api_key:
                    text = self._chat_completion(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        api_key=self.settings.openrouter_api_key,
                        model=self.settings.openrouter_model,
                        prompt=prompt,
                        system=system,
                        extra_headers={
                            "HTTP-Referer": "https://huggingface.co/spaces",
                            "X-Title": "Signal Divergence Agent",
                        },
                    )
                    return LLMResult(text=text, provider="openrouter", model=self.settings.openrouter_model)
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                logger.warning("LLM provider failed for task %s via %s: %s", task, provider, exc)

        fallback = rule_based_generate(prompt, task=task)
        if errors:
            logger.info("Using rule-based fallback after provider errors: %s", "; ".join(errors))
        return LLMResult(text=fallback, provider="rule_based", model="local", used_fallback=True)

    def _chat_completion(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        prompt: str,
        system: str,
        extra_headers: dict[str, str] | None = None,
    ) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = requests.post(
            url,
            headers=headers,
            json={"model": model, "messages": messages, "temperature": 0.2},
            timeout=self.settings.request_timeout_sec,
        )
        if response.status_code in {429, 500, 502, 503, 504}:
            raise LLMError(f"temporary provider error {response.status_code}")
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _gemini(self, *, prompt: str, system: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.gemini_model}:generateContent"
        )
        full_prompt = f"{system}\n\n{prompt}".strip()
        response = requests.post(
            url,
            params={"key": self.settings.gemini_api_key},
            json={"contents": [{"parts": [{"text": full_prompt}]}]},
            timeout=self.settings.request_timeout_sec,
        )
        if response.status_code in {429, 500, 502, 503, 504}:
            raise LLMError(f"temporary provider error {response.status_code}")
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def get_llm() -> LLMClient:
    return LLMClient()


def rule_based_generate(prompt: str, *, task: str = "general") -> str:
    lower = prompt.lower()
    if task == "router":
        recall_words = ("why", "explain", "previous", "earlier", "last report", "you said")
        if any(word in lower for word in recall_words):
            return '{"intent":"recall","topic":"","ticker":null}'
        return '{"intent":"new_research","topic":"","ticker":null}'
    if task == "supervisor":
        return (
            "Divergence verdict: mixed\n\n"
            "The model fallback could not perform deep reasoning, so this verdict is based on simple signal comparison. "
            "Review the news, market, and social sections together before making any decision."
        )
    return (
        "LLM providers were unavailable, so this section was generated with a local fallback. "
        "Use the fetched source data and labels as the primary evidence."
    )
