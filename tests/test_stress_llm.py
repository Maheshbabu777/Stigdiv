import pytest
from src.llm.client import LLMClient, LLMResult, rule_based_generate
from src.config import Settings

def test_rule_based_router_recall():
    res = rule_based_generate("why did you say that?", task="router")
    assert "recall" in res

def test_rule_based_router_research():
    res = rule_based_generate("analyze AAPL", task="router")
    assert "new_research" in res

def test_rule_based_supervisor():
    res = rule_based_generate("...", task="supervisor")
    assert "mixed" in res

def test_rule_based_unknown():
    res = rule_based_generate("...", task="unknown")
    assert "local fallback" in res

def test_llm_result_dataclass():
    res = LLMResult(text="hello", provider="groq", model="test_model", used_fallback=True)
    assert res.text == "hello"
    assert res.provider == "groq"
    assert res.model == "test_model"
    assert res.used_fallback is True

def test_llm_client_no_keys():
    settings = Settings(
        groq_api_key=None, gemini_api_key=None, openrouter_api_key=None,
        provider_order=('groq', 'gemini', 'openrouter'),
        groq_model='test', gemini_model='test', openrouter_model='test',
        request_timeout_sec=5,
    )
    client = LLMClient(settings)
    result = client.generate('test prompt')
    assert result.used_fallback is True

def test_llm_client_empty_provider():
    settings = Settings(
        groq_api_key='fake', gemini_api_key=None, openrouter_api_key=None,
        provider_order=(),
        groq_model='test', gemini_model='test', openrouter_model='test',
        request_timeout_sec=5,
    )
    client = LLMClient(settings)
    result = client.generate('test')
    assert result.used_fallback is True
