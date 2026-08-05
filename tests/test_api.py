from fastapi.testclient import TestClient

from src.api.main import app
from src.storage.session_store import clear_session


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


class FakeGraph:
    def invoke(self, state):
        if "why" in state["user_query"].lower():
            return {
                "intent": "recall",
                "topic": "Tesla",
                "ticker": "TSLA",
                "divergence_verdict": "mixed",
                "response": "From the current session report, Tesla had a mixed verdict.",
                "sources": {},
            }
        return {
            "intent": "stock_research",
            "topic": "Tesla",
            "ticker": "TSLA",
            "divergence_verdict": "mixed",
            "response": "Signal Divergence Report: Tesla",
            "sources": {"news": [], "market": [], "social": []},
        }


def test_chat_research_and_recall_api_shape(monkeypatch):
    import src.api.main as main

    clear_session("api-test")
    monkeypatch.setattr(main, "graph_app", FakeGraph())

    research = client.post(
        "/chat",
        json={
            "message": "research TSLA",
            "session_id": "api-test",
            "use_llm": False,
        },
    )
    assert research.status_code == 200
    assert research.json()["ticker"] == "TSLA"
    assert "sources" in research.json()

    recall = client.post(
        "/chat",
        json={
            "message": "why was it divergent?",
            "session_id": "api-test",
            "use_llm": False,
        },
    )
    assert recall.status_code == 200
    assert recall.json()["intent"] == "recall"
    assert "current session report" in recall.json()["response"]


def test_chat_rejects_test_only_live_data_flag():
    response = client.post(
        "/chat",
        json={
            "message": "research TSLA",
            "session_id": "api-test",
            "use_llm": False,
            "use_live_data": False,
        },
    )

    assert response.status_code == 422
