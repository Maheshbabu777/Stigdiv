import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.storage.session_store import clear_session

@pytest.fixture(autouse=True)
def _clear_sessions():
    clear_session("test")
    clear_session("demo")
    yield
    clear_session("test")
    clear_session("demo")

client = TestClient(app)

class FakeGraph:
    def invoke(self, state):
        return {
            "intent": "stock_research",
            "topic": "Test",
            "ticker": "TEST",
            "divergence_verdict": "mixed",
            "response": "Test response",
            "sources": {"news": [], "market": [], "social": []},
            "chart_data": {"ticker": "TEST", "period": "5d", "interval": "1d", "rows": []},
        }

@pytest.fixture
def mock_graph(monkeypatch):
    monkeypatch.setattr("src.api.main.graph_app", FakeGraph())

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_root_redirect():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307

def test_empty_message():
    response = client.post("/chat", json={"message": "", "session_id": "test"})
    assert response.status_code == 422

def test_missing_message():
    response = client.post("/chat", json={"session_id": "test"})
    assert response.status_code == 422

def test_empty_json():
    response = client.post("/chat", json={})
    assert response.status_code == 422

def test_non_json():
    response = client.post("/chat", content="not json")
    assert response.status_code == 422

def test_empty_session_id():
    response = client.post("/chat", json={"message": "hello", "session_id": ""})
    assert response.status_code == 422

def test_extra_field():
    response = client.post("/chat", json={"message": "hello", "extra_field": "bad"})
    assert response.status_code == 422

def test_valid_request(mock_graph):
    response = client.post("/chat", json={"message": "hello", "session_id": "test"})
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test"
    assert data["intent"] == "stock_research"
    assert data["topic"] == "Test"
    assert data["ticker"] == "TEST"
    assert data["divergence_verdict"] == "mixed"
    assert data["response"] == "Test response"
    assert data["chart_data"]["ticker"] == "TEST"

def test_chat_without_session_id_auto_generates_uuid(mock_graph):
    # When user sends message without providing session_id, backend dynamically creates a unique UUID
    response = client.post("/chat", json={"message": "hello without session"})
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert len(data["session_id"]) >= 16
    assert data["session_id"] != "demo"

def test_create_session_endpoint():
    response = client.post("/session/new")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert len(data["session_id"]) >= 16

def test_session_end():
    response = client.post("/session/end", json={"session_id": "test"})
    assert response.status_code == 200
    assert response.json() == {"session_id": "test", "removed_reports": 0}

def test_session_reports():
    response = client.get("/session/test/reports")
    assert response.status_code == 200
    assert response.json() == {"session_id": "test", "reports": []}
