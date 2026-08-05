from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from src.graph.build_graph import build_graph
from src.storage.session_store import clear_session, get_chat_history, get_reports

logger = logging.getLogger("stigdiv")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s")


app = FastAPI(
    title="Signal Divergence Agent",
    description="Multi-agent backend for stock signal divergence research and market intelligence.",
    version="0.3.0",
)

# --- CORS -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request logging middleware ---------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s → %s (%.0fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# --- Global exception handler -----------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again."})


# --- Graph singleton --------------------------------------------------------
graph_app = build_graph()


import uuid


# --- Models -----------------------------------------------------------------
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, description="User query or instruction.")
    session_id: str | None = Field(
        default=None,
        min_length=1,
        description="Optional session identifier for conversational continuity. If omitted, a unique UUID is dynamically generated.",
    )
    use_llm: bool = Field(default=True, description="Whether to use configured LLM providers or rule-based offline engine.")


class ChartData(BaseModel):
    ticker: str | None = None
    period: str = "5d"
    interval: str = "1d"
    rows: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    topic: str | None = None
    ticker: str | None = None
    divergence_verdict: str | None = None
    response: str
    sources: dict[str, Any] = Field(default_factory=dict)
    chart_data: ChartData | None = None


class SessionEndRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1, description="Session ID to terminate.")


from pathlib import Path
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# --- Endpoints --------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/session/new")
async def create_session() -> dict:
    """Generate a new unique dynamic session ID."""
    return {"session_id": uuid.uuid4().hex}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # Dynamically generate unique session_id if not provided by client
    session_id = request.session_id.strip() if request.session_id else uuid.uuid4().hex

    # Run graph execution in thread pool to prevent blocking the async event loop for concurrent users
    result = await asyncio.to_thread(
        graph_app.invoke,
        {
            "user_query": request.message,
            "session_id": session_id,
            "use_llm": request.use_llm,
            "use_live_data": True,
        },
    )
    chart_raw = result.get("chart_data")
    chart = ChartData(**chart_raw) if isinstance(chart_raw, dict) else None
    return ChatResponse(
        session_id=session_id,
        intent=result.get("intent", "new_research"),
        topic=result.get("topic"),
        ticker=result.get("ticker"),
        divergence_verdict=result.get("divergence_verdict"),
        response=result.get("response") or result.get("final_report") or "No response generated.",
        sources=result.get("sources") or {},
        chart_data=chart,
    )


@app.get("/session/{session_id}/reports")
async def list_session_reports(session_id: str) -> dict:
    return {"session_id": session_id, "reports": get_reports(session_id)}


@app.get("/session/{session_id}/chat")
async def get_session_chat(session_id: str) -> dict:
    return {"session_id": session_id, "history": get_chat_history(session_id)}


@app.post("/session/end")
async def end_session(request: SessionEndRequest) -> dict:
    removed = clear_session(request.session_id)
    return {"session_id": request.session_id, "removed_reports": removed}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Explicit RESTful endpoint to delete and wipe a dynamic session completely."""
    removed = clear_session(session_id)
    return {"session_id": session_id, "removed_reports": removed, "status": "deleted"}
