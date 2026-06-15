"""
app/main.py
-----------
FastAPI application.

Endpoints (existing preserved, /chat added):
  GET  /health    → readiness check (reflects FAISS index load status)
  GET  /          → API info
  GET  /recommend → existing semantic search endpoint (unchanged)
  POST /chat      → NEW conversational agent endpoint
"""
import asyncio
import logging

from fastapi import FastAPI, HTTPException, Query

import config
from app.models import RecommendationResponse
from app.retriever import retrieve
from app.chat_schema import ChatRequest, ChatResponse
from app.agent import run_agent

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SHL Assessment Recommendation API",
    version="2.0.0",
    description=(
        "Semantic search + conversational agent over the SHL product catalog. "
        "POST /chat implements the assignment's multi-turn agent. "
        "GET /recommend exposes the raw FAISS retrieval endpoint."
    ),
)

# ── Readiness flag ─────────────────────────────────────────────────────────────
# Flipped to True after the FAISS index is confirmed loaded.
# /health returns 503 while False so Render/Railway knows to keep waiting.
_ready: bool = False


@app.on_event("startup")
async def _startup() -> None:
    """
    Pre-warm the FAISS index on startup so the first /chat request is fast.
    Sets _ready=True once the index is confirmed in memory.
    """
    global _ready
    try:
        # retrieve() lazy-loads the index; calling it once here forces the load.
        _ = retrieve("warmup test", top_k=1)
        _ready = True
        logger.info("FAISS index pre-warmed successfully.")
    except Exception as exc:
        logger.error("Startup warmup failed (index may not exist yet): %s", exc)
        # Don't set _ready — health check will return 503 until fixed.


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", tags=["Infrastructure"])
def health() -> dict:
    """
    Readiness check.
    Returns 200 {"status": "ok"} when the FAISS index is loaded.
    Returns 503 {"status": "warming_up"} during startup.

    The assignment evaluator allows up to 2 minutes for this to become 200.
    For Docker deployments (baked index), this is instant.
    For cold-start services (Render/Railway), the index loads in ~5 seconds.
    """
    if not _ready:
        raise HTTPException(status_code=503, detail={"status": "warming_up"})
    return {"status": "ok"}


@app.get("/", tags=["Infrastructure"])
def home() -> dict:
    """API discovery endpoint."""
    return {
        "name":    "SHL Assessment Recommendation API",
        "version": "2.0.0",
        "docs":    "/docs",
        "endpoints": {
            "health":    "GET  /health",
            "recommend": "GET  /recommend?query=<text>&top_k=<1-10>",
            "chat":      "POST /chat",
        },
    }


@app.get("/recommend", response_model=RecommendationResponse, tags=["Retrieval"])
def recommend(
    query: str = Query(..., description="Free-text hiring query."),
    top_k: int = Query(10, ge=1, le=10, description="Max results (1-10)."),
) -> RecommendationResponse:
    """
    Semantic similarity search over the SHL catalog.
    Returns ranked assessments without conversation context.
    Unchanged from v1.
    """
    results = retrieve(query, top_k=top_k)
    return RecommendationResponse.from_retriever(query=query, raw=results)


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Conversational SHL Assessment Recommender.

    Accepts the full conversation history on every call (stateless).
    Returns a reply, an optional shortlist of recommendations, and an
    end_of_conversation flag.

    Behaviours:
      - Clarifies vague queries before recommending.
      - Recommends 1-10 catalog assessments when enough context exists.
      - Refines the shortlist when the user changes constraints.
      - Compares two named assessments on request.
      - Refuses off-topic questions and prompt injection attempts.
    """
    try:
        return await run_agent(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled error in /chat: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
