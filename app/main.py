"""Week 8 — Memory, State & Grounding (RAG).

Clinical Policy Assistant with permission-aware retrieval, citations, and an
"I don't know" guardrail. Run:  uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.service import PolicyAnswer, PolicyQuestion, get_backend, get_settings

settings = get_settings()
app = FastAPI(title="Week 8 — RAG (Clinical Policy Assistant)", version="0.2.0")


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok", "week": "8", "backend": "search" if settings.use_search else "mock"}


@app.get("/", tags=["root"])
def root() -> dict[str, str]:
    return {
        "service": "agentic-ai-azure-week08-rag-grounding",
        "endpoint": "/api/v1/policy/ask",
        "backend": "search" if settings.use_search else "mock",
        "docs": "/docs",
    }


@app.post("/api/v1/policy/ask", response_model=PolicyAnswer, tags=["week08"])
def ask(payload: PolicyQuestion) -> PolicyAnswer:
    return get_backend().ask(payload)
