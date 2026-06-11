"""Week 8 — Memory, State & Grounding (RAG) — starter FastAPI service.

Use case: Clinical Policy Assistant (Healthcare).
See README.md for the full lab brief. Run:  uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Week 8 — Memory, State & Grounding (RAG)", version="0.1.0")


class LabRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The clinician's question about a care protocol.")


@app.get("/health")
def health():
    return {"status": "ok", "week": "8", "use_case": "Clinical Policy Assistant"}


@app.get("/")
def root():
    return {
        "service": "agentic-ai-azure-week08-rag-grounding",
        "week": "8",
        "endpoint": "/api/v1/policy/ask",
        "docs": "/docs",
    }


@app.post("/api/v1/policy/ask")
def handler(payload: LabRequest):
    """Mock handler for the Clinical Policy Assistant.

    TODO (lab): replace this stub with the real implementation described in
    README.md (the Azure services for this week are listed in the Tech Stack).
    """
    return {
        "week": "8",
        "use_case": "Clinical Policy Assistant",
        "received": payload.question,
        "status": "accepted",
        "note": "Mock response — implement the real agent per README.md.",
    }
