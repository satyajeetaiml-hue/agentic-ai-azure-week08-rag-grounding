"""Smoke tests for Week 8 — Memory, State & Grounding (RAG)."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_endpoint_accepts_input():
    r = client.post("/api/v1/policy/ask", json={"question": "What is the post-operative antibiotic protocol for hip replacement?"})
    assert r.status_code == 200


def test_endpoint_rejects_empty():
    r = client.post("/api/v1/policy/ask", json={"question": ""})
    assert r.status_code == 422
