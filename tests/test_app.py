"""Hermetic tests for the Week 8 RAG assistant (mock backend)."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_mock():
    assert client.get("/health").json()["backend"] == "mock"


def test_grounded_answer_with_citation():
    r = client.post(
        "/api/v1/policy/ask",
        json={
            "question": "What is the post-operative antibiotic protocol for hip replacement?",
            "user_role": "clinician",
            "department": "orthopedics",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert any("Hip Replacement" in c["title"] for c in body["citations"])


def test_security_trimming_blocks_unauthorized_role():
    # A pharmacist in pharmacy cannot see the clinician orthopedics doc.
    r = client.post(
        "/api/v1/policy/ask",
        json={
            "question": "antibiotic protocol for hip replacement",
            "user_role": "pharmacist",
            "department": "pharmacy",
        },
    )
    body = r.json()
    assert body["grounded"] is False
    assert body["trimmed_count"] >= 1


def test_ungrounded_question_says_i_dont_know():
    r = client.post(
        "/api/v1/policy/ask",
        json={"question": "What is the cafeteria wifi password?", "user_role": "clinician"},
    )
    body = r.json()
    assert body["grounded"] is False
    assert "don't know" in body["answer"].lower()


def test_validation_rejects_empty():
    assert client.post("/api/v1/policy/ask", json={"question": ""}).status_code == 422
