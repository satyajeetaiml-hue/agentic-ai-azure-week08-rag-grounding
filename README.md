# Week 8 — Memory, State & Grounding (RAG)

[![CI](https://github.com/satyajeetaiml-hue/agentic-ai-azure-week08-rag-grounding/actions/workflows/ci.yml/badge.svg)](https://github.com/satyajeetaiml-hue/agentic-ai-azure-week08-rag-grounding/actions/workflows/ci.yml)

> **Standalone lab** from the *Agentic AI on Azure — Enterprise Master Class*.
> Course hub: [azure-agentic-ai-masterclass](https://github.com/satyajeetaiml-hue/azure-agentic-ai-masterclass).

---

## 🎯 Learning goal
Build a retrieval-grounded agent with **permission-aware** knowledge over enterprise data.

## 🏢 Enterprise use case — "Clinical Policy Assistant" (Healthcare)
Clinicians query care protocols. The agent retrieves from a knowledge base, **trims results by the user's
role/department** (security trimming), cites sources, and never answers beyond grounded content.

## ✅ What this repo implements
- **Retrieval** with relevance scoring (mock hybrid; **Azure AI Search** vector+hybrid+semantic in prod).
- **Security trimming** — each document carries a role/department ACL; unauthorized docs are dropped
  *before* reaching the model (the mock counts them as `trimmed_count`; the search backend pushes the ACL
  into an **OData filter**).
- **Citations** on every grounded answer + an **"I don't know"** guardrail for ungrounded questions.
- **Mock backend** (offline, tested) and **Search backend** (Azure AI Search + Foundry, lazy-imported).

## 🚀 Quick start
```bash
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```
```bash
# Authorized + grounded
curl -X POST http://127.0.0.1:8000/api/v1/policy/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "post-operative antibiotic protocol for hip replacement", "user_role": "clinician", "department": "orthopedics"}'

# Same question as pharmacist/pharmacy -> trimmed -> "I don't know"
```
Run tests: `pytest -q`

## ☁️ Search backend
Set `AZURE_AI_SEARCH_ENDPOINT` + `AZURE_AI_SEARCH_KEY` (and optionally `FOUNDRY_PROJECT_ENDPOINT` to
generate the answer). Your index needs `title`, `content`, and collection fields `roles`/`departments`
for ACL filtering. `GET /health` reports `"backend": "search"`.

## 🏗️ Architect's lens
- Retrieval quality: chunking, hybrid search, reranking, recency.
- **Security trimming as a first-class requirement** (PHI/PII never crosses boundaries).
- Grounding vs. hallucination guardrails; explicit "I don't know".

## 🧰 Tech stack
Azure AI Search (vector/hybrid/semantic), Azure OpenAI embeddings, Blob Storage, Entra ID groups for ACL,
FastAPI, azure-ai-projects v2.

## 📁 Structure
```
app/service.py   # settings, schemas, retrieval + security trimming, backends
app/main.py      # POST /api/v1/policy/ask
tests/test_app.py
```

## 🗺️ Series
Prev: [Weeks 6-7](https://github.com/satyajeetaiml-hue/agentic-ai-azure-week06-07-multi-agent) ·
Next: [Week 9 — Hosting & Scale](https://github.com/satyajeetaiml-hue/agentic-ai-azure-week09-hosting-scale) ·
[All labs](https://github.com/satyajeetaiml-hue?tab=repositories&q=agentic-ai-azure)

## 📄 License
MIT — see [`LICENSE`](LICENSE).
