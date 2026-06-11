# Week 8 — Memory, State & Grounding (RAG)

[![CI](https://github.com/satyajeetaiml-hue/agentic-ai-azure-week08-rag-grounding/actions/workflows/ci.yml/badge.svg)](https://github.com/satyajeetaiml-hue/agentic-ai-azure-week08-rag-grounding/actions/workflows/ci.yml)

> **Standalone lab** from the *Agentic AI on Azure — Enterprise Master Class* (12 weeks).
> Each lab is an independent, runnable FastAPI starter. Part of the
> [course series](https://github.com/satyajeetaiml-hue?tab=repositories&q=agentic-ai-azure).

---

## 🎯 Learning goal
Build retrieval-grounded agents with permission-aware knowledge over enterprise data.

## 🏢 Enterprise use case — "Clinical Policy Assistant" (Healthcare)
Clinicians query care protocols. The agent retrieves from a curated knowledge base in Azure AI Search, trims results by the user's role/department (security trimming), cites sources, and never answers beyond grounded content.

---

## 🧪 What you'll build (lab)
1. Ingest documents → chunk → embed → index in **Azure AI Search** (vector + hybrid + semantic ranker).
2. Implement **permission/ACL trimming** using Entra group claims.
3. Build a RAG endpoint in FastAPI that returns answers with citation payloads.
4. Add an "I don't know" guardrail for ungrounded questions.

> This starter ships with a **runnable mock** of the endpoint so you can run and test
> immediately, then progressively replace the mock with the real Azure implementation.

## 🏗️ Architect's lens
- Retrieval quality: chunking strategy, hybrid search, reranking, recency.
- Security trimming as a first-class requirement (PHI/PII — never leak across boundaries).
- Grounding vs. hallucination guardrails; explicit "I don't know" behavior.

## 🧰 Tech stack
Azure AI Search (vector/hybrid/semantic), Azure OpenAI embeddings, Azure Blob Storage, Cosmos DB, Entra ID groups for ACL, FastAPI.

---

## 🚀 Quick start

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) copy the env template — runs in MOCK mode without it
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux

# 4. Run the API
uvicorn app.main:app --reload
```

Open the interactive docs at **http://127.0.0.1:8000/docs**.

### Try the endpoint
```bash
curl -X POST http://127.0.0.1:8000/api/v1/policy/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the post-operative antibiotic protocol for hip replacement?"}'
```

### Run the tests
```bash
pytest -q
```

### Run with Docker
```bash
docker build -t agentic-ai-azure-week08-rag-grounding .
docker run -p 8000:8000 agentic-ai-azure-week08-rag-grounding
```

---

## 📁 Project structure
```
agentic-ai-azure-week08-rag-grounding/
├── app/
│   ├── __init__.py
│   └── main.py          # FastAPI app + the /api/v1/policy/ask endpoint
├── tests/
│   └── test_smoke.py
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

---

## 🗺️ Where this fits
This repo covers **Week 8 — Memory, State & Grounding (RAG)**. The full 12-week path and reference architecture
live in the master-class companion repo:
**[azure-agentic-ai-masterclass](https://github.com/satyajeetaiml-hue/azure-agentic-ai-masterclass)**.

## 📄 License
MIT — see [`LICENSE`](LICENSE).
