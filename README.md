# Week 8 — Memory, State & Grounding (RAG)

[![CI](https://github.com/satyajeetaiml-hue/agentic-ai-azure-week08-rag-grounding/actions/workflows/ci.yml/badge.svg)](https://github.com/satyajeetaiml-hue/agentic-ai-azure-week08-rag-grounding/actions/workflows/ci.yml)

> ▶️ **Run in VS Code — no Azure needed.** `pip install -r requirements.txt`, then `uvicorn app.main:app --reload` and open http://127.0.0.1:8000/docs. Runs in **mock mode** by default — no `az login`, keys, or `.env` required. Wiring real Azure (below) is optional.

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

## ☁️ Wire the real Azure backend (Azure AI Search)

This lab ships a **real, runnable** Azure AI Search backend plus a one-time setup script that creates the
index and ingests the sample clinical docs with embeddings.

### 1. Provision (Azure CLI)
```bash
az login
RG=rg-agentic-rag
az group create -n $RG -l eastus

# Azure AI Search service
az search service create -g $RG -n my-clinical-search --sku basic
az search admin-key show -g $RG --service-name my-clinical-search   # copy the primary key
```
You also need a **Microsoft Foundry** project with two deployments: a chat model (e.g. `gpt-4o`) and an
**embedding** model (e.g. `text-embedding-3-small`, 1536 dims). Grant your identity the **Azure AI User**
role on the Foundry project (`az login` provides the credential).

### 2. Configure `.env`
```bash
cp .env.example .env
```
```
AZURE_AI_SEARCH_ENDPOINT=https://my-clinical-search.search.windows.net
AZURE_AI_SEARCH_KEY=<primary-admin-key>
AZURE_AI_SEARCH_INDEX=clinical-policies
FOUNDRY_PROJECT_ENDPOINT=https://<account>.services.ai.azure.com/api/projects/<project>
FOUNDRY_MODEL_NAME=gpt-4o
FOUNDRY_EMBEDDING_DEPLOYMENT=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

### 3. Create the index + ingest docs (one time)
```bash
python scripts/setup_search.py
```
This creates a vector-enabled index with **ACL collection fields** (`roles`, `departments`), an **HNSW**
vector profile, and a **semantic** config, then embeds and uploads the docs.

### 4. Run — now backed by Azure AI Search
```bash
uvicorn app.main:app --reload   # GET /health -> "backend": "search"
```
The query path (`app/service.py` → `SearchRagBackend`) embeds the question, runs a **hybrid** query
(keyword + vector, RRF-fused), pushes **security trimming** into an OData `roles/any(...) and
departments/any(...)` filter, and grounds the answer on the retrieved citations via Foundry. If no
embedding deployment is set, it degrades to keyword-only search (still trimmed and grounded).

> Without `FOUNDRY_EMBEDDING_DEPLOYMENT` you get keyword-only search; with it you get true hybrid.

## 🏗️ Architect's lens
- Retrieval quality: chunking, hybrid search, reranking, recency.
- **Security trimming as a first-class requirement** (PHI/PII never crosses boundaries).
- Grounding vs. hallucination guardrails; explicit "I don't know".

## 🧰 Tech stack
Azure AI Search (vector/hybrid/semantic), Azure OpenAI embeddings, Blob Storage, Entra ID groups for ACL,
FastAPI, azure-ai-projects v2.

## 📁 Structure
```
app/service.py          # settings, schemas, mock + real (hybrid) Search backends
app/main.py             # POST /api/v1/policy/ask
scripts/setup_search.py # one-time: create index + ingest docs with embeddings
tests/test_app.py
```

## 🗺️ Series
Prev: [Weeks 6-7](https://github.com/satyajeetaiml-hue/agentic-ai-azure-week06-07-multi-agent) ·
Next: [Week 9 — Hosting & Scale](https://github.com/satyajeetaiml-hue/agentic-ai-azure-week09-hosting-scale) ·
[All labs](https://github.com/satyajeetaiml-hue?tab=repositories&q=agentic-ai-azure)

## 📄 License
MIT — see [`LICENSE`](LICENSE).

## 📊 Teaching slides

Download the **7-slide deck** for classroom use: [`agentic-ai-azure-week08-rag-grounding.pptx`](slides/agentic-ai-azure-week08-rag-grounding.pptx)

Prefer PDF? Download the **handout (slides + speaker notes)**: [`agentic-ai-azure-week08-rag-grounding-handout.pdf`](slides/agentic-ai-azure-week08-rag-grounding-handout.pdf)

> Slides: Title · Learning goal · Enterprise use case · Architecture/flow · Key concepts · Run it · Architect's takeaways.

