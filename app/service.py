"""Week 8 — Memory, State & Grounding (RAG): Clinical Policy Assistant.

Demonstrates retrieval-grounded answering with **permission-aware** knowledge:

* **Retrieval** — score documents against the question (mock hybrid scoring; Azure
  AI Search vector+hybrid+semantic in prod).
* **Security trimming** — drop documents the user's role/department may not see
  *before* they reach the model (a first-class requirement for PHI/PII).
* **Citations + grounding guardrail** — answers cite sources and say "I don't know"
  when nothing authorized is relevant.

``MockRagBackend`` runs offline/tested; ``SearchRagBackend`` uses Azure AI Search +
Foundry, lazy-imported, when search settings are present.
"""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── settings ────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    foundry_project_endpoint: str = ""
    foundry_model_name: str = "gpt-4o"
    foundry_embedding_deployment: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    azure_ai_search_endpoint: str = ""
    azure_ai_search_key: str = ""
    azure_ai_search_index: str = "clinical-policies"

    @property
    def use_search(self) -> bool:
        return bool(self.azure_ai_search_endpoint and self.azure_ai_search_key)

    @property
    def use_embeddings(self) -> bool:
        return bool(self.foundry_project_endpoint and self.foundry_embedding_deployment)


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ── schemas ─────────────────────────────────────────────────────────────
class PolicyQuestion(BaseModel):
    question: str = Field(..., min_length=1, description="Clinician's question about a care protocol.")
    user_role: str = Field(default="clinician", description="Caller role for security trimming.")
    department: str = Field(default="general", description="Caller department for security trimming.")


class Citation(BaseModel):
    title: str
    snippet: str
    score: float


class PolicyAnswer(BaseModel):
    answer: str
    grounded: bool
    citations: list[Citation]
    trimmed_count: int = Field(..., description="Docs excluded by security trimming (ACL).")
    mode: str


# ── knowledge base (with per-doc ACLs) — shared by the mock backend and the
#    setup_search.py ingestion script so both stay in sync. ────────────────
CLINICAL_DOCS = [
    {
        "id": "doc-1",
        "title": "Hip Replacement Post-Op Protocol",
        "content": "Post-operative antibiotic protocol for hip replacement: cefazolin 2g IV "
        "every 8 hours for 24 hours. Monitor for surgical-site infection.",
        "roles": ["clinician"],
        "departments": ["orthopedics", "general"],
    },
    {
        "id": "doc-2",
        "title": "Sepsis Care Pathway",
        "content": "Sepsis pathway: measure serum lactate, obtain blood cultures, and start "
        "broad-spectrum antibiotics within one hour of recognition.",
        "roles": ["clinician"],
        "departments": ["general", "icu"],
    },
    {
        "id": "doc-3",
        "title": "Controlled Substance Handling (Pharmacy)",
        "content": "Controlled substances require dual sign-off and reconciliation each shift.",
        "roles": ["pharmacist"],
        "departments": ["pharmacy"],
    },
]
_DOCS = CLINICAL_DOCS  # backwards-compatible alias used by the mock retriever
_WORD_RE = re.compile(r"[a-z]{3,}")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _visible(doc: dict, role: str, department: str) -> bool:
    """Security trimming: caller must satisfy the document's role + department ACL."""
    return role.lower() in doc["roles"] and department.lower() in doc["departments"]


def retrieve(question: str, role: str, department: str) -> tuple[list[Citation], int]:
    """Return (ranked citations, count_trimmed_by_acl)."""
    q = _tokens(question)
    trimmed = 0
    scored: list[Citation] = []
    for doc in _DOCS:
        if not _visible(doc, role, department):
            trimmed += 1
            continue
        overlap = q & _tokens(doc["title"] + " " + doc["content"])
        score = round(len(overlap) / (len(q) or 1), 3)
        if score > 0:
            scored.append(Citation(title=doc["title"], snippet=doc["content"][:160], score=score))
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored, trimmed


SYSTEM_INSTRUCTIONS = (
    "You are a clinical policy assistant. Answer ONLY from the provided context. "
    "Cite the document titles you used. If the context does not contain the answer, say you don't know."
)


# ── backends ────────────────────────────────────────────────────────────
class MockRagBackend:
    mode = "mock"

    def ask(self, q: PolicyQuestion) -> PolicyAnswer:
        citations, trimmed = retrieve(q.question, q.user_role, q.department)
        if not citations:
            return PolicyAnswer(
                answer="I don't know based on the available, authorized documents.",
                grounded=False, citations=[], trimmed_count=trimmed, mode=self.mode,
            )
        top = citations[0]
        answer = f"{top.snippet} (Source: {top.title})"
        return PolicyAnswer(
            answer=answer, grounded=True, citations=citations[:3], trimmed_count=trimmed, mode=self.mode,
        )


#: name of the vector field in the index (must match setup_search.py)
VECTOR_FIELD = "contentVector"


class SearchRagBackend:
    """Real Azure AI Search backend: hybrid (keyword + vector) retrieval with
    security-trimming pushed into the query, then grounded generation on Foundry.
    """

    mode = "search"

    def ask(self, q: PolicyQuestion) -> PolicyAnswer:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        s = get_settings()
        sc = SearchClient(
            endpoint=s.azure_ai_search_endpoint,
            index_name=s.azure_ai_search_index,
            credential=AzureKeyCredential(s.azure_ai_search_key),
        )
        # Security trimming: an OData filter on the ACL collection fields. Documents
        # the caller's role/department can't see never enter the result set.
        flt = (
            f"roles/any(r: r eq '{q.user_role}') and "
            f"departments/any(d: d eq '{q.department}')"
        )

        # When Foundry embeddings are configured we run a *hybrid* query (keyword +
        # vector, fused with RRF) and generate the answer in the same connection.
        if s.use_embeddings:
            return self._hybrid_ask(q, sc, flt, s)

        # Keyword-only fallback (still security-trimmed and grounded).
        results = list(sc.search(search_text=q.question, filter=flt, top=3))
        citations = _to_citations(results)
        if not citations:
            return _idk(self.mode)
        answer = f"{citations[0].snippet} (Source: {citations[0].title})"
        return PolicyAnswer(answer=answer, grounded=True, citations=citations, trimmed_count=0, mode=self.mode)

    def _hybrid_ask(self, q: PolicyQuestion, sc, flt: str, s) -> PolicyAnswer:
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential
        from azure.search.documents.models import VectorizedQuery

        with (
            DefaultAzureCredential() as cred,
            AIProjectClient(endpoint=s.foundry_project_endpoint, credential=cred) as proj,
        ):
            oai = proj.get_openai_client()
            query_vector = oai.embeddings.create(
                model=s.foundry_embedding_deployment, input=q.question
            ).data[0].embedding

            vector_query = VectorizedQuery(
                vector=query_vector, k_nearest_neighbors=5, fields=VECTOR_FIELD
            )
            results = list(
                sc.search(
                    search_text=q.question,          # keyword leg
                    vector_queries=[vector_query],   # vector leg (RRF-fused)
                    filter=flt,                      # security trimming
                    select=["id", "title", "content", "roles", "departments"],
                    top=3,
                )
            )
            citations = _to_citations(results)
            if not citations:
                return _idk(self.mode)

            context = "\n".join(f"[{c.title}] {c.snippet}" for c in citations)
            resp = oai.responses.create(
                model=s.foundry_model_name,
                input=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q.question}"},
                ],
            )
            answer = resp.output_text or f"{citations[0].snippet} (Source: {citations[0].title})"

        return PolicyAnswer(answer=answer, grounded=True, citations=citations, trimmed_count=0, mode=self.mode)


def _to_citations(results) -> list[Citation]:
    return [
        Citation(
            title=r.get("title", "doc"),
            snippet=(r.get("content", "") or "")[:160],
            score=float(r.get("@search.score", 0.0)),
        )
        for r in results
    ]


def _idk(mode: str) -> PolicyAnswer:
    return PolicyAnswer(
        answer="I don't know based on the available, authorized documents.",
        grounded=False, citations=[], trimmed_count=0, mode=mode,
    )


def get_backend():
    return SearchRagBackend() if get_settings().use_search else MockRagBackend()
