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
    azure_ai_search_endpoint: str = ""
    azure_ai_search_key: str = ""
    azure_ai_search_index: str = "clinical-policies"

    @property
    def use_search(self) -> bool:
        return bool(self.azure_ai_search_endpoint and self.azure_ai_search_key)


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


# ── mock knowledge base (with per-doc ACLs) ─────────────────────────────
_DOCS = [
    {
        "title": "Hip Replacement Post-Op Protocol",
        "content": "Post-operative antibiotic protocol for hip replacement: cefazolin 2g IV "
        "every 8 hours for 24 hours. Monitor for surgical-site infection.",
        "roles": ["clinician"],
        "departments": ["orthopedics", "general"],
    },
    {
        "title": "Sepsis Care Pathway",
        "content": "Sepsis pathway: measure serum lactate, obtain blood cultures, and start "
        "broad-spectrum antibiotics within one hour of recognition.",
        "roles": ["clinician"],
        "departments": ["general", "icu"],
    },
    {
        "title": "Controlled Substance Handling (Pharmacy)",
        "content": "Controlled substances require dual sign-off and reconciliation each shift.",
        "roles": ["pharmacist"],
        "departments": ["pharmacy"],
    },
]
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


class SearchRagBackend:
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
        # Security trimming pushed into the query as an OData filter on ACL fields.
        flt = f"roles/any(r: r eq '{q.user_role}') and departments/any(d: d eq '{q.department}')"
        results = list(sc.search(search_text=q.question, filter=flt, top=3))
        citations = [
            Citation(title=r.get("title", "doc"), snippet=(r.get("content", "")[:160]),
                     score=float(r.get("@search.score", 0.0)))
            for r in results
        ]
        if not citations:
            return PolicyAnswer(
                answer="I don't know based on the available, authorized documents.",
                grounded=False, citations=[], trimmed_count=0, mode=self.mode,
            )
        answer = self._generate(q.question, citations) or f"{citations[0].snippet} (Source: {citations[0].title})"
        return PolicyAnswer(answer=answer, grounded=True, citations=citations, trimmed_count=0, mode=self.mode)

    def _generate(self, question: str, citations: list[Citation]) -> str | None:
        s = get_settings()
        if not s.foundry_project_endpoint:
            return None
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential

        context = "\n".join(f"[{c.title}] {c.snippet}" for c in citations)
        with (
            DefaultAzureCredential() as cred,
            AIProjectClient(endpoint=s.foundry_project_endpoint, credential=cred) as proj,
        ):
            client = proj.get_openai_client()
            resp = client.responses.create(
                model=s.foundry_model_name,
                input=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            )
            return resp.output_text


def get_backend():
    return SearchRagBackend() if get_settings().use_search else MockRagBackend()
