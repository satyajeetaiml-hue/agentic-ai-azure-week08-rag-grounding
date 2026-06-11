"""One-time setup: create the Azure AI Search index and ingest the clinical docs.

Creates a vector-enabled index ``clinical-policies`` with:
  * keyword fields (title, content),
  * **ACL collection fields** (roles, departments) used for security trimming, and
  * a vector field (contentVector) backed by an HNSW profile.

Then it embeds each document's content with an Azure OpenAI embedding deployment
(via the Foundry project's OpenAI client) and uploads the documents.

Prerequisites (see README "Wire the real Azure backend"):
  * `az login`
  * env: AZURE_AI_SEARCH_ENDPOINT, AZURE_AI_SEARCH_KEY, AZURE_AI_SEARCH_INDEX,
          FOUNDRY_PROJECT_ENDPOINT, FOUNDRY_EMBEDDING_DEPLOYMENT, EMBEDDING_DIMENSIONS

Run:  python scripts/setup_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/setup_search.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.service import CLINICAL_DOCS, VECTOR_FIELD, get_settings  # noqa: E402


def _index_client():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient

    s = get_settings()
    return SearchIndexClient(s.azure_ai_search_endpoint, AzureKeyCredential(s.azure_ai_search_key))


def create_index() -> None:
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    s = get_settings()
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="roles",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
        ),
        SearchField(
            name="departments",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
        ),
        SearchField(
            name=VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=s.embedding_dimensions,
            vector_search_profile_name="clinical-vector-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="clinical-hnsw")],
        profiles=[
            VectorSearchProfile(
                name="clinical-vector-profile",
                algorithm_configuration_name="clinical-hnsw",
            )
        ],
    )
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="clinical-semantic",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                ),
            )
        ]
    )
    index = SearchIndex(
        name=s.azure_ai_search_index,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )
    result = _index_client().create_or_update_index(index)
    print(f"Created/updated index '{result.name}'.")


def _embed_all(texts: list[str]) -> list[list[float]]:
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    s = get_settings()
    with (
        DefaultAzureCredential() as cred,
        AIProjectClient(endpoint=s.foundry_project_endpoint, credential=cred) as proj,
    ):
        oai = proj.get_openai_client()
        resp = oai.embeddings.create(model=s.foundry_embedding_deployment, input=texts)
        return [d.embedding for d in resp.data]


def upload_documents() -> None:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    s = get_settings()
    vectors = _embed_all([d["content"] for d in CLINICAL_DOCS])
    docs = [{**d, VECTOR_FIELD: vec} for d, vec in zip(CLINICAL_DOCS, vectors, strict=True)]

    sc = SearchClient(s.azure_ai_search_endpoint, s.azure_ai_search_index, AzureKeyCredential(s.azure_ai_search_key))
    result = sc.upload_documents(documents=docs)
    print(f"Uploaded {len(result)} documents to '{s.azure_ai_search_index}'.")


def main() -> None:
    s = get_settings()
    if not s.use_search:
        sys.exit("Set AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_KEY first.")
    if not s.use_embeddings:
        sys.exit("Set FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_EMBEDDING_DEPLOYMENT first (and `az login`).")
    create_index()
    upload_documents()
    print("Done. Try: POST /api/v1/policy/ask")


if __name__ == "__main__":
    main()
