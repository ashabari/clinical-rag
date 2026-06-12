"""
src/generator.py
Calls Claude Haiku with the built prompt and returns a grounded answer.
Full pipeline: query → pseudonymise → retrieve → prompt → generate
"""
import os
import time
from anthropic import Anthropic
from dotenv import load_dotenv

from src.retriever import Retriever
from src.prompt_builder import build_prompt, format_response
from src.pseudonymise import Pseudonymiser

load_dotenv()

MODEL    = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

# Initialise once — expensive to reload
_retriever      = None
_client         = None
_pseudonymiser  = None


def get_components():
    global _retriever, _client, _pseudonymiser
    if _retriever is None:
        _retriever     = Retriever()
        _client        = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        _pseudonymiser = Pseudonymiser()
    return _retriever, _client, _pseudonymiser


def answer(
    query: str,
    top_k: int = 3,
    specialty_filter: str | None = None,
) -> dict:
    """
    Full RAG pipeline for a single query.

    Args:
        query:            Raw user question
        top_k:            Number of chunks to retrieve (default 3)
        specialty_filter: Optionally restrict retrieval to a specialty

    Returns:
        Dict with keys: answer, sources, num_sources, latency_ms, clean_query
    """
    t0 = time.time()
    retriever, client, pseudonymiser = get_components()

    # Step 1: Pseudonymise the query (second PII pass — first was on chunks)
    clean_query = pseudonymiser.pseudonymise_query(query)

    # Step 2: Retrieve relevant chunks. already_clean=True because the
    # query was already pseudonymised in Step 1 -- avoids a redundant
    # NER pass over the same text inside retriever.retrieve().
    chunks = retriever.retrieve(
        clean_query,
        top_k=top_k,
        specialty_filter=specialty_filter,
        already_clean=True,
    )

    if not chunks:
        return {
            "answer":      "I could not find any relevant clinical notes for this question.",
            "sources":     [],
            "num_sources": 0,
            "latency_ms":  int((time.time() - t0) * 1000),
            "clean_query": clean_query,
        }

    # Step 3: Build prompt
    system_prompt, user_message = build_prompt(clean_query, chunks)

    # Step 4: Call Claude Haiku
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = response.content[0].text
    latency_ms  = int((time.time() - t0) * 1000)

    # Step 5: Package with full provenance
    result = format_response(answer_text, chunks)
    result["latency_ms"]  = latency_ms
    result["clean_query"] = clean_query
    result["model"]       = MODEL

    return result


def pretty_print(result: dict):
    """Print a result dict in a readable format."""
    print(f"\n{'='*65}")
    print(f"QUERY:   {result['clean_query']}")
    print(f"LATENCY: {result['latency_ms']}ms | MODEL: {result['model']}")
    print(f"{'='*65}")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSOURCES ({result['num_sources']}):")
    for s in result["sources"]:
        print(f"  [{s['rank']}] {s['specialty']} — rerank: {s['rerank_score']:.3f}")
        print(f"       {s['text_preview'][:120]!r}")


# ------------------------------------------------------------------ #
# Smoke test — end-to-end pipeline
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    test_queries = [
        "What are the common symptoms of chest pain in cardiac patients?",
        "How is Type 2 diabetes managed in overweight patients?",
        "What medications are typically prescribed for anxiety?",
        "What are the signs of a stroke?",              # Should have good recall
        "What is the best cryptocurrency to invest in?",# Should trigger refusal
    ]

    for query in test_queries:
        result = answer(query)
        pretty_print(result)
        print()
