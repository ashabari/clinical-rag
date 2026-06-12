"""
src/prompt_builder.py
Builds the system prompt and user message for the clinical RAG pipeline.
Enforces minimum context, source labelling, and clinical guardrails.
"""

SYSTEM_PROMPT = """You are a clinical information assistant for Hilbi, a patient-centric digital healthcare platform.

Your role is to answer questions based ONLY on the clinical notes provided to you as context. You help patients and providers understand medical information from health records.

STRICT RULES:
1. Answer ONLY from the provided context. Never use outside knowledge.
2. If the context does not contain enough information to answer, say: "I cannot find sufficient information in the available clinical notes to answer this question."
3. Always cite your sources using [Source 1], [Source 2], etc.
4. Never provide diagnostic conclusions or treatment recommendations — describe only what is documented.
5. Never reveal or reconstruct any patient identifiers — all PII has been anonymised.
6. If asked something outside clinical scope (legal, financial, personal advice), politely decline.
7. Be concise, clear, and clinically accurate.

You are not a doctor. You are an information retrieval assistant. Always remind users to consult a qualified healthcare professional for medical decisions."""


def build_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Build the system and user messages for the LLM call.

    Args:
        query:  The (already pseudonymised) user query
        chunks: Top-k retrieved chunks from retriever.retrieve()

    Returns:
        Tuple of (system_prompt, user_message)
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source_header = (
            f"[Source {i}] "
            f"Specialty: {meta.get('specialty', 'Unknown')} | "
            f"Type: {meta.get('sample_name', 'Unknown')} | "
            f"Rerank score: {chunk.get('rerank_score', 0):.3f}"
        )
        context_blocks.append(f"{source_header}\n{chunk['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    user_message = f"""CLINICAL CONTEXT (retrieved from de-identified patient records):

{context_str}

---

QUESTION: {query}

Answer based solely on the clinical context above. Cite sources using [Source 1], [Source 2], etc."""

    return SYSTEM_PROMPT, user_message


def format_response(answer: str, chunks: list[dict]) -> dict:
    """
    Package the final response with full provenance for explainability.
    This is what gets returned to the user / eval system.

    Note: includes BOTH the full chunk text (for eval/judging, where the
    judge needs to see exactly what the generator saw) and a short
    text_preview (for human-readable display).
    """
    return {
        "answer": answer,
        "sources": [
            {
                "rank":          i + 1,
                "specialty":     c["metadata"].get("specialty", ""),
                "sample_name":   c["metadata"].get("sample_name", ""),
                "description":   c["metadata"].get("description", ""),
                "rerank_score":  round(float(c.get("rerank_score", 0)), 4),
                "rrf_score":     round(float(c.get("rrf_score", 0)), 4),
                "text":          c["text"],          # full chunk — used for eval grounding
                "text_preview":  c["text"][:200],    # short — used for display only
            }
            for i, c in enumerate(chunks)
        ],
        "num_sources": len(chunks),
    }
