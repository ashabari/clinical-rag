"""
src/chunker.py
Sentence-aware sliding window chunker with pseudonymisation built in.
Outputs chunks.jsonl to data/chunks/
"""
import json
import re
from pathlib import Path
from tqdm import tqdm
from src.pseudonymise import Pseudonymiser
from src.loader import load_mtsamples

CHUNK_SIZE    = 400   # target tokens (approx — we use words as proxy)
OVERLAP       = 60    # words of overlap between chunks
WORDS_PER_TOK = 0.75  # conservative word→token ratio


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, preserving clinical note structure."""
    # Split on period/exclamation/question followed by whitespace + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Also split on clinical section headers (ALL CAPS lines)
    result = []
    for sent in sentences:
        parts = re.split(r'\n(?=[A-Z\s]{4,}:)', sent)
        result.extend(p.strip() for p in parts if p.strip())
    return result


def chunk_document(doc: dict, pseudonymiser: Pseudonymiser) -> list[dict]:
    """
    Chunk a single document into overlapping windows of ~400 tokens.
    Each chunk is pseudonymised before being returned.
    """
    text = doc["text"]
    sentences = split_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_words = []
    current_sents = []
    chunk_idx = 0

    for sent in sentences:
        words = sent.split()
        # If adding this sentence would exceed chunk size, flush current chunk
        if current_words and len(current_words) + len(words) > CHUNK_SIZE:
            chunk_text = " ".join(current_words)
            clean_text = pseudonymiser.pseudonymise(chunk_text)

            chunks.append({
                "chunk_id":   f"{doc['doc_id']}_chunk_{chunk_idx:03d}",
                "doc_id":     doc["doc_id"],
                "text":       clean_text,
                "specialty":  doc["specialty"],
                "sample_name": doc["sample_name"],
                "description": doc["description"],
                "keywords":   doc["keywords"],
                "source_file": doc["source_file"],
                "chunk_index": chunk_idx,
                "word_count":  len(current_words),
            })
            chunk_idx += 1

            # Keep overlap — retain last N words for context continuity
            overlap_words = current_words[-OVERLAP:] if len(current_words) > OVERLAP else current_words
            current_words = overlap_words + words
        else:
            current_words.extend(words)
        current_sents.append(sent)

    # Flush the final chunk
    if current_words:
        chunk_text = " ".join(current_words)
        clean_text = pseudonymiser.pseudonymise(chunk_text)
        chunks.append({
            "chunk_id":    f"{doc['doc_id']}_chunk_{chunk_idx:03d}",
            "doc_id":      doc["doc_id"],
            "text":        clean_text,
            "specialty":   doc["specialty"],
            "sample_name": doc["sample_name"],
            "description": doc["description"],
            "keywords":    doc["keywords"],
            "source_file": doc["source_file"],
            "chunk_index": chunk_idx,
            "word_count":  len(current_words),
        })

    return chunks


def chunk_all_documents(
    csv_path: str = "data/raw/mtsamples.csv",
    output_path: str = "data/chunks/chunks.jsonl",
    max_docs: int | None = None,
) -> list[dict]:
    """Load, chunk, pseudonymise, and save all documents."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    docs = load_mtsamples(csv_path, max_docs=max_docs)
    pseudonymiser = Pseudonymiser()

    all_chunks = []
    print(f"\n[chunker] Processing {len(docs)} documents...")

    for doc in tqdm(docs, desc="Chunking"):
        chunks = chunk_document(doc, pseudonymiser)
        all_chunks.extend(chunks)

    # Write JSONL
    with open(output_path, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    avg_words = sum(c["word_count"] for c in all_chunks) / len(all_chunks)
    print(f"\n[chunker] ✓ {len(all_chunks):,} chunks from {len(docs):,} documents")
    print(f"[chunker] ✓ Avg chunk size: {avg_words:.0f} words (~{avg_words/WORDS_PER_TOK:.0f} tokens)")
    print(f"[chunker] ✓ Saved to: {output_path}")
    print(f"[chunker] ✓ PII entities redacted: {sum(pseudonymiser._counters.values())}")

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all_documents()

    print("\n--- Sample chunk ---")
    c = chunks[0]
    print(f"  chunk_id:  {c['chunk_id']}")
    print(f"  specialty: {c['specialty']}")
    print(f"  words:     {c['word_count']}")
    print(f"  text:      {c['text'][:300]!r}")
