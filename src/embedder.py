"""
src/embedder.py
Embeds all chunks using a local sentence-transformers model.
Stores in ChromaDB. Builds BM25 index in parallel.
No API key needed — runs fully locally.
"""
import json
import pickle
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

CHUNKS_PATH  = "data/chunks/chunks.jsonl"
CHROMA_PATH  = "data/indices/chroma"
BM25_PATH    = "data/indices/bm25.pkl"
BATCH_SIZE   = 64
MODEL_NAME   = "BAAI/bge-base-en-v1.5"  # strong, fast, free, clinical-friendly
COLLECTION   = "clinical_rag"


def load_chunks(path: str = CHUNKS_PATH) -> list[dict]:
    chunks = []
    with open(path) as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"[embedder] Loaded {len(chunks):,} chunks")
    return chunks


def build_chroma_index(chunks: list[dict]) -> chromadb.Collection:
    print(f"[embedder] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION)
        print(f"[embedder] Cleared existing collection")
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    texts     = [c["text"]     for c in chunks]
    ids       = [c["chunk_id"] for c in chunks]
    metadatas = [{
        "doc_id":      c["doc_id"],
        "specialty":   c["specialty"],
        "sample_name": c["sample_name"],
        "description": c["description"],
        "chunk_index": c["chunk_index"],
        "word_count":  c["word_count"],
    } for c in chunks]

    print(f"[embedder] Embedding {len(texts):,} chunks locally (no API needed)...")
    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_embeddings.extend(embeddings.tolist())

    print(f"[embedder] Storing in ChromaDB...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Storing"):
        collection.add(
            ids=ids[i:i + BATCH_SIZE],
            embeddings=all_embeddings[i:i + BATCH_SIZE],
            documents=texts[i:i + BATCH_SIZE],
            metadatas=metadatas[i:i + BATCH_SIZE],
        )

    print(f"[embedder] ✓ ChromaDB index built — {collection.count():,} vectors")
    return collection


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    print(f"[embedder] Building BM25 index...")
    tokenised = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenised)

    Path(BM25_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"[embedder] ✓ BM25 index saved to {BM25_PATH}")
    return bm25


def get_query_embedding(query: str) -> list[float]:
    """Embed a single query — used by retriever.py"""
    model = SentenceTransformer(MODEL_NAME)
    return model.encode([query], normalize_embeddings=True)[0].tolist()


if __name__ == "__main__":
    chunks     = load_chunks()
    collection = build_chroma_index(chunks)
    bm25       = build_bm25_index(chunks)

    # Smoke test
    print("\n--- Smoke test: 'chest pain shortness of breath' ---")
    model   = SentenceTransformer(MODEL_NAME)
    q_embed = model.encode(
        ["chest pain shortness of breath"],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(query_embeddings=[q_embed], n_results=3)
    print("\nTop 3 vector results:")
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        print(f"  [{i+1}] {meta['specialty']} — {doc[:120]!r}")

    with open(BM25_PATH, "rb") as f:
        saved = pickle.load(f)
    scores = saved["bm25"].get_scores("chest pain shortness of breath".split())
    top3   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    print("\nTop 3 BM25 results:")
    for idx in top3:
        c = saved["chunks"][idx]
        print(f"  [{idx}] {c['specialty']} — {c['text'][:120]!r}")
