"""
src/retriever.py
Hybrid retrieval: ChromaDB vector search + BM25, fused with RRF,
then re-ranked with FlashRank cross-encoder.
Single public interface: retrieve(query, top_k=3)
"""
import json
import pickle
import time
from pathlib import Path

import chromadb
from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer
from src.pseudonymise import Pseudonymiser

CHROMA_PATH  = "data/indices/chroma"
BM25_PATH    = "data/indices/bm25.pkl"
MODEL_NAME   = "BAAI/bge-base-en-v1.5"
COLLECTION   = "clinical_rag"
VECTOR_TOP_N = 20   # candidates from vector search
BM25_TOP_N   = 20   # candidates from BM25
RERANK_TOP_K = 3    # final chunks returned

# RRF constant — standard value
RRF_K = 60


def _rrf_score(rank: int) -> float:
    """Reciprocal Rank Fusion score for a given rank (1-indexed)."""
    return 1.0 / (RRF_K + rank)


class Retriever:
    """
    Hybrid retriever combining vector search + BM25 + cross-encoder re-ranking.
    Loads all indices once and keeps them in memory for fast repeated queries.
    """

    def __init__(self):
        print("[retriever] Loading indices...")
        t0 = time.time()

        # ChromaDB
        self._chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        self._collection = self._chroma.get_collection(COLLECTION)

        # Embedding model
        self._embedder = SentenceTransformer(MODEL_NAME)

        # BM25
        with open(BM25_PATH, "rb") as f:
            saved = pickle.load(f)
        self._bm25   = saved["bm25"]
        self._chunks = saved["chunks"]  # master list for BM25 lookup

        # FlashRank re-ranker (downloads small cross-encoder model)
        self._ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")

        # Pseudonymiser for query cleaning
        self._pseudonymiser = Pseudonymiser()

        print(f"[retriever] ✓ Ready in {time.time()-t0:.1f}s")

    def _embed_query(self, query: str) -> list[float]:
        return self._embedder.encode(
            [query], normalize_embeddings=True
        )[0].tolist()

    def _vector_search(self, query_embedding: list[float]) -> list[dict]:
        """Return top-N chunks from ChromaDB with their ranks."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=VECTOR_TOP_N,
            include=["documents", "metadatas", "distances"]
        )
        hits = []
        for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            hits.append({
                "chunk_id":    meta.get("doc_id", "") + f"_r{rank}",
                "text":        doc,
                "metadata":    meta,
                "vector_rank": rank + 1,
                "vector_score": 1 - dist,  # cosine similarity
            })
        return hits

    def _bm25_search(self, query: str) -> list[dict]:
        """Return top-N chunks from BM25 with their ranks."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:BM25_TOP_N]

        hits = []
        for rank, idx in enumerate(top_indices):
            c = self._chunks[idx]
            hits.append({
                "chunk_id":   c["chunk_id"],
                "text":       c["text"],
                "metadata": {
                    "doc_id":      c["doc_id"],
                    "specialty":   c["specialty"],
                    "sample_name": c["sample_name"],
                    "description": c["description"],
                    "chunk_index": c["chunk_index"],
                    "word_count":  c["word_count"],
                },
                "bm25_rank":  rank + 1,
                "bm25_score": float(scores[idx]),
            })
        return hits

    def _fuse_rrf(
        self,
        vector_hits: list[dict],
        bm25_hits: list[dict]
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion — merge vector and BM25 rankings.
        Chunks appearing in both lists get boosted scores.
        """
        scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for hit in vector_hits:
            cid = hit["chunk_id"]
            scores[cid]    = scores.get(cid, 0) + _rrf_score(hit["vector_rank"])
            chunk_map[cid] = hit

        for hit in bm25_hits:
            cid = hit["chunk_id"]
            scores[cid]    = scores.get(cid, 0) + _rrf_score(hit["bm25_rank"])
            if cid not in chunk_map:
                chunk_map[cid] = hit

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused = []
        for cid, rrf_score in ranked:
            entry = chunk_map[cid].copy()
            entry["rrf_score"] = rrf_score
            fused.append(entry)

        return fused

    def retrieve(
        self,
        query: str,
        top_k: int = RERANK_TOP_K,
        specialty_filter: str | None = None,
    ) -> list[dict]:
        """
        Main retrieval interface.

        Args:
            query:            Raw user query (PII will be stripped before search)
            top_k:            Number of final chunks to return (default 3)
            specialty_filter: Optionally restrict to a medical specialty

        Returns:
            List of top_k dicts with keys:
                text, metadata, rrf_score, rerank_score
        """
        t0 = time.time()

        # 1. Pseudonymise the query before any external call
        clean_query = self._pseudonymiser.pseudonymise_query(query)

        # 2. Embed the query
        q_embed = self._embed_query(clean_query)

        # 3. Vector search
        vector_hits = self._vector_search(q_embed)

        # 4. BM25 search
        bm25_hits = self._bm25_search(clean_query)

        # 5. RRF fusion
        fused = self._fuse_rrf(vector_hits, bm25_hits)

        # 6. Optional specialty filter
        if specialty_filter:
            fused = [
                h for h in fused
                if specialty_filter.lower() in h["metadata"].get("specialty", "").lower()
            ]

        # 7. Re-rank top-20 with FlashRank cross-encoder
        candidates = fused[:20]
        rerank_request = RerankRequest(
            query=clean_query,
            passages=[{"id": i, "text": h["text"]} for i, h in enumerate(candidates)]
        )
        reranked = self._ranker.rerank(rerank_request)

        # 8. Map rerank scores back and return top_k
        for item in reranked:
            candidates[item["id"]]["rerank_score"] = item["score"]

        final = sorted(
            candidates,
            key=lambda x: x.get("rerank_score", 0),
            reverse=True
        )[:top_k]

        latency = time.time() - t0

        # Attach pipeline trace for explainability
        for rank, chunk in enumerate(final):
            chunk["rank"]    = rank + 1
            chunk["latency"] = latency
            chunk["clean_query"] = clean_query

        return final


# ------------------------------------------------------------------ #
# Smoke test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    retriever = Retriever()

    test_queries = [
        "What are the symptoms of chest pain in cardiac patients?",
        "How is diabetes managed in elderly patients?",
        "What medications are used for depression and anxiety?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        results = retriever.retrieve(query, top_k=3)

        for chunk in results:
            print(f"\n  Rank {chunk['rank']}")
            print(f"  Specialty:     {chunk['metadata']['specialty']}")
            print(f"  RRF score:     {chunk.get('rrf_score', 0):.4f}")
            print(f"  Rerank score:  {chunk.get('rerank_score', 0):.4f}")
            print(f"  Latency:       {chunk['latency']:.2f}s")
            print(f"  Text preview:  {chunk['text'][:150]!r}")
