"""
eval/run_ablation.py
Compares retrieval quality across three configurations:
  1. vector-only (top-3 by embedding similarity)
  2. hybrid (RRF fusion of vector + BM25, no rerank)
  3. hybrid + rerank (full pipeline)
"""
import json
from pathlib import Path

import pandas as pd

from src.retriever import Retriever

EVAL_SET_PATH = "eval/eval_set.json"
OUTPUT_CSV = "outputs/ablation_results.csv"


def run_ablation():
    retriever = Retriever()

    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)

    rows = []

    for item in eval_set:
        if not item.get("expected_specialty"):
            continue

        query = item["question"]
        expected = item["expected_specialty"]

        clean_query = retriever._pseudonymiser.pseudonymise_query(query)
        q_embed = retriever._embed_query(clean_query)

        vector_hits = retriever._vector_search(q_embed)
        vector_top3 = vector_hits[:3]
        vector_specialties = set(h["metadata"]["specialty"] for h in vector_top3)
        vector_match = expected in vector_specialties

        bm25_hits = retriever._bm25_search(clean_query)
        fused = retriever._fuse_rrf(vector_hits, bm25_hits)
        hybrid_top3 = fused[:3]
        hybrid_specialties = set(h["metadata"]["specialty"] for h in hybrid_top3)
        hybrid_match = expected in hybrid_specialties

        final = retriever.retrieve(query, top_k=3)
        final_specialties = set(h["metadata"]["specialty"] for h in final)
        final_match = expected in final_specialties

        rows.append({
            "id": item["id"],
            "question": query[:50],
            "expected_specialty": expected,
            "vector_only_match": vector_match,
            "hybrid_rrf_match": hybrid_match,
            "hybrid_rerank_match": final_match,
        })

        print("  " + item["id"] + ": vector=" + str(vector_match) + " hybrid=" + str(hybrid_match) + " hybrid+rerank=" + str(final_match))

    df = pd.DataFrame(rows)
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 50)
    print("ABLATION SUMMARY (retrieval match rate)")
    print("=" * 50)
    print("Vector-only:       " + str(round(df["vector_only_match"].mean() * 100, 1)) + "%")
    print("Hybrid (RRF):      " + str(round(df["hybrid_rrf_match"].mean() * 100, 1)) + "%")
    print("Hybrid + rerank:   " + str(round(df["hybrid_rerank_match"].mean() * 100, 1)) + "%")
    print("\nSaved to " + OUTPUT_CSV)


if __name__ == "__main__":
    run_ablation()
