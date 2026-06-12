"""
eval/run_retrieval_eval.py

Computes Recall@K (K=1,3,5,10), Hit@K, and MRR for the 8 questions in
eval_set.json that have ground_truth_chunk_ids, against the full
retriever.retrieve(query, top_k=10) pipeline (pseudonymise -> embed ->
vector + BM25 -> RRF fuse -> cross-encoder rerank).

- Recall@K = fraction of this question's ground-truth chunk_ids found
  within the top-K retrieved chunk_ids. Handles multi-chunk ground
  truth (q03/04/07/08) by treating it as "fraction of the relevant
  set found", the standard definition.
- Hit@K = 1 if AT LEAST ONE ground-truth chunk_id is in the top-K,
  else 0. More forgiving for multi-chunk ground truth.
- MRR = 1 / rank of the first ground-truth chunk_id found in the
  top-10, or 0 if none of them appear at all.

Run: python -m eval.run_retrieval_eval
"""
import json
import pandas as pd
from src.retriever import Retriever

EVAL_SET_PATH = "eval/eval_set.json"
OUTPUT_CSV = "outputs/retrieval_eval_results.csv"
K_VALUES = [1, 3, 5, 10]
TOP_K = 10


def run():
    retriever = Retriever()

    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)

    items = [it for it in eval_set if "ground_truth_chunk_ids" in it]
    print(f"[retrieval-eval] {len(items)} questions with ground truth\n")

    rows = []
    for item in items:
        results = retriever.retrieve(item["question"], top_k=TOP_K)
        retrieved_ids = [r["chunk_id"] for r in results]
        gt_ids = set(item["ground_truth_chunk_ids"])

        row = {
            "id": item["id"],
            "category": item["category"],
            "n_gt": len(gt_ids),
            "retrieved_top10": retrieved_ids,
        }

        for k in K_VALUES:
            top_k_ids = set(retrieved_ids[:k])
            overlap = top_k_ids & gt_ids
            row[f"recall@{k}"] = len(overlap) / len(gt_ids)
            row[f"hit@{k}"] = 1.0 if overlap else 0.0

        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in gt_ids:
                rr = 1.0 / rank
                break
        row["mrr"] = rr

        rows.append(row)
        print(f"  {item['id']}: recall@3={row['recall@3']:.2f}  "
              f"hit@3={row['hit@3']:.0f}  mrr={row['mrr']:.2f}")
        print(f"        gt={sorted(gt_ids)}")
        print(f"        top3={retrieved_ids[:3]}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 50)
    print("RETRIEVAL EVAL SUMMARY (mean across 8 questions)")
    print("=" * 50)
    for k in K_VALUES:
        print(f"Recall@{k}: {df[f'recall@{k}'].mean():.3f}   "
              f"Hit@{k}: {df[f'hit@{k}'].mean():.3f}")
    print(f"MRR:      {df['mrr'].mean():.3f}")

    print("\nBy category:")
    print(df.groupby("category")[["recall@3", "hit@3", "mrr"]].mean().round(3))

    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
