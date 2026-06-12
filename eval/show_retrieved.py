"""
eval/show_retrieved.py
For the 4 single-ground-truth questions, prints the sample_name +
specialty + first 200 chars of each of the actual top-3 retrieved
chunks, so we can tell whether a "miss" is a different-but-valid
chunk or a genuinely off-topic result.
"""
import json
from src.retriever import Retriever

EVAL_SET_PATH = "eval/eval_set.json"
TARGET_IDS = {"q01", "q02", "q05", "q13"}

chunks_by_id = {}
with open("data/chunks/chunks.jsonl") as f:
    for line in f:
        c = json.loads(line)
        chunks_by_id[c["chunk_id"]] = c

retriever = Retriever()

with open(EVAL_SET_PATH) as f:
    eval_set = json.load(f)

for item in eval_set:
    if item["id"] not in TARGET_IDS:
        continue
    print("=" * 70)
    print(f"{item['id']}: {item['question']}")
    print(f"  ground truth: {item['ground_truth_chunk_ids']}")
    print("=" * 70)

    results = retriever.retrieve(item["question"], top_k=3)
    for rank, r in enumerate(results, start=1):
        cid = r["chunk_id"]
        c = chunks_by_id.get(cid, {})
        snippet = c.get("text", "")[:200].replace("\n", " ")
        print(f"\n  [{rank}] {cid}  ({c.get('sample_name','?')}, {c.get('specialty','?')})")
        print(f"      \"{snippet}...\"")
    print()
