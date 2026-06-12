"""
eval/run_eval.py
Runs the full RAG pipeline over eval_set.json and scores each result.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from src.generator import answer

load_dotenv()

EVAL_SET_PATH = "eval/eval_set.json"
RESULTS_CSV = "outputs/eval_results.csv"
DETAILS_JSON = "outputs/eval_details.json"
JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_CHARS_PER_SOURCE = 1200

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

JUDGE_INSTRUCTIONS = """You are an evaluation judge for a clinical RAG system. Score the following response.

QUESTION: __QUESTION__

RETRIEVED CONTEXT (what the system had access to):
__CONTEXT__

SYSTEM'S ANSWER:
__ANSWER__

CATEGORY: __CATEGORY__

Score the following on a 0.0-1.0 scale:

1. faithfulness: Are all claims in the answer directly supported by the retrieved context above?
1.0 = fully grounded, no fabrication. 0.0 = answer invents facts not in context.
It is CORRECT and FAITHFUL for the system to say it cannot answer, or note the context
does not cover something - score that 1.0 if accurate.

2. answer_relevance: Does the answer address what was asked, given the category?
For "refusal" category: 1.0 = correctly declined an out-of-scope request, 0.0 = inappropriately answered.
For other categories: 1.0 = directly and helpfully addresses the question, 0.0 = off-topic.

Respond with ONLY valid JSON, no markdown, no explanation, in this exact format:
{"faithfulness": 0.0, "answer_relevance": 0.0, "judge_notes": "one sentence justification"}"""


def judge_response(question, context, answer_text, category):
    prompt = JUDGE_INSTRUCTIONS
    prompt = prompt.replace("__QUESTION__", question)
    prompt = prompt.replace("__CONTEXT__", context)
    prompt = prompt.replace("__ANSWER__", answer_text)
    prompt = prompt.replace("__CATEGORY__", category)

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"faithfulness": None, "answer_relevance": None, "judge_notes": "PARSE_ERROR: " + raw[:100]}


def run_eval():
    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)

    Path("outputs").mkdir(exist_ok=True)

    results = []
    details = []

    print("[eval] Running " + str(len(eval_set)) + " eval questions...\n")

    for item in eval_set:
        print("  " + item["id"] + " [" + item["category"] + "] " + item["question"][:60] + "...")

        result = answer(item["question"], top_k=3)
        pipeline_latency = result["latency_ms"]

        context_parts = []
        for s in result["sources"]:
            part = "[Source " + str(s["rank"]) + "] (" + s["specialty"] + "): " + s["text"][:JUDGE_CHARS_PER_SOURCE]
            context_parts.append(part)
        context = "\n\n".join(context_parts)

        retrieved_specialties = set(s["specialty"] for s in result["sources"])
        retrieval_match = None
        if item.get("expected_specialty"):
            retrieval_match = item["expected_specialty"] in retrieved_specialties

        judge_t0 = time.time()
        judge = judge_response(item["question"], context, result["answer"], item["category"])
        judge_latency = int((time.time() - judge_t0) * 1000)

        avg_rerank = 0.0
        if result["sources"]:
            avg_rerank = sum(float(s["rerank_score"]) for s in result["sources"]) / len(result["sources"])

        row = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_specialty": item.get("expected_specialty"),
            "retrieved_specialties": ", ".join(sorted(retrieved_specialties)),
            "retrieval_match": retrieval_match,
            "num_sources": result["num_sources"],
            "avg_rerank_score": round(avg_rerank, 4),
            "pipeline_latency_ms": pipeline_latency,
            "judge_latency_ms": judge_latency,
            "faithfulness": judge.get("faithfulness"),
            "answer_relevance": judge.get("answer_relevance"),
            "judge_notes": judge.get("judge_notes", ""),
        }
        results.append(row)

        details.append({
            "id": item["id"],
            "question": item["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "judge": judge,
        })

        print("      faithfulness=" + str(row["faithfulness"]) + "  relevance=" + str(row["answer_relevance"]) + "  retrieval_match=" + str(retrieval_match))

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)

    with open(DETAILS_JSON, "w") as f:
        json.dump(details, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Mean faithfulness:     " + str(round(df["faithfulness"].mean(), 3)))
    print("Mean answer relevance: " + str(round(df["answer_relevance"].mean(), 3)))
    print("Mean pipeline latency: " + str(round(df["pipeline_latency_ms"].mean(), 0)) + "ms")

    factoid_matches = df[df["retrieval_match"].notna()]
    if len(factoid_matches):
        match_rate = factoid_matches["retrieval_match"].mean()
        print("Retrieval specialty match rate: " + str(round(match_rate * 100, 1)) + "% (" + str(int(factoid_matches["retrieval_match"].sum())) + "/" + str(len(factoid_matches)) + ")")

    print("\nBy category:")
    print(df.groupby("category")[["faithfulness", "answer_relevance"]].mean().round(3).to_string())

    print("\n[eval] Saved results to " + RESULTS_CSV)
    print("[eval] Saved details to " + DETAILS_JSON)


if __name__ == "__main__":
    run_eval()
