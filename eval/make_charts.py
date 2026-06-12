"""
eval/make_charts.py
Generates summary charts from eval_results.csv and ablation_results.csv
Saves PNGs to outputs/
"""
import pandas as pd
import matplotlib.pyplot as plt

EVAL_CSV = "outputs/eval_results.csv"
ABLATION_CSV = "outputs/ablation_results.csv"
OUT_DIR = "outputs"

plt.rcParams["figure.dpi"] = 120


def chart_scores_by_category():
    df = pd.read_csv(EVAL_CSV)
    grouped = df.groupby("category")[["faithfulness", "answer_relevance"]].mean()
    grouped = grouped.reindex(["refusal", "multi_hop", "negation", "factoid", "ambiguous"])

    ax = grouped.plot(kind="bar", figsize=(8, 5), color=["#4C72B0", "#DD8452"], rot=0)
    ax.set_title("Eval Scores by Question Category")
    ax.set_ylabel("Mean score (0-1)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.1)
    ax.legend(["Faithfulness", "Answer Relevance"])
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR + "/scores_by_category.png", bbox_inches="tight")
    plt.close()
    print("Saved scores_by_category.png")


def chart_ablation():
    df = pd.read_csv(ABLATION_CSV)
    means = {
        "Vector-only": df["vector_only_match"].mean(),
        "Hybrid (RRF)": df["hybrid_rrf_match"].mean(),
        "Hybrid + Rerank": df["hybrid_rerank_match"].mean(),
    }

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(means.keys(), means.values(), color=["#8C9EBC", "#4C72B0", "#2E5C8A"])
    ax.set_title("Retrieval Specialty-Match Rate by Configuration")
    ax.set_ylabel("Match rate")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, means.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, str(round(val * 100, 1)) + "%",
                ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT_DIR + "/ablation_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved ablation_comparison.png")


def chart_latency():
    df = pd.read_csv(EVAL_CSV)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["id"], df["pipeline_latency_ms"], color="#55A868")
    ax.axhline(df["pipeline_latency_ms"].mean(), color="red", linestyle="--", linewidth=1,
               label="Mean: " + str(round(df["pipeline_latency_ms"].mean())) + "ms")
    ax.set_title("End-to-End Pipeline Latency per Question")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Question ID")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_DIR + "/latency_per_question.png", bbox_inches="tight")
    plt.close()
    print("Saved latency_per_question.png")


if __name__ == "__main__":
    chart_scores_by_category()
    chart_ablation()
    chart_latency()
    print("\nAll charts saved to outputs/")
