# Clinical RAG Prototype + Eval

A retrieval-augmented generation system over clinical notes, featuring hybrid retrieval, pseudonymisation, and an evaluation framework that reports successes, failures, and tested-but-rejected fixes alike.

## Architecture

Pipeline stages, in order:

1. Loader - reads mtsamples.csv into clean docs with metadata
2. Pseudonymiser - replaces PII with stable tokens (spaCy NER + regex)
3. Chunker - splits into ~400-word sentence-aware chunks (approx. 500 tokens, depending on note style) with overlap
4. Embedder - BAAI/bge-base-en-v1.5 embeddings into ChromaDB, plus a parallel BM25 index
5. Retriever - hybrid search (vector + BM25, RRF fusion) then FlashRank re-ranking to top-3
6. Prompt builder - assembles pseudonymised query + labelled source context
7. Generator - Claude Haiku 4.5 produces a grounded, source-cited answer
8. Eval - LLM-as-judge scoring (faithfulness, answer relevance) saved to results.csv

## Setup

Clone and install:

    git clone <repo>
    cd clinical_rag
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_trf

Configure API keys:

    cp .env.example .env
    # Edit .env and add your ANTHROPIC_API_KEY

Add the dataset - place mtsamples.csv at data/raw/mtsamples.csv
(Download from https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)

## Run the pipeline

    python -m src.loader data/raw/mtsamples.csv
    python -m src.chunker
    python -m src.embedder
    python -m src.retriever
    python -m src.generator
    python -m eval.run_eval
    python -m eval.run_ablation
    python -m eval.run_retrieval_eval   # Recall@K / Hit@K / MRR vs ground truth
    python -m eval.validate_pii         # PII redaction spot-check
    python -m eval.make_charts

View results:

    jupyter notebook notebooks/clinical_rag_demo.ipynb

See `eval/EVAL_NOTES.md` for run-to-run variance and a documented, tested-and-reverted experiment.

## Dataset

MTSamples - 4,921 de-identified clinical transcription notes across 40 medical specialties, chunked into 8,600 passages (~400 words / approx. 500 tokens, 15% overlap).
Source: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions (CC0 Public Domain)

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM | Claude Haiku 4.5 | Fast, cost-effective, good instruction-following for RAG |
| Embeddings | BAAI/bge-base-en-v1.5 (local) | Solid open-source retrieval baseline; zero API cost, no rate limits |
| Vector store | ChromaDB | Local, zero-infra, rich metadata filtering |
| Keyword retrieval | BM25 | Catches exact medical terms embeddings miss |
| Re-ranker | FlashRank (ms-marco-MiniLM-L-12-v2) | Cross-encoder quality with minimal latency |
| Pseudonymisation | spaCy en_core_web_trf + regex | NER-based; two-layer privacy (pre-index + pre-query) |
| Eval | Custom LLM-as-judge (Haiku) | Transparent, auditable faithfulness and relevance scoring |
| Framework | Plain Python | No abstractions - every component is readable and auditable |

## Eval results

16-question eval set across 5 categories: factoid, multi-hop, refusal, negation, ambiguous. 8 of the 16 also have hand-verified `ground_truth_chunk_ids` (see Retrieval ground truth, below). Each answer is scored by an LLM judge (Claude Haiku) for faithfulness (grounded in retrieved context) and answer relevance (addresses the question appropriately given its category).

LLM-judged scores carry run-to-run sampling noise: 3 runs of the identical, committed pipeline produced the ranges below. `retrieval_match` was bit-identical across all 3 runs (75%, same 6/8 questions), since retrieval itself involves no sampling -- the variance below comes entirely from generation and judging. Full 3-run breakdown in `eval/EVAL_NOTES.md`.

| Metric | Value |
|---|---|
| Mean Faithfulness | 0.738 - 0.756 (3-run range) |
| Mean Answer Relevance | 0.862 - 0.919 (3-run range) |
| Mean Pipeline Latency | 6,840 - 7,271 ms (3-run range) |
| Retrieval Specialty-Match Rate | 75.0% (6/8) |

By category (3-run average; per-category min-max ranges in `eval/EVAL_NOTES.md`):

| Category | Faithfulness | Answer Relevance |
|---|---|---|
| refusal | 0.978 | 1.000 |
| negation | 0.950 | 0.883 |
| multi_hop | 0.750 | 0.856 |
| factoid | 0.707 | 0.950 |
| ambiguous | 0.456 | 0.736 |

### Retrieval ablation (8 questions with a known expected specialty)

| Configuration | Match Rate |
|---|---|
| Vector-only | 75.0% (6/8) |
| BM25-only | 75.0% (6/8) |
| Hybrid (RRF) | 62.5% (5/8) |
| Hybrid + Rerank | 75.0% (6/8) |

Vector-only and BM25-only each independently reach 75%, but fusing them with RRF drops the rate to 62.5% -- a chunk ranked moderately by *both* methods can be pushed out by a chunk ranked #1 by only one of them (q02). Re-ranking recovers the loss back to 75%, separately rescuing a BM25-only signal RRF had suppressed (q08) and preserving a vector-only signal RRF had kept (q13). Three distinct, traceable mechanisms -- see `eval/EVAL_NOTES.md` and `eval/run_ablation.py`.

### Retrieval ground truth: Recall@K, Hit@K, MRR

For the 8 questions above, ground-truth `chunk_id`s were found independently of the retriever -- via keyword search over `chunks.jsonl`, then verified by hand -- and measured against `retriever.retrieve()` (`eval/run_retrieval_eval.py`):

| Metric | Value |
|---|---|
| Recall@1 | 0.000 |
| Recall@3 | 0.167 |
| Recall@5 | 0.229 |
| Recall@10 | 0.292 |
| Hit@3 (at least one relevant chunk in top-3) | 0.375 |
| MRR | 0.182 |

These numbers are low in absolute terms, but tracing the misses (`eval/show_retrieved.py`) points to three distinct causes rather than "retrieval is broken":

1. **Near-duplicate chunks across specialty labels** - for q01 and q13, the top-3 are each the SAME note appearing verbatim 2-3x under different specialty tags. "Top-3" is often "1 unique chunk x N label-copies" -- a major reason the specialty-match rate above is a noisier signal than it first appears.
2. **A corpus with multiple valid answers** - for q02 and q05, the retrieved chunk is a different patient's equally-valid cardiac-cath / cholecystectomy note, not the hand-picked ground-truth `chunk_id`. Recall@K against ONE chosen chunk is necessarily stricter than "found a valid answer".
3. **A genuine topical-adjacency miss** - for q01, the retrieved chunk lists "hypertension" as a PMH comorbidity but names no antihypertensive medication -- topically near the question, not actually responsive to it.

## Key findings

- **Refusal handling is consistent**: 0.93-1.00 / 1.00 faithfulness and relevance across 3 runs on out-of-scope questions (crypto, creative writing, legal advice) -- correctly declined while staying clinically on-topic.
- **Multi-hop questions average 0.75 / 0.86** (3-run average) -- the system synthesises across multiple retrieved chunks rather than parroting one source.
- **Ambiguous queries score lowest** (0.46 / 0.74, 3-run average) -- a deliberate safety-over-helpfulness tradeoff from the system prompt's "never use outside knowledge" instruction, compounded by the near-duplicate-chunk issue above (no principled top-3 for a patient-unspecified query). We tested loosening the prompt rule; it made this category WORSE (0.467 -> 0.333 in that run), not better, and was reverted -- see `eval/EVAL_NOTES.md` for the full mechanism.
- **Privacy**: 6,219 PII entities redacted across 8,600 chunks via deterministic spaCy NER + regex (provider names such as "Dr. X" are labelled `[PROVIDER_n]`, distinct from `[PATIENT_n]`), plus a second pseudonymisation pass on every user query before it reaches the LLM. On a 23-span synthetic PII spot-check (`eval/validate_pii.py`), measured recall is 91.3% (21/23); the two misses are a street address and a standalone ZIP code, which need an address-parser rather than a regex to handle without false-positiving on lab values.

See `notebooks/clinical_rag_demo.ipynb` for the original walkthrough and charts, and `eval/EVAL_NOTES.md` for the latest correction-pass findings (run-to-run variance, the rejected prompt experiment, and the near-duplicate-chunk discovery).

## Project structure

    clinical_rag/
    ├── data/                  (not tracked - raw csv, chunks, indices)
    ├── src/
    │   ├── loader.py          CSV -> clean doc dicts
    │   ├── pseudonymise.py    PII detection and replacement
    │   ├── chunker.py         Sentence-aware sliding window
    │   ├── embedder.py        Local embeddings + ChromaDB + BM25
    │   ├── retriever.py       Hybrid RRF + FlashRank re-rank
    │   ├── prompt_builder.py  System prompt + context injection
    │   └── generator.py       Claude Haiku call + response
    ├── eval/
    │   ├── eval_set.json          16 gold questions, 8 with ground_truth_chunk_ids
    │   ├── run_eval.py             LLM-as-judge scoring
    │   ├── run_ablation.py         Retrieval ablation (vector/BM25/RRF/rerank)
    │   ├── run_retrieval_eval.py   Recall@K, Hit@K, MRR vs ground truth
    │   ├── validate_pii.py         PII redaction spot-check (synthetic cases)
    │   ├── show_retrieved.py       Diagnostic: inspect actual top-k chunks
    │   ├── make_charts.py          Chart generation
    │   └── EVAL_NOTES.md           Run-to-run variance + rejected experiments
    ├── notebooks/
    │   └── clinical_rag_demo.ipynb
    ├── outputs/               (not tracked - results, charts)
    ├── .env.example
    ├── .gitignore
    ├── requirements.txt
    └── README.md
