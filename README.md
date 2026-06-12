# Clinical RAG Prototype + Eval

Built for Hilbi - a patient-centric, AI and blockchain-powered digital healthcare platform.

A retrieval-augmented generation system over clinical notes, featuring hybrid retrieval, pseudonymisation, and a rigorous evaluation framework.

## Architecture

Pipeline stages, in order:

1. Loader - reads mtsamples.csv into clean docs with metadata
2. Pseudonymiser - replaces PII with stable tokens (spaCy NER + regex)
3. Chunker - splits into 400-token sentence-aware chunks with overlap
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
    python -m eval.make_charts

View results:

    jupyter notebook notebooks/clinical_rag_demo.ipynb

## Dataset

MTSamples - 4,921 de-identified clinical transcription notes across 40 medical specialties, chunked into 8,600 passages (~400 tokens, 15% overlap).
Source: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions (CC0 Public Domain)

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM | Claude Haiku 4.5 | Fast, cost-effective, strong instruction-following for RAG |
| Embeddings | BAAI/bge-base-en-v1.5 (local) | Strong open-source retrieval model; zero API cost, no rate limits |
| Vector store | ChromaDB | Local, zero-infra, rich metadata filtering |
| Keyword retrieval | BM25 | Catches exact medical terms embeddings miss |
| Re-ranker | FlashRank (ms-marco-MiniLM-L-12-v2) | Cross-encoder quality with minimal latency |
| Pseudonymisation | spaCy en_core_web_trf + regex | NER-based; two-layer privacy (pre-index + pre-query) |
| Eval | Custom LLM-as-judge (Haiku) | Transparent, auditable faithfulness and relevance scoring |
| Framework | Plain Python | No abstractions - every component is readable and auditable |

## Eval results

16-question eval set across 5 categories: factoid, multi-hop, refusal, negation, ambiguous.
Each answer scored by an LLM judge (Claude Haiku) for faithfulness (grounded in retrieved
context) and answer relevance (addresses the question appropriately given its category).

| Metric | Score |
|---|---|
| Mean Faithfulness | 0.712 |
| Mean Answer Relevance | 0.853 |
| Mean Pipeline Latency | 5,943ms |
| Retrieval Specialty-Match Rate | 75.0% (6/8) |

By category:

| Category | Faithfulness | Answer Relevance |
|---|---|---|
| refusal | 1.000 | 1.000 |
| multi_hop | 0.750 | 0.917 |
| negation | 0.725 | 0.725 |
| factoid | 0.650 | 0.920 |
| ambiguous | 0.483 | 0.617 |

Ablation - retrieval specialty-match rate (8 questions with a known expected specialty):

| Configuration | Match Rate |
|---|---|
| Vector-only | 75.0% |
| Hybrid (RRF) | 75.0% |
| Hybrid + Rerank | 75.0% |

The overall rate is flat across configurations, but reranking changes which
questions succeed - see the notebook for the full discussion. Reranking
optimises for semantic relevance to query content, not specialty-label
agreement, which matters for comparative vs navigational queries.

## Key findings

- Refusal handling is perfect (1.0/1.0) - crypto, creative writing, and legal-advice requests are correctly declined while staying clinically on-topic.
- Multi-hop reasoning is strong (0.75 faithfulness, 0.92 relevance) - the system synthesises across multiple retrieved chunks rather than parroting one source.
- Ambiguous queries score lowest (0.48 / 0.62) - a deliberate safety-over-helpfulness tradeoff from the system prompt's "never use outside knowledge" instruction. A production system would add a clarification step here.
- Privacy: 6,219 PII entities redacted across 8,600 chunks via deterministic spaCy NER + regex, with a second pseudonymisation pass on every user query before it reaches the LLM.

See notebooks/clinical_rag_demo.ipynb for the full walkthrough, charts, and discussion.

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
    │   ├── eval_set.json       16 gold questions, 5 categories
    │   ├── run_eval.py          LLM-as-judge scoring
    │   ├── run_ablation.py      Retrieval ablation study
    │   └── make_charts.py       Chart generation
    ├── notebooks/
    │   └── clinical_rag_demo.ipynb
    ├── outputs/               (not tracked - results, charts)
    ├── .env.example
    ├── .gitignore
    ├── requirements.txt
    └── README.md
