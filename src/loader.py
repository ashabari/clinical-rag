"""
src/loader.py
Loads mtsamples.csv and returns clean document dicts.
"""
import pandas as pd
from pathlib import Path

def load_mtsamples(csv_path, min_text_length=100, max_docs=None):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find: {csv_path}")

    print(f"[loader] Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]

    df = df.rename(columns={
        "medical_specialty": "specialty",
        "transcription": "text",
        "keywords": "keywords_raw",
    })

    df["text"] = df["text"].fillna("").str.strip()
    before = len(df)
    df = df[df["text"].str.len() >= min_text_length]
    print(f"[loader] Dropped {before - len(df)} rows with missing/short text")

    df["specialty"] = df["specialty"].fillna("Unknown").str.strip()

    def parse_keywords(raw):
        if pd.isna(raw) or str(raw).strip() == "":
            return []
        return [k.strip().lower() for k in str(raw).split(",") if k.strip()]

    df["keywords"] = df["keywords_raw"].apply(parse_keywords)

    for col in ("description", "sample_name"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").str.strip()

    df = df.reset_index(drop=True)
    df["doc_id"] = df.index.map(lambda i: f"doc_{i:04d}")

    if max_docs:
        df = df.head(max_docs)

    docs = []
    for _, row in df.iterrows():
        docs.append({
            "doc_id":      row["doc_id"],
            "text":        row["text"],
            "description": row["description"],
            "specialty":   row["specialty"],
            "sample_name": row.get("sample_name", ""),
            "keywords":    row["keywords"],
            "source_file": "mtsamples.csv",
        })

    specialties = sorted({d["specialty"] for d in docs})
    print(f"[loader] ✓ Loaded {len(docs)} documents")
    print(f"[loader] ✓ {len(specialties)} specialties: {', '.join(specialties[:8])}...")
    print(f"[loader] ✓ Avg note length: {sum(len(d['text']) for d in docs)//len(docs):,} chars")
    return docs

if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/raw/mtsamples.csv"
    docs = load_mtsamples(csv)
    print("\n--- Sample doc ---")
    d = docs[0]
    print(f"  doc_id:    {d['doc_id']}")
    print(f"  specialty: {d['specialty']}")
    print(f"  description: {d['description']}")
    print(f"  text[:200]: {d['text'][:200]!r}")
