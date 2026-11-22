#!/usr/bin/env python3
"""run_lda.py

Classical LDA baseline for the coarse reference topics in RQ3.
Fit scikit-learn's LatentDirichletAllocation to the text in one discovery sample.

Inputs:
  --parquet      sample_data.parquet  (columns: uuid, plain_text)
  --output_dir   directory to write results

Outputs (in output_dir/):
  topic_top_words.json   — {topic_id: [[word, score], ...]}
  doc_topics.parquet     — uuid + topic_id + topic_weights per document

Usage:
  bash rq3/run_lda.sh
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).parent))
from text_cleaning import clean_texts, get_combined_stopwords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_topics", type=int, default=20)
    parser.add_argument("--num_top_words", type=int, default=15)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading texts: {args.parquet}")
    df = pd.read_parquet(args.parquet, columns=["uuid", "plain_text"])
    uuids = df["uuid"].tolist()
    texts = clean_texts(df["plain_text"].fillna("").tolist())
    print(f"  {len(texts):,} documents")

    print("Building bag-of-words representation...")
    vectorizer = CountVectorizer(
        max_df=0.95, min_df=5, max_features=2000,
        stop_words=list(get_combined_stopwords()),
        token_pattern=r"(?u)\b[^\d\W]{2,}\b",
    )
    bow = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    print(f"  Vocabulary size: {len(vocab)}")

    print(f"Running LDA: {args.num_topics} topics, max_iter={args.max_iter}")
    model = LatentDirichletAllocation(
        n_components=args.num_topics,
        max_iter=args.max_iter,
        learning_method="online",
        random_state=args.seed,
        n_jobs=-1,
    )
    doc_topics = model.fit_transform(bow)
    print(f"  Doc-topic shape: {doc_topics.shape}")

    # ── Top words per topic ───────────────────────────────────────────────────
    top_words = {}
    for topic_id, comp in enumerate(model.components_):
        top_idx = comp.argsort()[::-1][:args.num_top_words]
        top_words[topic_id] = [[vocab[i], float(comp[i])] for i in top_idx]

    words_path = out / "topic_top_words.json"
    with open(words_path, "w") as f:
        json.dump(top_words, f, indent=2, ensure_ascii=False)
    print(f"Top words saved → {words_path}")

    # ── Per-document topic assignments ────────────────────────────────────────
    topic_ids = np.argmax(doc_topics, axis=1)
    result = pd.DataFrame({
        "uuid": uuids,
        "topic_id": topic_ids,
        "topic_weights": list(doc_topics),
    })
    result_path = out / "doc_topics.parquet"
    result.to_parquet(result_path, index=False)
    print(f"Doc topics saved → {result_path}")

    print("\n── Topic summary ──")
    counts = pd.Series(topic_ids).value_counts().sort_index()
    for tid, cnt in counts.items():
        words_preview = ", ".join(w[0] for w in top_words.get(tid, [])[:5])
        print(f"  Topic {tid:2d}: {cnt:6,} docs  [{words_preview}]")


if __name__ == "__main__":
    main()
