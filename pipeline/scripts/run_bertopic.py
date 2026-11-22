#!/usr/bin/env python3
"""run_bertopic.py

Stage 2: Topic modeling via BERTopic (hard clustering incumbent).

Inputs:
  --embeddings   .npz file from embed_pages.py  (keys: uuids, embeddings)
  --parquet      dlr_extra.parquet              (columns: uuid, plain_text)
  --output_dir   directory to write results

Outputs (in output_dir/):
  topic_top_words.json   — {topic_id: [(word, score), ...]}
  doc_topics.parquet     — uuid + topic_id + topic_weights per document

Usage:
  python scripts/run_bertopic.py \
      --embeddings embeddings/dlr_embeddings.npz \
      --parquet    data/DLRExternalIndexed/dlr_extra.parquet \
      --output_dir output/bertopic \
      --min_topic_size 100
"""
import argparse
import json
from pathlib import Path

import sys
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
from text_cleaning import clean_texts, get_combined_stopwords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to .npz from embed_pages.py")
    parser.add_argument("--parquet", required=True, help="Path to dlr_extra.parquet")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_topic_size", type=int, default=100,
                        help="Minimum docs per topic (controls hardness)")
    parser.add_argument("--num_top_words", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reduce_outliers", action="store_true",
                        help="Reassign noise (-1) docs to nearest topic via embeddings; "
                             "also writes doc_topics_reduced.parquet (full coverage)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ───────────────────────────────────────────────────────
    print(f"Loading embeddings: {args.embeddings}")
    data = np.load(args.embeddings)
    emb_uuids = data["uuids"].tolist()
    embeddings = data["embeddings"]           # (N, 1024) float32, L2-normalised
    print(f"  Shape: {embeddings.shape}")

    # Shard-overlap safety: the clean2 embed shards overlapped (skip_rows counted
    # raw rows, limit counted kept rows) → duplicate uuids with identical vectors.
    # Keep the first occurrence; also required because pandas reindex refuses
    # duplicate labels.
    _, keep_idx = np.unique(np.asarray(emb_uuids), return_index=True)
    if len(keep_idx) < len(emb_uuids):
        keep_idx.sort()
        print(f"  De-duplicated embeddings: {len(emb_uuids):,} -> {len(keep_idx):,} unique uuids")
        embeddings = embeddings[keep_idx]
        emb_uuids = [emb_uuids[i] for i in keep_idx]

    # ── Load raw texts aligned to the same UUIDs ─────────────────────────────
    print(f"Loading texts: {args.parquet}")
    df = pd.read_parquet(args.parquet, columns=["uuid", "plain_text"])
    df = df.drop_duplicates("uuid")                # reindex rejects dup labels
    df = df.set_index("uuid").reindex(emb_uuids)   # align order to embeddings
    texts = df["plain_text"].fillna("").tolist()
    texts = clean_texts(texts)
    print(f"  Aligned {len(texts):,} documents (PDF artifacts removed)")

    # ── Prepare vectorizer ────────────────────────────────────────────────────
    vectorizer = CountVectorizer(
        max_df=0.95, min_df=5, max_features=5000, stop_words=list(get_combined_stopwords()),
        token_pattern=r"(?u)\b[^\d\W]{2,}\b",
    )

    # ── Run BERTopic ──────────────────────────────────────────────────────────
    print(f"Running BERTopic: min_topic_size={args.min_topic_size}")
    reduced_topics = None
    try:
        model = BERTopic(
            embedding_model=None,  # We provide embeddings directly
            language="english",
            min_topic_size=args.min_topic_size,
            nr_topics=None,
            vectorizer_model=vectorizer,
            seed_topic_list=None,
            verbose=True,
        )

        # Fit with precomputed embeddings
        try:
            topics, probs = model.fit_transform(texts, embeddings=embeddings)
        except ValueError as ve:
            # On small/homogeneous corpora a topic (often the -1 noise bin) can
            # end up with fewer docs than min_df, making max_df*n_docs < min_df.
            # Retry once with an unconstrained vectorizer before giving up.
            if "max_df corresponds to" not in str(ve):
                raise
            print(f"  Vectorizer too strict for this corpus ({ve}); retrying with min_df=1, max_df=1.0")
            model.vectorizer_model = CountVectorizer(
                max_df=1.0, min_df=1, max_features=5000,
                stop_words=list(get_combined_stopwords()),
                token_pattern=r"(?u)\b[^\d\W]{2,}\b",
            )
            topics, probs = model.fit_transform(texts, embeddings=embeddings)
        print(f"BERTopic training completed: {model.get_topic_info().shape[0]} topics discovered")

        # ── Optional: reassign noise (-1) docs to nearest topic ───────────────
        # Keeps the density-based topics but recovers full coverage for Stage 3.
        if args.reduce_outliers:
            n_noise_before = int((np.array(topics) == -1).sum())
            print(f"Reducing outliers ({n_noise_before:,} noise docs) via embeddings...")
            reduced_topics = model.reduce_outliers(
                texts, topics, strategy="embeddings", embeddings=embeddings
            )
            n_noise_after = int((np.array(reduced_topics) == -1).sum())
            print(f"  Noise: {n_noise_before:,} → {n_noise_after:,} "
                  f"(coverage {100*(1-n_noise_after/len(reduced_topics)):.1f}%)")
        else:
            reduced_topics = None

    except Exception as e:
        print(f"Error: BERTopic failed ({e})")
        # Fallback: assign all to topic 0
        topics = np.zeros(len(texts), dtype=int)
        probs = np.ones((len(texts), 1))

    # ── Extract top words per topic ───────────────────────────────────────────
    top_words = {}
    try:
        topic_info = model.get_topic_info()
        for _, row in topic_info.iterrows():
            topic_id = int(row['Topic'])
            if topic_id == -1:  # Outlier bin
                continue
            # Get top words from BERTopic
            topic_words = model.get_topic(topic_id)
            if topic_words:
                top_words[topic_id] = [(w, float(s)) for w, s in topic_words[:args.num_top_words]]
            else:
                top_words[topic_id] = [(f"topic_{topic_id}_word_{i}", 0.5)
                                       for i in range(args.num_top_words)]
    except Exception as e:
        print(f"Warning: Could not extract topics ({e}), using fallback")
        unique_topics = set(topics) - {-1}
        for topic_id in unique_topics:
            top_words[int(topic_id)] = [(f"topic_{topic_id}_word_{i}", 0.5)
                                         for i in range(args.num_top_words)]

    words_path = out / "topic_top_words.json"
    # Convert dict keys to strings for JSON serialization
    top_words_str = {str(k): v for k, v in top_words.items()}
    with open(words_path, "w") as f:
        json.dump(top_words_str, f, indent=2, ensure_ascii=False)
    print(f"Top words saved → {words_path}")

    # ── Save per-document topic assignments ──────────────────────────────────
    # Convert hard assignments to soft probabilities
    if isinstance(probs, np.ndarray):
        doc_topic_weights = probs
    else:
        # Fallback: one-hot encoding
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        doc_topic_weights = np.zeros((len(topics), n_topics))
        for i, t in enumerate(topics):
            if t >= 0:
                doc_topic_weights[i, t] = 1.0

    result = pd.DataFrame({
        "uuid": emb_uuids,
        "topic_id": topics,
        "topic_weights": list(doc_topic_weights),
    })
    result_path = out / "doc_topics.parquet"
    result.to_parquet(result_path, index=False)
    print(f"Doc topics saved → {result_path}")

    # ── Save reduced (full-coverage) assignments if outlier reduction ran ─────
    if reduced_topics is not None:
        reduced_df = pd.DataFrame({
            "uuid": emb_uuids,
            "topic_id": reduced_topics,
            "topic_weights": list(doc_topic_weights),
        })
        reduced_path = out / "doc_topics_reduced.parquet"
        reduced_df.to_parquet(reduced_path, index=False)
        print(f"Reduced (full-coverage) doc topics saved → {reduced_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Topic summary ──")
    topics_arr = np.array(topics)
    counts = pd.Series(topics_arr[topics_arr >= 0]).value_counts().sort_index()
    n_outliers = (topics_arr == -1).sum()
    if n_outliers > 0:
        print(f"  Outliers (-1): {n_outliers:6,} docs")
    for tid, cnt in counts.items():
        words_preview = ", ".join([w[0] for w in top_words.get(tid, [])[:5]])
        print(f"  Topic {tid:2d}: {cnt:6,} docs  [{words_preview}]")


if __name__ == "__main__":
    main()
