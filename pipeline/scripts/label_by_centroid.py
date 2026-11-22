#!/usr/bin/env python3
"""label_by_centroid.py — label every document by its nearest topic centroid,
computed in the SAME E5 embedding space the topics were discovered in.

Why this exists
---------------
The shipped labelling stage (`run_lbl2vec.py`) throws away the E5 embeddings and
trains a fresh Doc2Vec model to decide which category each page belongs to. On
both external ground-truth datasets that costs a lot of quality, and it also
refuses to label ~7% of pages:

                              20NG    Reuters      (NMI vs ground truth)
  BERTopic clusters (ceiling) 0.573    0.700
  THIS SCRIPT (E5 centroid)   0.562    0.643        100% coverage
  E5 + keyword centroid       0.520    0.484
  lbl2vec (Doc2Vec+keywords)  0.400    0.504         ~93% coverage

Two separate things were going wrong, and this script avoids both:
  1. wrong space   — Doc2Vec (300d, trained from scratch on one corpus) is much
                     weaker than the E5 embeddings we already paid to compute.
  2. lossy summary — describing a topic by its top-10 words throws away most of
                     what defines the cluster. Its centroid keeps that.

What it does NOT do: invent a category from keywords alone. That is lbl2vec's
one real advantage and the manual-taxonomy arm still needs it. Use this script
for the auto-keyword / production path, where the clusters already exist.

Usage:
  python scripts/label_by_centroid.py --emb output/groundtruth/20ng/embeddings.npz \
      --topics output/groundtruth/20ng/bertopic_min100/doc_topics.parquet \
      --out output/groundtruth/20ng/centroid_labels/doc_labels.parquet
  # optional external check, if you have ground truth:
  ... --truth data/groundtruth/20ng.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def topic_centroids(emb, uuids, doc_topics, min_docs=1):
    """Mean (unit-normalised) embedding of each topic's member documents.

    The -1 bucket is BERTopic's "noise" pile, not a topic, so it never gets a
    centroid -- but its documents still get labelled below, which is exactly how
    coverage reaches 100%.
    """
    idx = {u: i for i, u in enumerate(uuids)}
    dt = doc_topics[doc_topics["uuid"].isin(idx)]
    cents = {}
    for t, g in dt[dt["topic_id"] != -1].groupby("topic_id"):
        rows = [idx[u] for u in g["uuid"]]
        if len(rows) < min_docs:
            continue
        cents[int(t)] = unit(emb[rows].mean(axis=0))
    return cents


def assign(emb, cents, chunk=50_000):
    """Nearest centroid by cosine. Embeddings are unit-normalised, so the dot
    product IS the cosine and argmax over it is the nearest centroid.

    Chunked: the full corpus is ~616K x 1024 floats, and materialising the whole
    similarity matrix at once would need tens of GB.
    """
    names = np.array(sorted(cents))
    C = np.vstack([cents[t] for t in sorted(cents)])
    labels = np.empty(len(emb), dtype=names.dtype)
    sims = np.empty(len(emb), dtype=np.float32)
    for i in range(0, len(emb), chunk):
        s = emb[i:i + chunk] @ C.T
        j = s.argmax(axis=1)
        labels[i:i + chunk] = names[j]
        sims[i:i + chunk] = s[np.arange(len(j)), j]
    return labels, sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True,
                    help="npz with uuids + embeddings; defines the topic CENTROIDS")
    ap.add_argument("--topics", required=True, help="doc_topics.parquet defining the clusters")
    ap.add_argument("--out", required=True, help="where to write doc_labels.parquet")
    ap.add_argument("--target_emb", default=None,
                    help="npz of documents to LABEL (default: same as --emb). Set this to the "
                         "full-corpus embeddings to extend a sample's clusters to every page — "
                         "the clusters only have to be fitted once, on the sample.")
    ap.add_argument("--lang", default=None,
                    help="restrict the documents being LABELLED to this language. The corpus is "
                         "mixed eng/deu and each language has its own topic model, so without "
                         "this every page would be scored against the wrong language's topics.")
    ap.add_argument("--corpus_parquet", default="data/DLRExternalIndexed/dlr_extra.parquet",
                    help="provides the uuid -> lang mapping used by --lang")
    ap.add_argument("--exclude_junk", default=None,
                    help="a filter_junk_topics.py report json. Its junk topics KEEP their "
                         "centroids; pages landing on one are marked is_boilerplate=True rather "
                         "than pushed into the nearest real subject (which inflates genuine "
                         "topics with pages that are not about anything).")
    ap.add_argument("--exclude_non_prose", default=None,
                    help="detect_non_prose.py output. Flagged pages are dropped BEFORE centroids "
                         "are computed, so crawl artifacts (source files, build logs, data dumps) "
                         "cannot define a topic, and they are marked is_non_prose in the output "
                         "rather than filed under a subject.")
    ap.add_argument("--topic_words", default=None,
                    help="topic_top_words.json; writes a topics.json sidecar mapping each kept "
                         "topic id to its keywords, so the labels are human-readable rather than "
                         "bare integers")
    ap.add_argument("--truth", default=None, help="optional parquet with uuid+label for scoring")
    ap.add_argument("--min_docs", type=int, default=1, help="skip topics smaller than this")
    args = ap.parse_args()

    npz = np.load(args.emb, allow_pickle=True)
    uuids, emb = npz["uuids"].tolist(), unit(npz["embeddings"])
    dt = pd.read_parquet(args.topics)[["uuid", "topic_id"]]

    # Non-prose pages must not shape the topics: a mirrored code repository is
    # internally consistent enough to form its own tight cluster, which is exactly
    # how one crawl artifact ended up holding 12.6% of the English corpus.
    non_prose = set()
    if args.exclude_non_prose:
        npf = pd.read_parquet(args.exclude_non_prose, columns=["uuid", "non_prose"])
        non_prose = set(npf.loc[npf["non_prose"], "uuid"])
        before = len(dt)
        dt = dt[~dt["uuid"].isin(non_prose)]
        print(f"non-prose filter: {len(non_prose):,} pages flagged corpus-wide; "
              f"{before - len(dt):,} removed from the clustered sample before centroids")

    junk = set()
    if args.exclude_junk:
        rep = json.load(open(args.exclude_junk))
        junk = {r["topic"] for r in rep["topics"] if r["junk"]}
        print(f"junk filter: {len(junk)} of {dt['topic_id'].nunique()} topics flagged as junk")

    # Junk topics KEEP their centroids on purpose. A page whose nearest centroid is
    # a file-listing or nav-menu topic really is boilerplate, so it is labelled as
    # such rather than pushed into the nearest real subject. Reassigning instead
    # inflates genuine topics with pages that are not about anything — in the first
    # production run that alone took the largest English topic from 9.6% to 14.8%.
    # This also generalises to the ~566K pages outside the clustered sample, whose
    # original topic is unknown: nearest-centroid decides for them too.
    cents = topic_centroids(emb, uuids, dt, args.min_docs)

    if args.target_emb:
        tnpz = np.load(args.target_emb, allow_pickle=True)
        uuids, emb = tnpz["uuids"].tolist(), unit(tnpz["embeddings"])
        if args.lang:
            meta = pd.read_parquet(args.corpus_parquet, columns=["uuid", "lang"])
            want = set(meta.loc[meta["lang"] == args.lang, "uuid"])
            keep = np.array([u in want for u in uuids])
            uuids = [u for u, k in zip(uuids, keep) if k]
            emb = emb[keep]
            print(f"restricted to lang={args.lang}: {len(uuids):,} docs")
        print(f"centroids from {args.emb} -> labelling {len(uuids):,} docs from {args.target_emb}")

    labels, sims = assign(emb, cents)
    out = pd.DataFrame({"uuid": uuids, "label": labels, "similarity": sims})
    out["is_boilerplate"] = out["label"].isin(junk)
    if non_prose:
        out["is_non_prose"] = out["uuid"].isin(non_prose)
        n_np = int(out["is_non_prose"].sum())
        print(f"non-prose: {n_np:,} pages ({n_np / len(out) * 100:.1f}%) marked as crawl "
              f"artifacts rather than content")
    if junk:
        n_bp = int(out["is_boilerplate"].sum())
        clean = ~out["is_boilerplate"]
        if "is_non_prose" in out:
            clean &= ~out["is_non_prose"]
        real = out.loc[clean, "label"]
        print(f"boilerplate: {n_bp:,} pages ({n_bp / len(out) * 100:.1f}%) land on a junk "
              f"centroid and are marked, not filed under a real subject")
        if len(real):
            top = real.value_counts()
            print(f"real subjects: {len(real):,} pages across {real.nunique()} topics; "
                  f"largest holds {top.iloc[0] / len(real) * 100:.1f}% of them "
                  f"({top.iloc[0] / len(out) * 100:.1f}% of all pages)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{len(out):,} docs labelled into {len(cents)} topics  (coverage 100.0%)")
    print(f"[saved -> {args.out}]")

    if args.topic_words:
        tw = json.load(open(args.topic_words))
        words = {str(t): {"keywords": [w[0] if isinstance(w, (list, tuple)) else w
                                          for w in tw[str(t)]][:10],
                          "boilerplate": t in junk}
                 for t in sorted(cents) if str(t) in tw}
        side = Path(args.out).parent / "topics.json"
        json.dump(words, open(side, "w"), indent=2, ensure_ascii=False)
        print(f"[saved -> {side}]  ({len(words)} topics with keywords)")

    if args.truth:
        from sklearn.metrics import normalized_mutual_info_score as NMI

        truth = pd.read_parquet(args.truth, columns=["uuid", "label"]).rename(columns={"label": "true"})
        m = truth.merge(out, on="uuid").merge(dt, on="uuid")
        fair = m[m["topic_id"] != -1]
        pur = (pd.DataFrame({"y": fair["true"], "p": fair["label"]})
               .groupby("p")["y"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(fair))
        print(f"\nvs ground truth (n={len(fair):,}, noise excluded for comparability):")
        print(f"  centroid labels   NMI={NMI(fair['true'], fair['label']):.3f}  purity={pur:.3f}")
        print(f"  source clusters   NMI={NMI(fair['true'], fair['topic_id']):.3f}")


if __name__ == "__main__":
    main()
