#!/usr/bin/env python3
"""granularity_score.py — put a number on "how broad / how fine" a topic is.

Idea — a word-specialization score, after the level-specialization measure in
HyHTM (Shahid, Anand, Srikanth, Bhatia, Krishnamurthy & Puri, "HyHTM: Hyperbolic
Geometry based Hierarchical Topic Models", Findings of ACL 2023, arXiv:2305.09258).
HyHTM compares each hierarchy level's topic-word distribution against the whole
corpus's and shows the distance grows from root to leaf; we apply the same
comparison per topic, so a single topic gets a granularity number of its own.

Compare each topic's word-usage pattern to the WHOLE CORPUS's word-usage pattern.

  - A topic that "sounds like" the overall corpus (generic, common words)
    is BROAD.
  - A topic whose words are unusual / rare in the corpus as a whole is FINE.

We don't have raw documents handy here, so the corpus word distribution is
built by combining every topic's own top-word weights, scaled by how many
documents that topic has (so big topics count for more, same as they would
if we counted words directly from the corpus).

  corpus_vec   = sum over topics t of (doc_count[t] * topic_word_vec[t])
  topic_vec[t] = that topic's own top-word weights (zero elsewhere)
  score[t]     = 1 - cosine(topic_vec[t], corpus_vec)   # 0=broad, 1=fine

This is a per-topic, standalone score — no pairing with another topic needed.
Run once per (lang, model) directory; compare the averages across LDA vs
BERTopic to turn "broad vs fine-grained" into an actual number.

Usage:
  python scripts/granularity_score.py --lang eng --sub lda_k20
  python scripts/granularity_score.py --lang eng --sub bertopic_min100
  python scripts/granularity_score.py --all          # every lang x model combo
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# junk detector: topics whose top-TOPWORD_K words overlap another topic's by at
# least DUP_JACCARD (Jaccard) are near-duplicates => boilerplate, not subjects.
TOPWORD_K = 10
DUP_JACCARD = 0.6


def load_topic_words(lang, sub):
    base = REPO / f"output/stage2_clean2/{lang}/{sub}"
    tw = json.load(open(base / "topic_top_words.json"))
    dt = pd.read_parquet(base / "doc_topics.parquet")
    sizes = dt["topic_id"].value_counts().to_dict()
    topics = {}
    for t, ws in tw.items():
        t = int(t)
        if t == -1:
            continue
        topics[t] = {w: float(wt) for w, wt in ws}
    return topics, sizes


def build_vectors(topics, sizes):
    """Shared vocabulary across all topics; per-topic vector + size-weighted
    corpus vector in that same vocabulary space."""
    vocab = sorted({w for words in topics.values() for w in words})
    idx = {w: i for i, w in enumerate(vocab)}

    topic_vec = {}
    corpus_vec = np.zeros(len(vocab))
    for t, words in topics.items():
        v = np.zeros(len(vocab))
        for w, wt in words.items():
            v[idx[w]] = wt
        topic_vec[t] = v
        corpus_vec += sizes.get(t, 0) * v
    return topic_vec, corpus_vec


def cosine_distance(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return 1.0 - float(a @ b) / (na * nb)


def duplicate_twins(topics, top_k=TOPWORD_K, thresh=DUP_JACCARD):
    """Junk detector: which topics share a near-identical top-word list with
    some OTHER topic?

    A real topic is supposed to have its own distinctive words. When the same
    word list shows up under many different topic ids (e.g. the eight copies of
    "month, year, bw, measurement, latest" in eng/bertopic_min100), that is one
    boilerplate template the model has shredded into pieces -- not several
    genuine subjects. Compares top-word SETS with Jaccard overlap.

    Returns {topic: [other topics it duplicates]}.
    """
    tops = {t: {w for w, _ in sorted(ws.items(), key=lambda kv: -kv[1])[:top_k]}
            for t, ws in topics.items()}
    twins = {t: [] for t in tops}
    ids = sorted(tops)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sa, sb = tops[a], tops[b]
            union = len(sa | sb)
            if union and len(sa & sb) / union >= thresh:
                twins[a].append(b)
                twins[b].append(a)
    return twins


def score(lang, sub):
    topics, sizes = load_topic_words(lang, sub)
    topic_vec, corpus_vec = build_vectors(topics, sizes)
    twins = duplicate_twins(topics)

    rows = []
    for t in sorted(topics):
        d = cosine_distance(topic_vec[t], corpus_vec)
        top_words = sorted(topics[t].items(), key=lambda kv: -kv[1])[:5]
        rows.append(dict(topic=t, size=int(sizes.get(t, 0)),
                          granularity=d, junk=bool(twins[t]),
                          twins=sorted(twins[t]),
                          words=[w for w, _ in top_words]))
    rows.sort(key=lambda r: -r["granularity"])

    clean = [r["granularity"] for r in rows if not r["junk"]]
    avg = float(np.mean([r["granularity"] for r in rows]))
    avg_clean = float(np.mean(clean)) if clean else float("nan")
    return rows, avg, avg_clean


def report(lang, sub):
    rows, avg, avg_clean = score(lang, sub)
    n_junk = sum(r["junk"] for r in rows)
    bar = "=" * 78
    print(bar)
    print(f"GRANULARITY SCORE  [{lang} / {sub}]   ({len(rows)} topics)")
    print(bar)
    print(f"{'topic':<8}{'size':>8}  {'granularity':>11}  {'junk':>5}  words")
    print("-" * 90)
    for r in rows:
        print(f"T{r['topic']:<6}{r['size']:>8,}  {r['granularity']:>11.3f}  "
              f"{'DUP' if r['junk'] else '':>5}  {', '.join(r['words'])}")
    print("-" * 90)
    print(f"AVERAGE granularity (0=broad, 1=fine): {avg:.3f}   "
          f"[excluding junk: {avg_clean:.3f}]")
    print(f"junk (duplicate top-word list): {n_junk} / {len(rows)} topics")
    return rows, avg, avg_clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["eng", "deu"])
    ap.add_argument("--sub", default="lda_k20", help="topic-model subdir, e.g. lda_k20 / bertopic_min100")
    ap.add_argument("--all", action="store_true", help="run every lang x {lda_k20, bertopic_min100} combo")
    ap.add_argument("--save", action="store_true", help="write JSON to output/granularity/")
    args = ap.parse_args()

    combos = [(args.lang, args.sub)]
    if args.all:
        combos = [(l, s) for l in ("eng", "deu") for s in ("lda_k20", "bertopic_min100")]

    summary = {}
    for lang, sub in combos:
        rows, avg, avg_clean = report(lang, sub)
        print()
        summary[f"{lang}/{sub}"] = (avg, avg_clean, sum(r["junk"] for r in rows), len(rows))
        if args.save:
            out_dir = REPO / "output/granularity"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{lang}_{sub.replace('/', '_')}.json"
            json.dump({"lang": lang, "sub": sub,
                       "average_granularity": avg,
                       "average_granularity_excl_junk": avg_clean,
                       "junk_criterion": dict(top_words=TOPWORD_K, jaccard=DUP_JACCARD),
                       "topics": rows},
                       open(path, "w"), indent=2, ensure_ascii=False)
            print(f"[saved -> {path}]\n")

    if len(combos) > 1:
        print("=" * 78)
        print("SUMMARY (average granularity, 0=broad .. 1=fine)")
        print("=" * 78)
        print(f"  {'model':<30}{'all':>8}{'excl junk':>12}{'junk':>10}")
        for k, (avg, avg_clean, n_junk, n) in summary.items():
            print(f"  {k:<30}{avg:>8.3f}{avg_clean:>12.3f}{f'{n_junk}/{n}':>10}")


if __name__ == "__main__":
    main()
