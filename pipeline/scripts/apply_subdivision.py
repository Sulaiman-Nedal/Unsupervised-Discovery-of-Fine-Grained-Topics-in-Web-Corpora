#!/usr/bin/env python3
"""apply_subdivision.py — split oversized topics into sub-topics, in place.

A topic holding a large share of the corpus is usually several subjects the
clustering failed to separate. `subdivide_topic.py` diagnoses one topic; this
applies the fix across the whole label set and rewrites the production labels.

Subdivision refines selected large topics after centroid assignment.

Sub-topics are found by k-means over the pages' E5 embeddings (the space the
topics were discovered in) and named by class-based TF-IDF over their text, the
same recipe the rest of the pipeline uses.

Labels become strings: an untouched topic stays "27", a split one becomes
"27_0", "27_1", ... so a sub-topic's parent is always readable from its name and
the hierarchy is not lost.

Boilerplate and non-prose pages are never split — they are not content, and their
rows pass through unchanged.

Usage:
  python scripts/apply_subdivision.py --lang eng
  python scripts/apply_subdivision.py --lang deu --max_share 0.03 --target_share 0.015
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from text_cleaning import clean_texts, get_combined_stopwords   # noqa: E402

# Filters for words returned as topic labels.
NO_DIGIT = r"(?u)\b[^\d\W]{2,}\b"
WEB_NOISE = frozenset({
    "http", "https", "www", "url", "href", "nbsp", "aspx", "tabid",
    "desktopdefault", "html", "htm", "php", "css", "svg", "jpg", "jpeg",
    "png", "gif", "pdf", "related", "topics", "policy", "privacy", "web",
    "datenschutzerklärung", "datenschutzerklaerung", "datenschutzerklrung",
})


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def derive_words(texts, groups, top_n=10):
    from sklearn.feature_extraction.text import TfidfVectorizer
    names = sorted(set(groups))
    corpus = [" ".join(t for t, g in zip(texts, groups) if g == n) for n in names]
    vec = TfidfVectorizer(stop_words=list(get_combined_stopwords() | WEB_NOISE),
                          token_pattern=NO_DIGIT, min_df=1, max_df=0.95)
    X = vec.fit_transform(corpus)
    vocab = np.array(vec.get_feature_names_out())
    return {n: vocab[X[i].toarray().ravel().argsort()[::-1][:top_n]].tolist()
            for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["eng", "deu"])
    ap.add_argument("--labels", default=None)
    ap.add_argument("--emb", default="embeddings/dlr_embeddings_clean2_dedup.npz")
    ap.add_argument("--corpus", default="data/DLRExternalIndexed/dlr_extra.parquet")
    ap.add_argument("--max_share", type=float, default=0.03,
                    help="split any topic holding more than this share of clean content pages")
    ap.add_argument("--target_share", type=float, default=0.015,
                    help="aim each resulting sub-topic at roughly this share")
    ap.add_argument("--max_k", type=int, default=6)
    ap.add_argument("--text_cap", type=int, default=8000,
                    help="pages sampled per topic for the keyword pass; clustering uses all")
    args = ap.parse_args()

    base = REPO / f"output/production_labels/{args.lang}"
    lab = pd.read_parquet(args.labels or base / "doc_labels.parquet")
    words = json.load(open(base / "topics.json"))

    content = ~lab.get("is_boilerplate", False) & ~lab.get("is_non_prose", False)
    clean = lab[content]
    n_clean = len(clean)
    vc = clean["label"].value_counts()
    over = [(int(t), int(c)) for t, c in vc.items() if c / n_clean > args.max_share]
    print(f"[{args.lang}] {n_clean:,} clean content pages, {len(vc)} topics; "
          f"{len(over)} exceed {args.max_share*100:.0f}%")
    if not over:
        print("nothing to split")
        return

    npz = np.load(REPO / args.emb, allow_pickle=True)
    idx = {u: i for i, u in enumerate(npz["uuids"].tolist())}
    text = pd.read_parquet(REPO / args.corpus, columns=["uuid", "plain_text"])
    text = text[text["uuid"].isin(set(clean["uuid"]))].set_index("uuid")["plain_text"]

    new_label = lab["label"].astype(str).copy()
    new_words = {k: v for k, v in words.items()}

    from sklearn.cluster import KMeans
    for t, c in over:
        k = min(args.max_k, max(2, round((c / n_clean) / args.target_share)))
        uu = clean.loc[clean["label"] == t, "uuid"].tolist()
        rows = [idx[u] for u in uu if u in idx]
        keep = [u for u in uu if u in idx]
        E = unit(npz["embeddings"][rows])
        sub = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(E)
        share = c / n_clean * 100
        print(f"  topic {t:<4} {c:>7,} pages ({share:4.1f}%) -> {k} sub-topics")

        smap = dict(zip(keep, sub))
        new_label.loc[lab["uuid"].isin(smap)] = [
            f"{t}_{smap[u]}" for u in lab.loc[lab["uuid"].isin(smap), "uuid"]]

        samp = pd.DataFrame({"uuid": keep, "sub": sub})
        if len(samp) > args.text_cap:
            samp = samp.sample(args.text_cap, random_state=0)
        txt = clean_texts(text.reindex(samp["uuid"]).fillna("").tolist())
        derived = derive_words(txt, samp["sub"].tolist())
        parent_kw = words.get(str(t), {}).get("keywords", [])
        for s, kw in derived.items():
            new_words[f"{t}_{s}"] = {"keywords": kw, "boilerplate": False,
                                     "parent": str(t), "parent_keywords": parent_kw}
            print(f"      {t}_{s}: {', '.join(kw[:7])}")
        new_words.pop(str(t), None)

    lab["label"] = new_label
    lab.to_parquet(base / "doc_labels.parquet", index=False)
    json.dump(new_words, open(base / "topics.json", "w"), indent=2, ensure_ascii=False)

    after = lab[content]["label"].value_counts()
    print(f"\n{len(after)} topics after splitting; largest now "
          f"{after.iloc[0]/n_clean*100:.1f}% (was {over[0][1]/n_clean*100:.1f}%)")
    print(f"[saved -> {base}/doc_labels.parquet, topics.json]")


if __name__ == "__main__":
    main()
