#!/usr/bin/env python3
"""filter_junk_topics.py — drop topics that are website furniture, not subjects.

The existing cleaning (`text_cleaning.is_noise_token`) works on single WORDS, so
it cannot catch these: "month, year, bw, measurement, latest" is eight ordinary
words that together are a data-table header, and it appears as EIGHT separate
BERTopic topics. That is why ~25% junk survived every earlier cleaning pass — the
signal only exists at the TOPIC level.

Four topic-level flags (a topic is junk if ANY fires; each is reported so you
can see why, and every threshold is a CLI arg):

  dup    — its top words overlap another topic's by >= --dup_jaccard.
           Real topics are distinctive; the same word list under many topic ids
           is one boilerplate template shredded into pieces.
           NB: duplicates are DEDUPED, not deleted — within each group of twins
           the largest topic is KEPT and only the redundant copies are dropped.
           Deleting every twin would throw away real science that merely got
           fragmented (e.g. rhea/tethys/hrsc, iop/doppler/lidar).
  noise  — >= --noise_share of its top words fail `is_good_label_word`
           (PDF operators, dirlist tokens, consonant garbage).
  generic— its word distribution sits closer than --min_granularity to the
           whole-corpus word distribution, i.e. it says nothing specific.
           Deliberately the weakest/last flag: a genuinely broad topic can look
           generic too, so keep this threshold low and check what it removes.
  structural — >= --struct_share of its top words are file-listing / web-chrome
           tokens (STRUCTURAL below). Added after the first production run, where
           the three flags above still let ~20% of English pages land in topics
           like "directory, file, geoportal, parent, eoweb" and "applicationpdf,
           windows, version": individually ordinary words that collectively
           describe a file listing rather than a subject.

Writes a filtered topic_top_words.json (same format in, same format out, so it
drops straight into label_by_keywords_e5.py / run_lbl2vec.py) plus a report
listing every dropped topic and which flag caught it.

Usage:
  python scripts/filter_junk_topics.py --lang eng --sub bertopic_min100 \
      --out output/rq3_e5/eng/keywords_clean.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from text_cleaning import is_good_label_word          # noqa: E402
from granularity_score import (build_vectors, cosine_distance,   # noqa: E402
                               duplicate_twins, load_topic_words)

# Words that are individually ordinary English/German but, taken together, describe
# a FILE LISTING or a WEB CHROME element rather than a subject. `is_good_label_word`
# passes every one of them, which is why the first pass let topics like
# "directory, file, geoportal, parent, eoweb" and "applicationpdf, windows, version"
# through — they were ~20% of English pages in the first production run.
#
# Deliberately conservative: only tokens with no plausible research meaning in this
# corpus. NOT included, though they appear in junk topics: orbit, events, time,
# minutes, meters, start, stop, line — each is genuine DLR vocabulary somewhere
# (orbital mechanics, event pages, measurement series), and removing topics for
# containing them would delete real science.
STRUCTURAL = frozenset({
    "file", "files", "filename", "directory", "dir", "folder", "parent", "index",
    "portal", "geoportal", "eoweb", "sitemap", "url", "link", "links", "page",
    "pages", "click", "login", "download", "downloads", "upload", "version",
    "windows", "applicationpdf", "pdf", "utc", "timestamp", "auxiliary",
    "metadata", "path", "browser", "navigation", "menu", "footer", "header",
    "previous", "next", "archive", "documentation", "glossary", "partner",
})


def analyse(lang, sub, top_k, dup_jaccard, noise_share, min_granularity, struct_share):
    topics, sizes = load_topic_words(lang, sub)
    topic_vec, corpus_vec = build_vectors(topics, sizes)
    twins = duplicate_twins(topics, top_k=top_k, thresh=dup_jaccard)

    # Resolve duplicate groups first: keep the largest member of each group of
    # twins, mark only the redundant copies. Uses union-find style expansion so
    # a chain of near-duplicates collapses to a single representative.
    redundant, seen = set(), set()
    for t in sorted(topics):
        if t in seen or not twins[t]:
            continue
        group, stack = {t}, [t]
        while stack:
            cur = stack.pop()
            for other in twins[cur]:
                if other not in group:
                    group.add(other)
                    stack.append(other)
        seen |= group
        keeper = max(group, key=lambda x: sizes.get(x, 0))
        redundant |= (group - {keeper})

    rows = []
    for t in sorted(topics):
        words = [w for w, _ in sorted(topics[t].items(), key=lambda kv: -kv[1])][:top_k]
        bad = sum(1 for w in words if not is_good_label_word(w))
        share = bad / len(words) if words else 1.0
        struct = sum(1 for w in words if w in STRUCTURAL) / len(words) if words else 0.0
        gran = cosine_distance(topic_vec[t], corpus_vec)

        flags = []
        if t in redundant:
            flags.append("dup")
        if share >= noise_share:
            flags.append("noise")
        if gran < min_granularity:
            flags.append("generic")
        if struct >= struct_share:
            flags.append("structural")

        rows.append(dict(topic=t, size=int(sizes.get(t, 0)), granularity=gran,
                         noise_share=share, struct_share=struct, twins=sorted(twins[t]),
                         kept_as_representative=bool(twins[t]) and t not in redundant,
                         flags=flags, junk=bool(flags), words=words[:6]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["eng", "deu"])
    ap.add_argument("--sub", default="bertopic_min100")
    ap.add_argument("--out", required=True, help="filtered topic_top_words.json")
    ap.add_argument("--report", default=None, help="defaults to <out>.report.json")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--dup_jaccard", type=float, default=0.6)
    ap.add_argument("--noise_share", type=float, default=0.5)
    ap.add_argument("--min_granularity", type=float, default=0.70)
    ap.add_argument("--struct_share", type=float, default=0.30,
                    help="flag a topic when this share of its top words are file-listing / "
                         "web-chrome tokens (see STRUCTURAL)")
    args = ap.parse_args()

    rows = analyse(args.lang, args.sub, args.top_k, args.dup_jaccard,
                   args.noise_share, args.min_granularity, args.struct_share)
    junk = [r for r in rows if r["junk"]]
    keep = [r for r in rows if not r["junk"]]

    bar = "=" * 88
    print(bar)
    print(f"JUNK FILTER [{args.lang}/{args.sub}]  {len(junk)}/{len(rows)} topics dropped "
          f"({len(junk)/len(rows)*100:.1f}%)")
    print(bar)
    print(f"{'topic':<7}{'size':>7}{'gran':>7}{'noise':>7}{'strct':>7}  {'flags':<26}words")
    print("-" * 88)
    for r in sorted(junk, key=lambda r: r["granularity"]):
        print(f"T{r['topic']:<6}{r['size']:>7,}{r['granularity']:>7.3f}"
              f"{r['noise_share']:>7.2f}{r['struct_share']:>7.2f}  "
              f"{','.join(r['flags']):<26}{', '.join(r['words'][:5])}")
    from collections import Counter
    c = Counter(f for r in junk for f in r["flags"])
    print("-" * 88)
    print("flag counts (a topic can trip several): " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    print(f"kept {len(keep)} topics")

    # write filtered keywords in the SAME format the labellers expect
    src = json.load(open(REPO / f"output/stage2_clean2/{args.lang}/{args.sub}/topic_top_words.json"))
    kept_ids = {str(r["topic"]) for r in keep}
    out = {t: ws for t, ws in src.items() if t in kept_ids}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[saved filtered keywords -> {args.out}]")

    rep = args.report or (args.out + ".report.json")
    json.dump({"lang": args.lang, "sub": args.sub,
               "thresholds": dict(top_k=args.top_k, dup_jaccard=args.dup_jaccard,
                                  noise_share=args.noise_share,
                                  min_granularity=args.min_granularity,
                                  struct_share=args.struct_share),
               "n_total": len(rows), "n_junk": len(junk), "n_kept": len(keep),
               "topics": rows}, open(rep, "w"), ensure_ascii=False, indent=2)
    print(f"[saved report -> {rep}]")


if __name__ == "__main__":
    main()
