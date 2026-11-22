#!/usr/bin/env python3
"""detect_non_prose.py — flag pages that are machine output rather than writing.

A crawl of any large site picks up things that are not documents: source files,
build and test logs, data dumps, directory indexes. They are perfectly coherent
to a topic model — they share vocabulary, so they cluster tightly — but nobody
searches for them, and they crowd out real subjects. In this corpus one such
cluster held 12.6% of English pages (a mirrored simulator repository: C++ source,
compiler output, test logs, simulation dumps).

Deliberately DATA-NEUTRAL. No site, topic id, or project name appears here. The
three signals are properties of prose itself, so the rule transfers to any corpus
in any language:

  stopword ratio — prose is full of function words (the/and/der/und). Code and
                   logs have almost none. This is the strongest single signal and
                   needs no language detection, since the stopword list is the
                   combined multilingual one the pipeline already uses.
  symbol density — share of characters that are neither alphanumeric nor space.
                   Punctuation-heavy syntax ({};=<>) versus ordinary prose.
  digit density  — logs, tables and dumps are mostly numbers.

Measured separation on this corpus (means): real research pages sit at 28%
stopwords / 6% symbols / 5% digits; crawled code at 17% / 14% / 10%; nav-menu and
file-listing boilerplate at 7% / 16% / 32%.

The default rule is symbol density > 0.09, which on a labelled sample caught 69%
of the code pages and 83% of the boilerplate while flagging 4.6% of genuine
research pages. That last number is the honest cost: no threshold separates
cleanly, because the boundary is genuinely fuzzy — a research page that is mostly
a data table really does look like machine output. Tune with --symbol/--stopword/
--digit and check what moves.

Usage:
  python scripts/detect_non_prose.py --out output/cleaning/non_prose.parquet
  python scripts/detect_non_prose.py --symbol 0.10 --report_only
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from text_cleaning import get_combined_stopwords          # noqa: E402

STOP = get_combined_stopwords()
WORD = re.compile(r"[A-Za-zÀ-ÿ]+")
HEAD = 4000          # chars per page; enough to judge, and keeps a 616K pass quick
MIN_TOKENS = 20      # below this the ratios are noise, so the page is left alone


def features(text):
    """(stopword_ratio, symbol_density, digit_density) or None if too short."""
    t = (text or "")[:HEAD]
    toks = [w.lower() for w in WORD.findall(t)]
    if len(toks) < MIN_TOKENS:
        return None
    n = max(len(t), 1)
    return (sum(1 for w in toks if w in STOP) / len(toks),
            sum(1 for c in t if not (c.isalnum() or c.isspace())) / n,
            sum(1 for c in t if c.isdigit()) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/DLRExternalIndexed/dlr_extra.parquet")
    ap.add_argument("--out", default="output/cleaning/non_prose.parquet")
    ap.add_argument("--symbol", type=float, default=0.09, help="flag above this symbol density")
    ap.add_argument("--stopword", type=float, default=None,
                    help="also flag below this stopword ratio (off by default: on its own it "
                         "costs more real pages than it catches)")
    ap.add_argument("--digit", type=float, default=None, help="also flag above this digit density")
    ap.add_argument("--report_only", action="store_true", help="print the summary, write nothing")
    args = ap.parse_args()

    df = pd.read_parquet(REPO / args.corpus, columns=["uuid", "lang", "plain_text"])
    print(f"scoring {len(df):,} pages …")
    feats = [features(t) for t in df["plain_text"]]
    ok = np.array([f is not None for f in feats])
    F = np.zeros((len(df), 3))
    F[ok] = np.array([f for f in feats if f is not None])

    flag = ok & (F[:, 1] > args.symbol)
    if args.stopword is not None:
        flag |= ok & (F[:, 0] < args.stopword)
    if args.digit is not None:
        flag |= ok & (F[:, 2] > args.digit)

    out = pd.DataFrame({"uuid": df["uuid"], "lang": df["lang"],
                        "stopword_ratio": F[:, 0], "symbol_density": F[:, 1],
                        "digit_density": F[:, 2], "too_short": ~ok, "non_prose": flag})

    print(f"\n{'group':<22}{'pages':>10}{'share':>9}")
    print("-" * 42)
    print(f"{'flagged non-prose':<22}{flag.sum():>10,}{flag.mean() * 100:>8.1f}%")
    print(f"{'too short to judge':<22}{(~ok).sum():>10,}{(~ok).mean() * 100:>8.1f}%")
    print(f"{'kept as prose':<22}{(ok & ~flag).sum():>10,}{(ok & ~flag).mean() * 100:>8.1f}%")
    for lang, g in out.groupby("lang"):
        if lang in ("eng", "deu"):
            print(f"  {lang}: {g['non_prose'].sum():,}/{len(g):,} flagged "
                  f"({g['non_prose'].mean() * 100:.1f}%)")

    if not args.report_only:
        p = REPO / args.out
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(p, index=False)
        print(f"\n[saved -> {p}]")


if __name__ == "__main__":
    main()
