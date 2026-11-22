#!/usr/bin/env python3
"""build_sample.py — build the Stage 2 benchmark sample with boilerplate filtering.

Excludes legal/nav/directory-listing template pages (via url_filters.is_boilerplate_url)
before drawing the random sample, so the benchmark reflects real research content
rather than the corpus's most common page TYPE.

Usage:
  python scripts/build_sample.py \
      --embeddings embeddings/dlr_embeddings_clean.npz \
      --parquet    data/DLRExternalIndexed/dlr_extra.parquet \
      --output_dir output/stage2_clean/sample_data \
      --sample_size 50000 --seed 42
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from url_filters import is_boilerplate_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lang", type=str, default=None,
                        help="Restrict to a single language code (e.g. eng, deu). "
                             "Omit to keep all languages.")
    args = parser.parse_args()

    print(f"Loading embeddings: {args.embeddings}")
    data = np.load(args.embeddings)
    uuids, embs = data["uuids"], data["embeddings"]
    print(f"  {len(uuids):,} docs")

    print(f"\nLoading URL metadata: {args.parquet}")
    df_meta = pd.read_parquet(args.parquet, columns=["uuid", "url", "url_path", "url_query", "lang"])
    df_meta = df_meta.set_index("uuid").reindex(uuids)

    print("Flagging boilerplate / code-doc URLs...")
    df_meta["is_boilerplate"] = df_meta.apply(
        lambda r: is_boilerplate_url(r["url_path"], r["url_query"]), axis=1
    )
    n_boiler = int(df_meta["is_boilerplate"].sum())
    print(f"  Boilerplate/code URLs flagged: {n_boiler:,} / {len(df_meta):,} "
          f"({n_boiler / len(df_meta) * 100:.2f}%)")

    keep_mask = ~df_meta["is_boilerplate"].values
    if args.lang is not None:
        lang_mask = (df_meta["lang"] == args.lang).values
        keep_mask = keep_mask & lang_mask
        print(f"  Restricting to lang={args.lang!r}: {int(lang_mask.sum()):,} docs of that language")

    uuids_kept, embs_kept = uuids[keep_mask], embs[keep_mask]
    print(f"  {len(uuids_kept):,} docs remain after filtering")

    rng = np.random.RandomState(args.seed)
    n_sample = min(args.sample_size, len(uuids_kept))
    idx = rng.choice(len(uuids_kept), n_sample, replace=False)
    idx.sort()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out / "sample_embeddings.npz",
                         uuids=uuids_kept[idx], embeddings=embs_kept[idx])

    df_full = pd.read_parquet(args.parquet, columns=["uuid", "plain_text", "lang", "url"])
    df_full = df_full.set_index("uuid").reindex(uuids_kept[idx].tolist()).reset_index()
    df_full.to_parquet(out / "sample_data.parquet", index=False)

    print(f"\nSample saved: {len(df_full):,} docs → {out}")


if __name__ == "__main__":
    main()
