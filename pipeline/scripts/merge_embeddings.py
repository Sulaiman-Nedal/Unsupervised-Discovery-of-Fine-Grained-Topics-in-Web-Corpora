#!/usr/bin/env python3
"""Merge sharded embedding .npz files into one."""
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", required=True, help="Input .npz shard files in order")
    parser.add_argument("--output", required=True, help="Output merged .npz file")
    args = parser.parse_args()

    all_uuids = []
    all_embeddings = []

    for path in args.parts:
        print(f"Loading {path} ...")
        data = np.load(path)
        all_uuids.append(data["uuids"])
        all_embeddings.append(data["embeddings"])
        print(f"  {len(data['uuids']):,} docs, shape {data['embeddings'].shape}")

    uuids = np.concatenate(all_uuids)
    embeddings = np.vstack(all_embeddings)
    print(f"\nMerged: {len(uuids):,} docs, shape {embeddings.shape}")

    # Defensive dedup: if shards overlapped (as in the clean2 skip/limit-unit
    # bug), the same uuid appears in several parts with an identical vector.
    # Keep the first occurrence and report LOUDLY — a nonzero count means the
    # sharding upstream is wrong and should be investigated.
    _, keep_idx = np.unique(uuids, return_index=True)
    if len(keep_idx) < len(uuids):
        keep_idx.sort()
        n_dupes = len(uuids) - len(keep_idx)
        print(f"\n*** WARNING: {n_dupes:,} duplicate uuids across shards "
              f"({n_dupes/len(uuids)*100:.2f}%) — shard boundaries overlap! ***")
        print(f"*** Keeping first occurrence: {len(uuids):,} -> {len(keep_idx):,} docs ***")
        uuids = uuids[keep_idx]
        embeddings = embeddings[keep_idx]

    np.savez_compressed(args.output, uuids=uuids, embeddings=embeddings)
    print(f"Saved → {args.output}  ({Path(args.output).stat().st_size / 1e9:.2f} GB)")

if __name__ == "__main__":
    main()
