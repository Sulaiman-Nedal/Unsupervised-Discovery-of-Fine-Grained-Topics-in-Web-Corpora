#!/usr/bin/env python3
"""embed_pages.py

Generate multilingual-e5-large-instruct embeddings for the DLR web corpus.
Reads uuid + plain_text from the input parquet, filters to eng/deu pages
with sufficient content, and writes uuid + embedding to a .npz file.

Usage:
  python scripts/embed_pages.py \
      --input data/DLRExternalIndexed/dlr_extra.parquet \
      --output embeddings/dlr_embeddings.npz \
      --batch_size 128 \
      --num_threads 8 \
      --device cuda
"""
import argparse
import time
from pathlib import Path

import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, str(Path(__file__).parent))
from text_cleaning import clean_text

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
DEFAULT_TASK = "Identify the main research topic of this web page"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def embed_batch(texts, tokenizer, model, device, task, max_length=512):
    inputs = tokenizer(
        [f"Instruct: {task}\nQuery: {t}" for t in texts],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    emb = mean_pooling(out, inputs["attention_mask"])
    return F.normalize(emb, p=2, dim=1).cpu().numpy().astype(np.float32)


def embed_documents_chunked(texts, tokenizer, model, device, task,
                            max_length, max_chunks, batch_size):
    """Embed each document by splitting it into ≤max_length-token chunks,
    embedding each chunk with E5, then mean-pooling the (L2-normalized) chunk
    vectors and re-normalizing. Reads the whole document only if it fits in
    max_chunks chunks; text after the last chunk is dropped. Returns one
    (1024,) vector per input text.
    """
    # Leave room for the instruct prefix + 2 special tokens within max_length.
    prefix_len = len(tokenizer.encode(f"Instruct: {task}\nQuery: ",
                                      add_special_tokens=False))
    content_len = max_length - prefix_len - 2
    if content_len < 32:
        content_len = max_length - 2   # degenerate fallback

    # Build a flat list of all chunk texts, tracking chunks-per-document.
    all_chunk_texts = []
    doc_chunk_counts = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode("unknown", add_special_tokens=False)
        windows = [ids[i:i + content_len] for i in range(0, len(ids), content_len)]
        windows = windows[:max_chunks]
        chunk_texts = [tokenizer.decode(w) for w in windows]
        all_chunk_texts.extend(chunk_texts)
        doc_chunk_counts.append(len(chunk_texts))

    # Embed every chunk (each chunk re-gets the instruct prefix in embed_batch).
    chunk_embs = []
    for i in range(0, len(all_chunk_texts), batch_size):
        chunk_embs.append(
            embed_batch(all_chunk_texts[i:i + batch_size], tokenizer, model,
                        device, task, max_length)
        )
    chunk_embs = np.vstack(chunk_embs)   # (total_chunks, 1024)

    # Mean-pool chunk vectors back into one vector per document, re-normalize.
    out = np.empty((len(texts), chunk_embs.shape[1]), dtype=np.float32)
    pos = 0
    for j, n in enumerate(doc_chunk_counts):
        pooled = chunk_embs[pos:pos + n].mean(axis=0)
        pooled /= (np.linalg.norm(pooled) + 1e-9)
        out[j] = pooled.astype(np.float32)
        pos += n
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to dlr_extra.parquet")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_threads", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min_tokens", type=int, default=20,
                        help="Minimum whitespace-token count to keep a page")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max RAW rows to scan after --skip_rows; 0 = all. Counts the "
                             "same unit as --skip_rows so shard [skip, skip+limit) "
                             "partitions the parquet exactly (no overlap, no gaps).")
    parser.add_argument("--skip_rows", type=int, default=0, help="Skip this many rows at the start (for sharding)")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="E5 instruction prompt prepended to each document")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max tokens per chunk (E5 hard limit is 512)")
    parser.add_argument("--chunk", action="store_true",
                        help="Split long docs into chunks and mean-pool (full-doc coverage)")
    parser.add_argument("--max_chunks", type=int, default=8,
                        help="Max chunks per doc when --chunk is set (bounds compute)")
    args = parser.parse_args()
    print(f"Task instruction: {args.task!r}")
    print(f"Max length: {args.max_length}  chunk={args.chunk}  max_chunks={args.max_chunks}")

    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream parquet in chunks so plain_text is never fully in RAM at once.
    # Each chunk is embedded then discarded; only float32 embeddings accumulate.
    READ_CHUNK = 50_000
    pf = pq.ParquetFile(args.input)
    total_meta = pf.metadata
    print(f"Parquet: {total_meta.num_rows:,} rows, {total_meta.num_row_groups} row group(s)")
    print(f"Streaming in chunks of {READ_CHUNK:,} rows …")

    all_embeddings = []
    all_uuids = []
    total_seen = 0
    total_kept = 0
    total_skipped = 0
    limit_hit = False
    t_start = time.time()

    for arrow_batch in pf.iter_batches(batch_size=READ_CHUNK,
                                        columns=["uuid", "plain_text", "lang"]):
        if limit_hit:
            break

        chunk = arrow_batch.to_pandas()

        # Skip rows for sharding
        if args.skip_rows > 0 and total_skipped < args.skip_rows:
            remaining_to_skip = args.skip_rows - total_skipped
            if len(chunk) <= remaining_to_skip:
                total_skipped += len(chunk)
                continue
            chunk = chunk.iloc[remaining_to_skip:]
            total_skipped += remaining_to_skip

        # Raw-row limit, applied BEFORE any filtering: counts the same unit as
        # --skip_rows, so shard [skip, skip+limit) partitions the parquet
        # exactly. (The old behaviour counted KEPT rows against --limit, which
        # made each shard scan past its boundary into the next shard's rows —
        # the clean2 11.4%-duplicate-embeddings bug.)
        if args.limit > 0:
            remaining_raw = args.limit - total_seen
            if remaining_raw <= 0:
                limit_hit = True
                break
            if len(chunk) >= remaining_raw:
                chunk = chunk.iloc[:remaining_raw]
                limit_hit = True   # process this final partial chunk, then stop

        total_seen += len(chunk)

        chunk = chunk[chunk["lang"].isin(["eng", "deu"])].copy()
        chunk["plain_text"] = chunk["plain_text"].fillna("")
        chunk = chunk[chunk["plain_text"].str.split().str.len() >= args.min_tokens]

        if len(chunk) == 0:
            continue

        texts = [clean_text(t) for t in chunk["plain_text"].tolist()]
        uuids = chunk["uuid"].tolist()

        # Re-filter: cleaning may have reduced some docs below min_tokens
        valid = [(u, t) for u, t in zip(uuids, texts) if len(t.split()) >= args.min_tokens]
        if not valid:
            continue
        uuids, texts = zip(*valid)

        if args.chunk:
            all_embeddings.append(
                embed_documents_chunked(list(texts), tokenizer, model, device,
                                        args.task, args.max_length, args.max_chunks,
                                        args.batch_size)
            )
        else:
            for i in range(0, len(texts), args.batch_size):
                all_embeddings.append(
                    embed_batch(texts[i : i + args.batch_size], tokenizer, model, device, args.task, args.max_length)
                )

        all_uuids.extend(uuids)
        total_kept += len(chunk)
        elapsed_so_far = time.time() - t_start
        rate = total_kept / elapsed_so_far if elapsed_so_far > 0 else 0
        print(f"  seen {total_seen:,}  kept {total_kept:,}  "
              f"({rate:.0f} pages/sec)", flush=True)

    embeddings = np.vstack(all_embeddings)
    elapsed = time.time() - t_start
    n = len(all_uuids)
    print(f"\nEmbedded {n:,} pages in {elapsed:.1f}s  ({n/elapsed:.0f} pages/sec)")
    print(f"Shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    np.savez_compressed(out_path, uuids=np.array(all_uuids), embeddings=embeddings)
    print(f"Saved → {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
