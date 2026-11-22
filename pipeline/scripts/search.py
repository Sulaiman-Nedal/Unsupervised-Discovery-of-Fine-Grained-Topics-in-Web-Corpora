#!/usr/bin/env python3
"""search.py — semantic search over the labelled DLR corpus, on the fast path.

Turns a measured benchmark result into a usable tool. The speed-up itself was
established by bench_query_encoding.py and bench_exact_search.py; this is the
same two stages wired together behind one query interface:

    query text --[ONNX fp32 encoder]--> vector --[exact GPU scan]--> ranked pages

Measured on an A100 (30 real queries, 10 repeats, full corpus):

    stage        original (CPU)   this script (GPU)
    encoding       284.8 ms            4.8 ms
    search         113.1 ms            1.15 ms
    end to end     387.7 ms            6.1 ms      ~63x

ONNX fp32 is used, NOT int8: fp32 reproduces the PyTorch vector exactly
(cosine 1.0000), whereas int8 was measured at 99.4% embedding similarity yet
changed 25.5% of real top-100 results, because this corpus's scores are packed
so tightly near the cutoff. Speed that changes the answers was not worth having.

COVERAGE (a strict project rule): every query scans EVERY page. There is no
topic filter, no pruning, no approximate index -- topic-based filtering was
tested and rejected precisely because it could not keep full coverage
(cascading dropped recall to 0.34; provably-safe pruning ran at 0.4x). Coverage
is therefore 100% by construction: the scored array has one entry per page, and
that invariant is asserted rather than assumed. `--topic` narrows what is
DISPLAYED only, after the full scan, and never changes what was searched.

Falls back to CPU when no GPU is present, and says so -- it will not report a
CPU run as though it were the fast path.

Usage:
  python scripts/search.py --lang eng --query "Mars surface images from orbiter camera"
  python scripts/search.py --lang deu --query "Windenergieanlage Rotorblatt" --top_n 20
  python scripts/search.py --lang eng --query "robot control" --topic 13
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "embeddings/dlr_embeddings_clean2_dedup.npz"
CORPUS = REPO / "data/corpus.parquet"
ONNX_DIR = REPO / "output/onnx_e5/fp32"
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
TASK = "Identify the main research topic of this web page"


def norm_url(u):
    """Collapse the ~3.8 crawls per URL this corpus holds into one key.

    Embeddings were deduplicated by document, not by address, so without this a
    single page can occupy every slot in the result list.
    """
    if not u:
        return None
    try:
        sp = urlsplit(u.strip())
    except ValueError:
        return None
    host = (sp.netloc or "").lower().removeprefix("www.")
    path = (sp.path or "/").rstrip("/") or "/"
    return f"{host}{path}" + (f"?{sp.query}" if sp.query else "")


def unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)


def load_corpus(lang):
    """Embeddings, uuids, urls and topic labels for one language."""
    npz = np.load(EMB, allow_pickle=True)
    uu, E = npz["uuids"].tolist(), npz["embeddings"]
    meta = pd.read_parquet(CORPUS, columns=["uuid", "url", "lang"])
    want = set(meta.loc[meta["lang"] == lang, "uuid"])
    keep = np.fromiter((u in want for u in uu), dtype=bool, count=len(uu))
    uu = [u for u, k in zip(uu, keep) if k]
    E = unit(E[keep])

    lab = pd.read_parquet(REPO / f"output/production_labels/{lang}/doc_labels.parquet")
    topics = json.load(open(REPO / f"output/production_labels/{lang}/topics.json"))
    return uu, E, dict(zip(meta.uuid, meta.url)), \
        dict(zip(lab.uuid, lab.label.astype(str))), topics


class Encoder:
    """ONNX fp32 on GPU when available; PyTorch CPU otherwise. Never pretends."""

    def __init__(self, prefer_gpu=True):
        import torch
        self.torch = torch
        self.mode = "pytorch_cpu"
        self.model = None
        if prefer_gpu and torch.cuda.is_available() and ONNX_DIR.exists():
            try:
                import onnxruntime as ort
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from transformers import AutoTokenizer
                so = ort.SessionOptions()
                so.intra_op_num_threads = len(os.sched_getaffinity(0))
                so.inter_op_num_threads = 1
                self.tok = AutoTokenizer.from_pretrained(ONNX_DIR)
                m = ORTModelForFeatureExtraction.from_pretrained(
                    ONNX_DIR, provider="CUDAExecutionProvider", session_options=so)
                # onnxruntime silently falls back to CPU when a CUDA library is
                # missing; refuse to call that the fast path
                if "CUDAExecutionProvider" in m.model.get_providers():
                    self.model, self.mode = m, "onnx_fp32_gpu"
            except Exception as e:                       # noqa: BLE001
                print(f"[onnx unavailable: {type(e).__name__}: {e}] falling back", flush=True)
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(MODEL_NAME, device="cpu")

    def __call__(self, text):
        text = f"Instruct: {TASK}\nQuery: {text}"
        if self.mode != "onnx_fp32_gpu":
            return self.st.encode([text], normalize_embeddings=True,
                                  show_progress_bar=False)[0].astype(np.float32)
        t = self.torch
        inp = {k: v.to("cuda") for k, v in
               self.tok([text], padding=True, truncation=True, return_tensors="pt").items()}
        with t.no_grad():
            out = self.model(**inp)
        m = inp["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        emb = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return t.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy()[0].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["eng", "deu"])
    ap.add_argument("--query", required=True)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--topic", default=None,
                    help="show only hits in this topic. A DISPLAY filter applied after the "
                         "full scan — it never reduces what was searched.")
    ap.add_argument("--cpu", action="store_true", help="force the CPU path")
    args = ap.parse_args()

    print(f"loading {args.lang} corpus ...", flush=True)
    uuids, E, urls, labels, topics = load_corpus(args.lang)
    print(f"  {len(uuids):,} pages x {E.shape[1]} dims", flush=True)

    enc = Encoder(prefer_gpu=not args.cpu)
    print(f"  encoder: {enc.mode}", flush=True)

    # Everything below the "setup" line is paid ONCE by a running service: the
    # ONNX session, the CUDA context, and moving the corpus onto the device.
    # Timing the first call instead would report cold-start cost as query cost.
    t0 = time.perf_counter()
    Et = None
    try:
        import torch
        if torch.cuda.is_available() and not args.cpu:
            Et = torch.from_numpy(np.ascontiguousarray(E)).to("cuda")
            qw = torch.from_numpy(enc("warm up")).to("cuda")
            _ = (Et @ qw).cpu().numpy()
            torch.cuda.synchronize()
            backend = "gpu"
        else:
            backend = "cpu"
    except ImportError:
        backend = "cpu"
    if Et is None:
        enc("warm up")
        backend = "cpu"
    t_setup = (time.perf_counter() - t0) * 1000
    print(f"  setup (session + corpus onto {backend}): {t_setup:,.0f} ms — paid once\n", flush=True)

    t0 = time.perf_counter()
    qv = enc(args.query)
    t_enc = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if Et is not None:
        import torch
        scores = (Et @ torch.from_numpy(qv).to("cuda")).cpu().numpy()
    else:
        scores = E @ qv
    t_search = (time.perf_counter() - t0) * 1000

    # coverage is not a hope: one score per page, every page scored
    assert scores.shape[0] == len(uuids), "every page must be scored — coverage is 100%"

    order = np.argsort(-scores)
    print(f"\nencoded in {t_enc:.1f} ms ({enc.mode}) | searched "
          f"{len(uuids):,} pages in {t_search:.1f} ms ({backend}) | 100% coverage\n")
    print(f'"{args.query}"\n' + "-" * 100)

    shown, seen = 0, set()
    for i in order:
        t = labels.get(uuids[i], "?")
        if args.topic and t != args.topic:
            continue
        key = norm_url(urls.get(uuids[i]))
        if key in seen:          # same page, crawled repeatedly — show it once
            continue
        seen.add(key)
        kw = ", ".join(topics.get(t, {}).get("keywords", [])[:4])
        print(f"{scores[i]:.4f}  [{t:<6}] {kw[:42]:<42} {(urls.get(uuids[i]) or '')[:70]}")
        shown += 1
        if shown >= args.top_n:
            break
    if args.topic:
        print(f"\n(--topic {args.topic} filtered the DISPLAY only; all {len(uuids):,} "
              f"pages were still scored)")


if __name__ == "__main__":
    main()
