"""
Benchmark BGE-M3 top-K caption rerank latency.

The benchmark compares:
  1. on_the_fly: encode query + top-K captions on every search
  2. precomputed: encode top-K captions once, then each search only encodes query

Usage:
    python worker/scripts/benchmark_precomputed_rerank.py --caption-file captions.txt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worker.models.encoders import get_bge_encoder


def load_captions(path: Path, top_k: int) -> list[str]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            captions = list(data.values())
        else:
            captions = data
    else:
        captions = path.read_text(encoding="utf-8").splitlines()
    captions = [str(c).strip() for c in captions if str(c).strip()]
    if len(captions) < top_k:
        raise SystemExit(f"Need at least {top_k} captions; got {len(captions)}")
    return captions[:top_k]


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_idx],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-file", type=Path, required=True)
    parser.add_argument("--query", default="海邊夕陽")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    captions = load_captions(args.caption_file, args.top_k)
    encoder = get_bge_encoder()

    _ = encoder.encode([args.query] + captions[:2])

    t0 = time.perf_counter()
    caption_vecs = encoder.encode(captions)
    precompute_caption_once_ms = (time.perf_counter() - t0) * 1000.0

    on_the_fly = []
    precomputed = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        q_vec = encoder.encode_one(args.query)
        c_vecs = encoder.encode(captions)
        _ = int(np.argmax(c_vecs @ q_vec))
        on_the_fly.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        q_vec = encoder.encode_one(args.query)
        _ = int(np.argmax(caption_vecs @ q_vec))
        precomputed.append((time.perf_counter() - t0) * 1000.0)

    result = {
        "query": args.query,
        "top_k": args.top_k,
        "repeat": args.repeat,
        "precompute_caption_once_ms": precompute_caption_once_ms,
        "on_the_fly": summarize(on_the_fly),
        "precomputed": summarize(precomputed),
        "speedup_median": statistics.median(on_the_fly) / statistics.median(precomputed),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
