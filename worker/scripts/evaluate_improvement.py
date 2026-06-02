"""
Compare original CLIP retrieval with BGE-M3 caption rerank.

Usage:
    python worker/scripts/evaluate_improvement.py       --ground-truth eval/sample_ground_truth.json       --baseline eval/sample_clip_predictions.json       --rerank eval/sample_bge_rerank_predictions.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


K_VALUES = (1, 5, 10)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / k


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(retrieved[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 1.0
    hits = 0
    total = 0.0
    for i, item in enumerate(retrieved):
        if item in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / len(relevant)


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    return {row['query_id']: set(row['relevant_photo_ids']) for row in data['queries']}


def load_predictions(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    return {row['query_id']: row['retrieved_photo_ids'] for row in data['predictions']}


def evaluate(gt: dict[str, set[str]], pred: dict[str, list[str]]) -> dict[str, float]:
    rows = []
    for query_id, relevant in gt.items():
        retrieved = pred.get(query_id)
        if retrieved is None:
            continue
        row = {'ap': average_precision(retrieved, relevant)}
        for k in K_VALUES:
            row[f'recall@{k}'] = recall_at_k(retrieved, relevant, k)
            row[f'precision@{k}'] = precision_at_k(retrieved, relevant, k)
            row[f'ndcg@{k}'] = ndcg_at_k(retrieved, relevant, k)
        rows.append(row)

    if not rows:
        raise SystemExit('No matching query_id between ground truth and predictions.')

    summary = {'num_queries': len(rows)}
    keys = (
        [f'recall@{k}' for k in K_VALUES]
        + [f'precision@{k}' for k in K_VALUES]
        + [f'ndcg@{k}' for k in K_VALUES]
        + ['ap']
    )
    for key in keys:
        out_key = 'MAP' if key == 'ap' else f'mean_{key}'
        summary[out_key] = sum(row[key] for row in rows) / len(rows)
    return summary


def diff_metrics(baseline: dict[str, float], rerank: dict[str, float]) -> dict[str, float]:
    return {
        key: rerank[key] - baseline[key]
        for key in baseline
        if key != 'num_queries' and key in rerank
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='Original CLIP prediction JSON')
    parser.add_argument('--rerank', type=Path, required=True, help='CLIP + BGE rerank prediction JSON')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)
    baseline = evaluate(gt, load_predictions(args.baseline))
    rerank = evaluate(gt, load_predictions(args.rerank))
    improvement = diff_metrics(baseline, rerank)

    result = {
        'baseline_name': 'Original CLIP',
        'rerank_name': 'CLIP + BGE-M3 Caption Rerank',
        'baseline': baseline,
        'rerank': rerank,
        'improvement': improvement,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
