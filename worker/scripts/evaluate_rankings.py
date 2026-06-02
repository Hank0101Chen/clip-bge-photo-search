"""
Evaluate retrieval rankings against ground truth.

Usage:
    python worker/scripts/evaluate_rankings.py       --ground-truth eval/sample_ground_truth.json       --predictions eval/sample_predictions.json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    args = parser.parse_args()

    gt_data = json.loads(args.ground_truth.read_text(encoding='utf-8'))
    pred_data = json.loads(args.predictions.read_text(encoding='utf-8'))

    gt_by_id = {q['query_id']: q for q in gt_data['queries']}
    pred_by_id = {p['query_id']: p for p in pred_data['predictions']}

    rows = []
    for query_id, gt in gt_by_id.items():
        pred = pred_by_id.get(query_id)
        if not pred:
            continue
        relevant = set(gt['relevant_photo_ids'])
        retrieved = pred['retrieved_photo_ids']
        row = {'query_id': query_id, 'ap': average_precision(retrieved, relevant)}
        for k in K_VALUES:
            row[f'recall@{k}'] = recall_at_k(retrieved, relevant, k)
            row[f'precision@{k}'] = precision_at_k(retrieved, relevant, k)
            row[f'ndcg@{k}'] = ndcg_at_k(retrieved, relevant, k)
        rows.append(row)

    if not rows:
        raise SystemExit('No matching query_id between ground truth and predictions.')

    summary = {'num_queries': len(rows)}
    for key in [f'recall@{k}' for k in K_VALUES] + [f'precision@{k}' for k in K_VALUES] + [f'ndcg@{k}' for k in K_VALUES] + ['ap']:
        out_key = 'MAP' if key == 'ap' else f'mean_{key}'
        summary[out_key] = sum(row[key] for row in rows) / len(rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
