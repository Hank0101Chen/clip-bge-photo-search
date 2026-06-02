# Evaluation Data Format

This folder contains tiny sample files that document the expected format for retrieval-quality evaluation.

The full Flickr30K-CN test set is **not included** in this repository because it is large and should be obtained from its original source/license.

## Ground Truth

`sample_ground_truth.json`:

```json
{
  "queries": [
    {
      "query_id": "q001",
      "query": "海邊夕陽",
      "relevant_photo_ids": ["photo_001", "photo_003"]
    }
  ]
}
```

## Predictions

`sample_predictions.json`:

```json
{
  "predictions": [
    {
      "query_id": "q001",
      "retrieved_photo_ids": ["photo_003", "photo_002", "photo_001"]
    }
  ]
}
```

## Run Evaluation

```bash
python worker/scripts/evaluate_rankings.py   --ground-truth eval/sample_ground_truth.json   --predictions eval/sample_predictions.json
```

Metrics:

```text
Recall@1 / Recall@5 / Recall@10
Precision@1 / Precision@5 / Precision@10
nDCG@1 / nDCG@5 / nDCG@10
MAP
```
