# Evaluation Data Format

This folder documents how to evaluate retrieval quality improvement from original CLIP retrieval to CLIP + BGE-M3 caption reranking.

The full Flickr30K-CN test set is **not included** in this repository because it is large and should be obtained from its original source/license.

Chinese-CLIP provides a preprocessed Flickr30K-CN package:

```text
https://huggingface.co/datasets/OFA-Sys/chinese-clip-eval/resolve/main/Flickr30k-CN.zip
```

## Files

```text
sample_ground_truth.json
sample_clip_predictions.json
sample_bge_rerank_predictions.json
```

## Ground Truth Format

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

## Prediction Format

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

## Compare CLIP vs BGE Rerank

```bash
python worker/scripts/evaluate_improvement.py   --ground-truth eval/sample_ground_truth.json   --baseline eval/sample_clip_predictions.json   --rerank eval/sample_bge_rerank_predictions.json
```

Metrics:

```text
Recall@1 / Recall@5 / Recall@10
Precision@1 / Precision@5 / Precision@10
nDCG@1 / nDCG@5 / nDCG@10
MAP
```
