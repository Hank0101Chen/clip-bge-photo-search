"""
Backfill BGE-M3 caption embeddings for existing rows.

Usage:
    python worker/scripts/backfill_caption_embeddings.py --style all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.config import settings
from api.db import db_cursor
from worker.models.encoders import get_bge_encoder


STYLE_TO_COLUMN = {
    "A": "caption_a_embedding",
    "B": "caption_b_embedding",
}


def fetch_missing(style: str, limit: int | None):
    column = STYLE_TO_COLUMN[style]
    sql = f"""
        SELECT cloud_id, caption ->> %s
        FROM photos
        WHERE caption ? %s
          AND caption ->> %s IS NOT NULL
          AND {column} IS NULL
        ORDER BY cloud_id
    """
    params = [style, style, style]
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with db_cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def update(style: str, rows, batch_size: int) -> int:
    if not rows:
        return 0
    column = STYLE_TO_COLUMN[style]
    encoder = get_bge_encoder()
    updated = 0
    with db_cursor(commit=True) as cur:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            vecs = encoder.encode([row[1] for row in chunk], batch_size=batch_size)
            for (cloud_id, _), vec in zip(chunk, vecs):
                cur.execute(
                    f"""
                    UPDATE photos
                    SET {column} = %s,
                        caption_embedding_model = %s
                    WHERE cloud_id = %s
                    """,
                    (vec.astype(np.float32).tolist(), settings.text_rerank_model_id, cloud_id),
                )
                updated += 1
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=["A", "B", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=settings.text_rerank_batch_size)
    args = parser.parse_args()

    styles = ["A", "B"] if args.style == "all" else [args.style]
    for style in styles:
        rows = fetch_missing(style, args.limit)
        print(f"style={style}: {len(rows)} rows need backfill")
        print(f"style={style}: updated {update(style, rows, args.batch_size)} rows")


if __name__ == "__main__":
    main()
