from __future__ import annotations

import json
import time

import numpy as np

from api.config import settings
from api.db import db_cursor
from api.image_io import open_image, save_upload
from worker.models.encoders import get_bge_encoder, get_clip_encoder


STYLE_TO_EMBEDDING_COLUMN = {
    "A": "caption_a_embedding",
    "B": "caption_b_embedding",
}


def ensure_user(user_id: int) -> None:
    with db_cursor(commit=True) as cur:
        cur.execute("INSERT INTO users (user_id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))


def encode_caption_embeddings(captions: dict[str, str]) -> dict[str, list[float]]:
    items = [
        (style, text)
        for style, text in captions.items()
        if style in STYLE_TO_EMBEDDING_COLUMN and text and text.strip()
    ]
    if not items:
        return {}
    vecs = get_bge_encoder().encode([text for _, text in items])
    return {style: vec.astype(np.float32).tolist() for (style, _), vec in zip(items, vecs)}


def save_captions(cloud_id: int, captions: dict[str, str]) -> None:
    caption_embeddings = encode_caption_embeddings(captions)
    set_clauses = ["caption = COALESCE(caption, '{}'::jsonb) || %s::jsonb"]
    params = [json.dumps(captions, ensure_ascii=False)]
    for style, vec in caption_embeddings.items():
        set_clauses.append(f"{STYLE_TO_EMBEDDING_COLUMN[style]} = %s")
        params.append(vec)
    if caption_embeddings:
        set_clauses.append("caption_embedding_model = %s")
        params.append(settings.text_rerank_model_id)
    params.append(cloud_id)

    with db_cursor(commit=True) as cur:
        cur.execute(
            f"""
            UPDATE photos
            SET {", ".join(set_clauses)}
            WHERE cloud_id = %s
            """,
            params,
        )


def upload_photo_record(image_bytes: bytes, original_name: str | None, user_id: int, local_id: str) -> dict:
    ensure_user(user_id)
    image = open_image(image_bytes)
    saved_path = save_upload(image_bytes, original_name)
    image_vec = get_clip_encoder().encode_image(image).tolist()

    with db_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO photos (user_id, local_id, file_path, image_embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, local_id)
            DO UPDATE SET file_path = EXCLUDED.file_path,
                          image_embedding = EXCLUDED.image_embedding
            RETURNING cloud_id
            """,
            (user_id, local_id, str(saved_path), image_vec),
        )
        cloud_id = cur.fetchone()[0]

    return {
        "cloud_id": cloud_id,
        "user_id": user_id,
        "local_id": local_id,
        "file_path": str(saved_path),
        "image_embedding_dim": len(image_vec),
        "image": image,
    }


def get_photo_file_path(cloud_id: int) -> str | None:
    with db_cursor() as cur:
        cur.execute("SELECT file_path FROM photos WHERE cloud_id = %s", (cloud_id,))
        row = cur.fetchone()
    return row[0] if row else None


def search_photos(
    query: str,
    user_id: int,
    limit: int = 10,
    rerank: bool = True,
    rerank_k: int = 50,
    alpha: float = 0.7,
    caption_style: str = "A",
) -> dict:
    if caption_style not in STYLE_TO_EMBEDDING_COLUMN:
        raise ValueError("caption_style must be A or B")

    started = time.perf_counter()
    clip_query_vec = get_clip_encoder().encode_text(query).tolist()
    fetch_k = max(limit, rerank_k) if rerank else limit
    caption_col = STYLE_TO_EMBEDDING_COLUMN[caption_style]

    with db_cursor() as cur:
        cur.execute(
            f"""
            SELECT cloud_id, local_id, file_path, caption, {caption_col},
                   1 - (image_embedding <=> %s::vector) AS clip_sim
            FROM photos
            WHERE user_id = %s
            ORDER BY image_embedding <=> %s::vector
            LIMIT %s
            """,
            (clip_query_vec, user_id, clip_query_vec, fetch_k),
        )
        rows = cur.fetchall()

    candidates = []
    for row in rows:
        caption_data = row[3] if isinstance(row[3], dict) else {}
        candidates.append(
            {
                "cloud_id": row[0],
                "local_id": row[1],
                "file_path": row[2],
                "caption": caption_data.get(caption_style) if caption_data else None,
                "_caption_embedding": row[4],
                "similarity_score": round(float(row[5]), 4),
                "_clip_sim": float(row[5]),
            }
        )

    bge_ms = 0.0
    precomputed_count = 0
    fallback_texts = []
    fallback_positions = []
    if rerank and candidates:
        t0 = time.perf_counter()
        query_bge_vec = get_bge_encoder().encode_one(query)
        bge_ms += (time.perf_counter() - t0) * 1000.0

        caption_sims = np.zeros(len(candidates), dtype=np.float32)
        for i, item in enumerate(candidates):
            emb = item.get("_caption_embedding")
            if emb is not None:
                precomputed_count += 1
                caption_sims[i] = float(np.dot(query_bge_vec, np.asarray(emb, dtype=np.float32)))
            elif item.get("caption"):
                fallback_positions.append(i)
                fallback_texts.append(item["caption"])

        if fallback_texts:
            t0 = time.perf_counter()
            fallback_vecs = get_bge_encoder().encode(fallback_texts)
            bge_ms += (time.perf_counter() - t0) * 1000.0
            fallback_sims = fallback_vecs @ query_bge_vec
            for pos, sim in zip(fallback_positions, fallback_sims):
                caption_sims[pos] = float(sim)

        for i, item in enumerate(candidates):
            if item.get("_caption_embedding") is not None or item.get("caption"):
                item["caption_sim"] = round(float(caption_sims[i]), 4)
                item["rerank_score"] = round(
                    alpha * item["_clip_sim"] + (1.0 - alpha) * float(caption_sims[i]),
                    4,
                )
            else:
                item["caption_sim"] = None
                item["rerank_score"] = round(item["_clip_sim"], 4)
        candidates.sort(key=lambda x: -x["rerank_score"])

    results = candidates[:limit]
    for item in results:
        item.pop("_caption_embedding", None)
        item.pop("_clip_sim", None)

    return {
        "query": query,
        "user_id": user_id,
        "limit": limit,
        "rerank": {
            "enabled": rerank,
            "alpha": alpha,
            "caption_style": caption_style,
            "rerank_k": rerank_k,
            "precomputed_caption_embeddings": precomputed_count,
            "fallback_caption_texts_encoded": len(fallback_texts),
            "bge_elapsed_ms": round(bge_ms, 2),
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "results": results,
    }
