from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.caption_client import generate_all_captions
from api.config import settings
from api.image_io import open_image
from worker.pipeline import get_photo_file_path, save_captions, search_photos as run_search, upload_photo_record


app = FastAPI(title="Photo Search BGE Rerank Service", version="0.1.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "clip_model": settings.clip_model_id,
        "text_reranker": settings.text_rerank_model_id,
        "caption_service_configured": bool(settings.caption_service_url),
    }


@app.post("/api/photos/upload")
async def upload_photo(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    local_id: str = Form(...),
    generate_caption: bool = Form(False),
):
    image_bytes = await file.read()
    record = upload_photo_record(image_bytes, file.filename, user_id, local_id)

    captions = None
    if generate_caption:
        captions = await generate_all_captions(record["image"])
        save_captions(record["cloud_id"], captions)

    record.pop("image", None)
    record["captions"] = captions
    return record


@app.post("/api/photos/{cloud_id}/captions")
async def set_captions(
    cloud_id: int,
    caption_a: Optional[str] = Form(None),
    caption_b: Optional[str] = Form(None),
):
    captions = {}
    if caption_a:
        captions["A"] = caption_a
    if caption_b:
        captions["B"] = caption_b
    if not captions:
        raise HTTPException(status_code=400, detail="Provide caption_a and/or caption_b.")

    save_captions(cloud_id, captions)
    return {
        "cloud_id": cloud_id,
        "captions_saved": sorted(captions.keys()),
        "caption_embedding_model": settings.text_rerank_model_id,
    }


@app.post("/api/photos/{cloud_id}/generate_caption")
async def generate_caption_for_photo(cloud_id: int):
    file_path = get_photo_file_path(cloud_id)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Photo {cloud_id} not found.")

    image = open_image(open(file_path, "rb").read())
    captions = await generate_all_captions(image)
    save_captions(cloud_id, captions)
    return {
        "cloud_id": cloud_id,
        "captions": captions,
        "caption_embedding_model": settings.text_rerank_model_id,
    }


@app.post("/api/photos/search")
async def search_photos(
    query: str = Form(...),
    user_id: int = Form(...),
    limit: int = Form(10),
    rerank: bool = Form(True),
    rerank_k: int = Form(50),
    alpha: float = Form(0.7),
    caption_style: str = Form("A"),
):
    try:
        payload = run_search(
            query=query,
            user_id=user_id,
            limit=limit,
            rerank=rerank,
            rerank_k=rerank_k,
            alpha=alpha,
            caption_style=caption_style,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(payload)
