from __future__ import annotations

import httpx
from fastapi import HTTPException
from PIL import Image

from .config import settings
from .image_io import shrink_for_caption


def require_caption_service():
    if not settings.caption_service_url:
        raise HTTPException(
            status_code=400,
            detail="CAPTION_SERVICE_URL is not configured. Set captions manually or configure a caption service.",
        )


async def generate_all_captions(image: Image.Image) -> dict[str, str]:
    """Call an external caption service compatible with /caption/all.

    Expected response shape:
        {"A": {"caption": "..."}, "B": {"caption": "..."}}
    or:
        {"A": "...", "B": "..."}
    """
    require_caption_service()
    image_bytes = shrink_for_caption(image)
    headers = {}
    if settings.caption_service_token:
        headers["Authorization"] = f"Bearer {settings.caption_service_token}"

    async with httpx.AsyncClient(timeout=settings.caption_timeout_sec) as client:
        resp = await client.post(
            f"{settings.caption_service_url}/caption/all",
            files={"file": ("photo.jpg", image_bytes, "image/jpeg")},
            headers=headers,
        )
    resp.raise_for_status()
    payload = resp.json()

    captions = {}
    for style in ("A", "B"):
        value = payload.get(style)
        if isinstance(value, dict):
            value = value.get("caption")
        if value:
            captions[style] = str(value)
    if not captions:
        raise HTTPException(status_code=502, detail=f"Caption service returned no usable captions: {payload}")
    return captions
