from __future__ import annotations

import io
import uuid
from pathlib import Path

from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

from .config import settings


def open_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def save_upload(image_bytes: bytes, original_name: str | None) -> Path:
    suffix = Path(original_name or "photo.jpg").suffix.lower() or ".jpg"
    filename = f"{uuid.uuid4().hex}{suffix}"
    path = settings.upload_dir / filename
    path.write_bytes(image_bytes)
    return path


def shrink_for_caption(image: Image.Image) -> bytes:
    img = image.copy().convert("RGB")
    img.thumbnail((settings.qwen_max_side, settings.qwen_max_side), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()
