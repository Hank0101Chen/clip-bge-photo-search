from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "photo_db")
    db_user: str = os.getenv("DB_USER", "admin")
    db_password: str = os.getenv("DB_PASSWORD", "1234")

    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))

    clip_model_id: str = os.getenv("CLIP_MODEL_ID", "OFA-Sys/chinese-clip-vit-base-patch16")
    text_rerank_model_id: str = os.getenv("TEXT_RERANK_MODEL_ID", "BAAI/bge-m3")
    text_rerank_max_length: int = int(os.getenv("TEXT_RERANK_MAX_LENGTH", "256"))
    text_rerank_batch_size: int = int(os.getenv("TEXT_RERANK_BATCH_SIZE", "32"))

    caption_service_url: str = os.getenv("CAPTION_SERVICE_URL", "").rstrip("/")
    caption_service_token: str = os.getenv("CAPTION_SERVICE_TOKEN", "")
    caption_timeout_sec: float = float(os.getenv("CAPTION_TIMEOUT_SEC", "120"))
    qwen_max_side: int = int(os.getenv("QWEN_MAX_SIDE", "512"))


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
