from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, ChineseCLIPModel, ChineseCLIPProcessor

from api.config import settings


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ChineseClipEncoder:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = choose_device()
        try:
            self.model = ChineseCLIPModel.from_pretrained(model_id, local_files_only=True)
            self.processor = ChineseCLIPProcessor.from_pretrained(model_id, local_files_only=True)
        except OSError:
            self.model = ChineseCLIPModel.from_pretrained(model_id)
            self.processor = ChineseCLIPProcessor.from_pretrained(model_id)
        self.model = self.model.to(self.device).eval()

    def encode_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        # ChineseCLIPModel.get_text_features can be brittle in some versions.
        dummy = Image.new("RGB", (224, 224), (0, 0, 0))
        inputs = self.processor(
            text=[text],
            images=[dummy],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=52,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        feats = torch.nn.functional.normalize(outputs.text_embeds, p=2, dim=-1)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)


class BgeTextEncoder:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = choose_device()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_id, local_files_only=True)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id)
        self.model = self.model.to(self.device).eval()

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if not texts:
            hidden = int(getattr(self.model.config, "hidden_size", 0))
            return np.zeros((0, hidden), dtype=np.float32)

        bs = batch_size or settings.text_rerank_batch_size
        vecs = []
        for i in range(0, len(texts), bs):
            chunk = [text if text and text.strip() else " " for text in texts[i:i + bs]]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.text_rerank_max_length,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            feats = outputs.last_hidden_state[:, 0]
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            vecs.append(feats.cpu().numpy())
        return np.vstack(vecs).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text], batch_size=1)[0]


@lru_cache(maxsize=1)
def get_clip_encoder() -> ChineseClipEncoder:
    return ChineseClipEncoder(settings.clip_model_id)


@lru_cache(maxsize=1)
def get_bge_encoder() -> BgeTextEncoder:
    return BgeTextEncoder(settings.text_rerank_model_id)
