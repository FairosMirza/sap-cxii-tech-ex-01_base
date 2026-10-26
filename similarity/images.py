"""Offline image pipeline: download product images and embed them with ResNet-50 (ONNX).

Only the build step uses this module; the API serves precomputed vectors.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_RETRY_STATUS = {429, 500, 502, 503, 504}


def image_path(image_dir: Path, url: str) -> Path:
    return image_dir / f"{hashlib.sha1(url.encode()).hexdigest()[:20]}.jpg"


def _download_one(client: httpx.Client, url: str, dest: Path, retries: int = 4) -> bool:
    for attempt in range(retries):
        try:
            response = client.get(url)
            if response.status_code == 200 and response.headers.get("content-type", "").startswith("image/"):
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(response.content)
                tmp.replace(dest)  # atomic, so an interrupted run never leaves a half-written file
                return True
            if response.status_code not in _RETRY_STATUS:
                return False
        except httpx.HTTPError:
            pass
        time.sleep(0.5 * 2**attempt)
    return False


def download_images(candidates: Sequence[Sequence[str]], image_dir: Path, workers: int = 16) -> dict[str, int]:
    """Fetch one image per product, trying its candidate URLs in order. Cached files are reused."""
    image_dir.mkdir(parents=True, exist_ok=True)
    todo = [urls for urls in candidates if urls and first_cached(image_dir, urls) is None]
    stats = {"cached": sum(1 for urls in candidates if urls) - len(todo), "downloaded": 0, "failed": 0}
    headers = {"User-Agent": "product-similarity-assessment/1.0 (offline feature extraction)"}
    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)

    def fetch(urls: Sequence[str]) -> bool:
        return any(_download_one(client, url, image_path(image_dir, url)) for url in urls)

    with httpx.Client(headers=headers, timeout=20, limits=limits, follow_redirects=True) as client:
        with ThreadPoolExecutor(workers) as pool:
            for done, ok in enumerate(pool.map(fetch, todo), start=1):
                stats["downloaded" if ok else "failed"] += 1
                if done % 2000 == 0:
                    log.info("images: %d/%d (%d failed)", done, len(todo), stats["failed"])
    return stats


def first_cached(image_dir: Path, urls: Sequence[str]) -> Path | None:
    return next((path for url in urls if (path := image_path(image_dir, url)).exists()), None)


def load_image(path: Path, size: int = IMAGE_SIZE) -> np.ndarray | None:
    """Pad to a white square (keeps the whole garment), resize, ImageNet-normalise -> CHW float32."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if min(img.size) < 32:  # Amazon serves tiny placeholders for missing images
                return None
            side = max(img.size)
            canvas = Image.new("RGB", (side, side), (255, 255, 255))
            canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
            canvas = canvas.resize((size, size), Image.Resampling.BICUBIC)
    except (OSError, ValueError):
        return None
    array = (np.asarray(canvas, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return array.transpose(2, 0, 1)


class ImageEncoder:
    """ResNet-50 (ImageNet) penultimate-layer features, 2048-d, via onnxruntime."""

    def __init__(self, model_path: Path, threads: int | None = None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or os.cpu_count() or 1
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def encode(self, paths: Sequence[Path | None], batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Returns (embeddings[N, 2048], ok_mask[N]); rows for unreadable images are zero."""
        out = np.zeros((len(paths), 2048), dtype=np.float32)
        ok = np.zeros(len(paths), dtype=bool)
        with ThreadPoolExecutor(8) as pool:  # JPEG decoding overlaps with inference
            for start in range(0, len(paths), batch_size):
                chunk = paths[start : start + batch_size]
                arrays = list(pool.map(lambda p: load_image(p) if p and p.exists() else None, chunk))
                rows = [i for i, a in enumerate(arrays) if a is not None]
                if rows:
                    batch = np.stack([arrays[i] for i in rows])
                    out[[start + i for i in rows]] = self.session.run(None, {self.input_name: batch})[0]
                    ok[[start + i for i in rows]] = True
                if (start // batch_size) % 50 == 0:
                    log.info("image embeddings: %d/%d", start + len(chunk), len(paths))
        return out, ok
