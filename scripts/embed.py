"""Offline step: embed product names (MiniLM) and images (ResNet-50), both with ONNX Runtime.

Writes artifacts/embeddings/{text_minilm,image_resnet50}.npz, aligned by uniq_id.
Run scripts.download_images first; products whose image could not be fetched are
marked unknown and handled by the fallback in the scorer.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from similarity.data import DATA_DIR, load_products
from similarity.images import ImageEncoder, first_cached
from similarity.text import MiniLMEncoder

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "embeddings"


def pca(x: np.ndarray, dim: int) -> tuple[np.ndarray, float]:
    """Project centred rows onto the top ``dim`` principal axes (thin SVD)."""
    centred = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    explained = float((s[:dim] ** 2).sum() / (s**2).sum())
    return centred @ vt[:dim].T, explained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--images", type=Path, default=DATA_DIR / "images")
    parser.add_argument("--image-dim", type=int, default=256)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    products = load_products()
    ids = np.array([p.uniq_id for p in products])

    start = time.perf_counter()
    text = MiniLMEncoder(args.models / "all-MiniLM-L6-v2").encode([p.name for p in products])
    np.savez(OUT_DIR / "text_minilm.npz", ids=ids, vectors=text.astype(np.float16))
    logging.info("text: %s in %.1fs", text.shape, time.perf_counter() - start)

    if args.skip_images:
        return
    start = time.perf_counter()
    paths = [first_cached(args.images, p.image_urls) for p in products]
    raw, known = ImageEncoder(args.models / "resnet50-onnx" / "model.onnx").encode(paths)
    raw[known] /= np.linalg.norm(raw[known], axis=1, keepdims=True)
    reduced = np.zeros((len(products), args.image_dim), dtype=np.float32)
    reduced[known], explained = pca(raw[known], args.image_dim)
    np.savez(OUT_DIR / "image_resnet50.npz", ids=ids, vectors=reduced.astype(np.float16), known=known)
    logging.info("images: %d/%d embedded, PCA-%d keeps %.1f%% variance, %.1fs",
                 known.sum(), len(products), args.image_dim, 100 * explained, time.perf_counter() - start)


if __name__ == "__main__":
    main()
