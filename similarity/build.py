"""Build the feature store the API serves: ``python -m similarity.build``.

Inputs: the raw catalogue (data/archive.zip or the extracted .ldjson) and the
precomputed embeddings in artifacts/embeddings/ (see scripts/embed.py).
Fallbacks: without MiniLM embeddings the text component uses TF-IDF + LSA
(needs scikit-learn); without image embeddings the image component is disabled.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from .data import load_products
from .features import DEFAULT_INDEX_DIR as INDEX_DIR, build_store

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT / "artifacts" / "embeddings"
log = logging.getLogger(__name__)


def load_aligned(path: Path, ids: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """Rows of an embedding file reordered to ``ids``; (vectors, known) or None if absent."""
    if not path.exists():
        return None
    with path.open("rb") as handle:
        if handle.read(40).startswith(b"version https://git-lfs"):
            raise RuntimeError(f"{path} is a Git LFS pointer, not the data: install git-lfs and run `git lfs pull`")
    with np.load(path, allow_pickle=False) as data:
        position = {pid: i for i, pid in enumerate(data["ids"].tolist())}
        rows = np.array([position.get(pid, -1) for pid in ids])
        vectors = data["vectors"].astype(np.float32)[np.maximum(rows, 0)]
        known = (rows >= 0) & (data["known"][np.maximum(rows, 0)] if "known" in data else True)
    return vectors, known


def build(data_path: Path | None = None, out_dir: Path = INDEX_DIR, embeddings: Path = EMBEDDINGS) -> None:
    start = time.perf_counter()
    products = load_products(data_path)
    ids = [p.uniq_id for p in products]
    meta = {"products": len(products)}

    text = load_aligned(embeddings / "text_minilm.npz", ids)
    if text is not None and text[1].all():
        text_vectors, meta["text_model"] = text[0], "all-MiniLM-L6-v2"
    else:
        log.warning("MiniLM embeddings missing or stale; falling back to TF-IDF + LSA")
        from .text import lsa_embeddings

        text_vectors, meta["text_model"] = lsa_embeddings([p.name for p in products]), "tfidf-lsa-256"

    image = load_aligned(embeddings / "image_resnet50.npz", ids)
    if image is None:
        log.warning("image embeddings missing; the image component is disabled")
    meta["image_model"] = "resnet50-pca256" if image is not None else None
    meta["image_coverage"] = round(float(image[1].mean()), 4) if image is not None else 0.0

    store = build_store(products, text_vectors, *(image or (None, None)), meta=meta)
    store.save(out_dir)
    log.info("feature store with %d products -> %s (%.1fs)", len(store), out_dir, time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=INDEX_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(args.data, args.out)


if __name__ == "__main__":
    main()
