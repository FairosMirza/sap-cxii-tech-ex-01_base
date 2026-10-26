"""Feature store: every product as a set of per-attribute (component) vectors.

The similarity of two products on a component is the inner product of their
component vectors, so the overall score is a weighted sum of inner products
(``engine.py``) and the dense part can be packed into one vector index (``ann.py``):

  text, image     centred + L2-normalised embeddings           -> cosine similarity
  color           L2-normalised multi-hot over colour families  -> overlap (Ochiai)
  segment         rows of a PSD similarity matrix                -> 1 same, 0.5 related, 0 other
  category, brand integer codes (one-hot)                        -> 1 if equal else 0
  numeric         angle encoding theta = pi * t, t in [0, 1]     -> cos(pi * |t_a - t_b|)

Missing values: a candidate with an unknown component gets the component's mean
vector, so it scores the *expected* similarity (neutral) instead of being punished
or rewarded. A query with an unknown component drops it and renormalises the weights.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import COLOR_FAMILIES, Product

COMPONENTS = ("text", "image", "category", "brand", "segment", "color", "sales_price", "price", "weight", "rating")
NUMERIC = ("sales_price", "price", "weight", "rating")
SEGMENTS = ("women", "men", "unisex", "girls", "boys", "kids", "baby")
COLORS = tuple(COLOR_FAMILIES)
DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "index"
_RELATED_SEGMENTS = {("unisex", "women"), ("unisex", "men"), ("kids", "girls"), ("kids", "boys"), ("kids", "baby")}


def segment_vectors() -> np.ndarray:
    """Rows whose inner products are 1 (same), 0.5 (related) or 0 (unrelated) segment."""
    sim = np.eye(len(SEGMENTS))
    for a, b in _RELATED_SEGMENTS:
        i, j = SEGMENTS.index(a), SEGMENTS.index(b)
        sim[i, j] = sim[j, i] = 0.5
    values, vectors = np.linalg.eigh(sim)  # sim is PSD, so sim = V V^T with V = U sqrt(L)
    return (vectors * np.sqrt(np.clip(values, 0, None))).astype(np.float32)


def unit_rows(x: np.ndarray, center: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if center:
        x = x - x.mean(axis=0)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def angle_encode(values: Sequence[float | None], log: bool) -> np.ndarray:
    """Map values to unit vectors on a half circle: similarity = cos(pi * |t_a - t_b|).

    t is the value scaled to [0, 1] between the 1st and 99th percentile (log scale for
    prices and weights, so 100 vs 200 INR is as far apart as 1000 vs 2000 INR).
    """
    x = np.array([v if v is not None else np.nan for v in values], dtype=np.float64)
    if log:
        x = np.log(x)
    if np.isnan(x).all():
        return np.full((len(x), 2), np.nan, dtype=np.float32)
    lo, hi = np.nanpercentile(x, [1, 99])
    theta = math.pi * np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)


def _codes(values: Sequence[str | None]) -> np.ndarray:
    lookup = {v: i for i, v in enumerate(sorted({v for v in values if v}))}
    return np.array([lookup[v] if v else -1 for v in values], dtype=np.int32)


def _impute_mean(vectors: np.ndarray, known: np.ndarray) -> np.ndarray:
    vectors = np.nan_to_num(vectors.astype(np.float32))
    vectors[~known] = vectors[known].mean(axis=0) if known.any() else 0
    return vectors


@dataclass
class FeatureStore:
    ids: np.ndarray  # (N,) uniq_id
    group: np.ndarray  # (N,) ASIN code: listings of the same ASIN are the same product
    category: np.ndarray  # (N,) int code, -1 unknown
    brand: np.ndarray  # (N,) int code, -1 unknown
    text: np.ndarray  # (N, Dt)
    image: np.ndarray  # (N, Di)
    segment: np.ndarray  # (N, len(SEGMENTS))
    color: np.ndarray  # (N, len(COLORS))
    numeric: np.ndarray  # (N, len(NUMERIC), 2)
    known: np.ndarray  # (N, len(COMPONENTS)) bool
    rating: np.ndarray  # tie-breakers
    reviews: np.ndarray
    sales_price: np.ndarray
    catalog: list[dict] = field(default_factory=list)  # display fields for the details endpoint
    meta: dict = field(default_factory=dict)

    ARRAYS = ("ids", "group", "category", "brand", "text", "image", "segment", "color", "numeric", "known",
              "rating", "reviews", "sales_price")

    def __len__(self) -> int:
        return len(self.ids)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self, name) for name in self.ARRAYS}
        arrays["text"], arrays["image"] = arrays["text"].astype(np.float16), arrays["image"].astype(np.float16)
        np.savez(directory / "features.npz", **arrays)
        (directory / "catalog.json").write_text(json.dumps({"meta": self.meta, "products": self.catalog}))

    @classmethod
    def load(cls, directory: Path) -> FeatureStore:
        with np.load(directory / "features.npz", allow_pickle=False) as data:
            arrays = {name: data[name] for name in cls.ARRAYS}
        arrays["text"], arrays["image"] = arrays["text"].astype(np.float32), arrays["image"].astype(np.float32)
        info = json.loads((directory / "catalog.json").read_text())
        return cls(**arrays, catalog=info["products"], meta=info["meta"])


def build_store(
    products: Sequence[Product],
    text: np.ndarray,
    image: np.ndarray | None = None,
    image_known: np.ndarray | None = None,
    meta: dict | None = None,
) -> FeatureStore:
    n = len(products)
    known = {name: np.ones(n, dtype=bool) for name in COMPONENTS}

    category, brand = _codes([p.category for p in products]), _codes([p.brand for p in products])
    known["category"], known["brand"] = category >= 0, brand >= 0

    if image is None:
        image, image_known = np.zeros((n, 1), dtype=np.float32), np.zeros(n, dtype=bool)
    known["image"] = np.asarray(image_known, dtype=bool)
    image_vectors = np.zeros((n, image.shape[1]), dtype=np.float32)
    if known["image"].any():
        image_vectors[known["image"]] = unit_rows(image[known["image"]])

    seg_table = segment_vectors()
    segment = np.zeros((n, len(SEGMENTS)), dtype=np.float32)
    color = np.zeros((n, len(COLORS)), dtype=np.float32)
    for i, p in enumerate(products):
        if p.segment:
            segment[i] = seg_table[SEGMENTS.index(p.segment)]
        color[i, [COLORS.index(c) for c in p.colors]] = 1
    known["segment"] = np.array([p.segment is not None for p in products])
    known["color"] = color.any(axis=1)
    color /= np.maximum(np.linalg.norm(color, axis=1, keepdims=True), 1e-12)

    raw_numeric = {
        "sales_price": ([p.sales_price for p in products], True),
        "price": ([p.list_price for p in products], True),
        "weight": ([p.weight_g for p in products], True),
        "rating": ([p.rating for p in products], False),
    }
    numeric = np.zeros((n, len(NUMERIC), 2), dtype=np.float32)
    for j, name in enumerate(NUMERIC):
        values, log = raw_numeric[name]
        known[name] = np.array([v is not None for v in values])
        numeric[:, j] = _impute_mean(angle_encode(values, log), known[name])

    return FeatureStore(
        ids=np.array([p.uniq_id for p in products]),
        group=_codes([p.asin for p in products]),
        category=category,
        brand=brand,
        text=unit_rows(text),
        image=_impute_mean(image_vectors, known["image"]),
        segment=_impute_mean(segment, known["segment"]),
        color=_impute_mean(color, known["color"]),
        numeric=numeric,
        known=np.stack([known[name] for name in COMPONENTS], axis=1),
        rating=np.array([p.rating or 0 for p in products], dtype=np.float32),
        reviews=np.array([p.review_count or 0 for p in products], dtype=np.float32),
        sales_price=np.array([p.sales_price or np.inf for p in products], dtype=np.float32),
        catalog=[
            {"name": p.name, "brand": p.brand, "category": p.category, "segment": p.segment,
             "colors": list(p.colors), "sales_price": p.sales_price, "rating": p.rating,
             "image_url": p.image_urls[0] if p.image_urls else None}
            for p in products
        ],
        meta=meta or {},
    )
