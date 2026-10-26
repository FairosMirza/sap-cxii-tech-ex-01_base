"""Exact similarity scoring and top-k retrieval (Part 1).

score(q, c) = sum_k w_k * s_k(q, c) / sum_k w_k   over components k known for q

s_k is the per-component similarity from ``features.py``. Components the query lacks
are dropped and the weights renormalised; candidates lacking a component score the
neutral expected value. Ties (equal scores after rounding to 6 decimals) are broken by
rating, review count, lower price and finally uniq_id, so results are deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import numpy as np

from .features import COMPONENTS, DEFAULT_INDEX_DIR, NUMERIC, FeatureStore

DEFAULT_WEIGHTS: dict[str, float] = {
    "text": 0.30,
    "image": 0.20,
    "category": 0.15,
    "brand": 0.10,
    "segment": 0.05,
    "color": 0.05,
    "sales_price": 0.08,
    "price": 0.02,
    "weight": 0.02,
    "rating": 0.03,
}
_DENSE = ("text", "image", "segment", "color")
_CODED = ("category", "brand")


class ProductNotFoundError(KeyError):
    pass


@lru_cache(maxsize=1)
def default_engine() -> SimilarityEngine:
    return SimilarityEngine(FeatureStore.load(DEFAULT_INDEX_DIR))


def find_similar_products(product_id: str, num_similar: int) -> list[str]:
    """Part 1 entry point: uniq_ids of the ``num_similar`` products most similar to ``product_id``."""
    return default_engine().find_similar(product_id, num_similar)


def validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    unknown = set(weights) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown components {sorted(unknown)}; expected a subset of {list(COMPONENTS)}")
    if any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("weights must be finite and non-negative")
    if sum(weights.values()) <= 0:
        raise ValueError("at least one weight must be positive")
    return {name: float(weights.get(name, 0.0)) for name in COMPONENTS}


class SimilarityEngine:
    def __init__(self, store: FeatureStore, weights: Mapping[str, float] | None = None, ann=None,
                 cache_size: int = 4096, ann_candidates: int = 100):
        self.store = store
        self.weights = validate_weights(weights or DEFAULT_WEIGHTS)
        self.ann = ann
        self.ann_candidates = ann_candidates  # HNSW hits re-scored exactly per query
        self._row = {pid: i for i, pid in enumerate(store.ids.tolist())}
        self._cached = lru_cache(maxsize=cache_size)(self._search)

    # ------------------------------------------------------------------ public API

    def row(self, product_id: str) -> int:
        try:
            return self._row[product_id]
        except KeyError:
            raise ProductNotFoundError(product_id) from None

    def find_similar(self, product_id: str, num_similar: int, weights: Mapping[str, float] | None = None,
                     method: str = "exact") -> list[str]:
        rows, _ = self.search(product_id, num_similar, weights, method)
        return self.store.ids[rows].tolist()

    def search(self, product_id: str, num_similar: int, weights: Mapping[str, float] | None = None,
               method: str = "exact") -> tuple[np.ndarray, np.ndarray]:
        """Top ``num_similar`` (rows, scores). Results are cached (the catalogue is static)."""
        if num_similar < 1:
            raise ValueError("num_similar must be >= 1")
        if method not in ("exact", "ann"):
            raise ValueError("method must be 'exact' or 'ann'")
        key = tuple(sorted(validate_weights(weights).items())) if weights else None
        return self._cached(self.row(product_id), num_similar, key, method)

    def explain(self, q: int, rows: np.ndarray, weights: Mapping[str, float] | None = None) -> list[dict]:
        """Weighted contribution of every component to each result's score."""
        w = self.query_weights(q, validate_weights(weights) if weights else self.weights)
        parts = self.component_scores(q, rows, w)
        return [{name: round(float(w[name] * parts[name][i]), 4) for name in w} for i in range(len(rows))]

    # ------------------------------------------------------------------ scoring

    def query_weights(self, q: int, weights: Mapping[str, float]) -> dict[str, float]:
        known = self.store.known[q]
        active = {name: w for name, w in weights.items() if w > 0 and known[COMPONENTS.index(name)]}
        if not active:  # every weighted component is unknown for this product: fall back to text
            active = {"text": 1.0}
        total = sum(active.values())
        return {name: w / total for name, w in active.items()}

    def component_scores(self, q: int, rows, components) -> dict[str, np.ndarray]:
        s = self.store
        out = {}
        for name in components:
            if name in _DENSE:
                matrix = getattr(s, name)
                out[name] = matrix[rows] @ matrix[q]
            elif name in _CODED:
                codes = getattr(s, name)
                out[name] = (codes[rows] == codes[q]).astype(np.float32)
            else:
                j = NUMERIC.index(name)
                out[name] = s.numeric[rows, j] @ s.numeric[q, j]
        return out

    def scores(self, q: int, weights: Mapping[str, float], rows=None) -> np.ndarray:
        rows = slice(None) if rows is None else rows
        w = self.query_weights(q, weights)
        parts = self.component_scores(q, rows, w)
        return sum(w[name] * parts[name] for name in w)

    # ------------------------------------------------------------------ ranking

    def _search(self, q: int, k: int, weights_key, method: str) -> tuple[np.ndarray, np.ndarray]:
        weights = dict(weights_key) if weights_key else self.weights
        if method == "ann" and self.ann is not None:
            candidates = self.ann.search(q, self.query_weights(q, weights), max(self.ann_candidates, 4 * k))
            return self._rank(q, candidates, self.scores(q, weights, candidates), k)
        return self._rank(q, np.arange(len(self.store)), self.scores(q, weights), k)

    def _rank(self, q: int, rows: np.ndarray, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.store
        keep = s.group[rows] != s.group[q]  # the product itself and duplicate listings of it
        rows, scores = rows[keep], np.round(scores[keep], 6)
        m, pool = len(rows), 2 * k + 8
        while True:
            if pool < m:  # every row tied with the cut-off stays in, so tie-breaking is exact
                pick = np.flatnonzero(scores >= np.partition(scores, m - pool)[m - pool])
            else:
                pick = np.arange(m)
            r = rows[pick]
            order = np.lexsort((s.ids[r], s.sales_price[r], -s.reviews[r], -s.rating[r], -scores[pick]))
            out, seen = [], set()
            for i in pick[order]:
                if s.group[rows[i]] not in seen:  # one listing per ASIN
                    seen.add(s.group[rows[i]])
                    out.append(i)
                    if len(out) == k:
                        break
            if len(out) == k or pool >= m:
                result = rows[out], scores[out]
                for array in result:
                    array.setflags(write=False)  # shared through the LRU cache
                return result
            pool *= 2
