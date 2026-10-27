"""Approximate nearest-neighbour candidate generation with HNSW (Part 3).

Each product is packed into one vector  x = concat_k( sqrt(w_k) * v_k )  using the
default weights w_k. A query uses  y = concat_k( w'_k / sqrt(w_k) * v_k(q) )  with its
own (renormalised) weights w'_k, so  <x, y> = sum_k w'_k * s_k  is exactly the
engine's score. Per-request weights work without rebuilding the index.

Brand and category are one-hot in the exact scorer; here they become random unit
codes (64-d), whose inner products are ~N(0, 1/64) for different values. That noise
only affects which candidates are fetched: the engine re-scores the candidates exactly.

Index: HNSW (Malkov & Yashunin, 2016, arXiv:1603.09320) from FAISS (Johnson et al., 2017,
arXiv:1702.08734; Douze et al., 2024, arXiv:2401.08281).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .features import NUMERIC, FeatureStore, index_dir

_DENSE = ("text", "image", "segment", "color")
_CODED = ("category", "brand")
INDEX_FILE, PARAMS_FILE = "hnsw.faiss", "hnsw.json"
log = logging.getLogger(__name__)


class HnswIndex:
    def __init__(self, store: FeatureStore, weights: Mapping[str, float], m: int = 32,
                 ef_construction: int = 200, ef_search: int = 100, code_dim: int = 64, seed: int = 0,
                 index_dir: Path | None = None):
        import faiss

        self.store = store
        self.ef_search = ef_search
        self.base = {name: w for name, w in weights.items() if w > 0}
        self.params = {"weights": self.base, "m": m, "ef_construction": ef_construction, "code_dim": code_dim,
                       "seed": seed, "products": len(store), "data": store.meta.get("fingerprint")}
        rng = np.random.default_rng(seed)
        self.codebooks = {}
        for name in _CODED:
            codes = rng.standard_normal((int(getattr(store, name).max()) + 2, code_dim)).astype(np.float32)
            codes /= np.linalg.norm(codes, axis=1, keepdims=True)
            codes[-1] = 0  # code -1 (unknown) -> zero vector -> neutral
            self.codebooks[name] = codes

        self.index = self._load(index_dir) if index_dir else None
        if self.index is None:
            vectors = np.hstack([np.sqrt(w) * block for w, block in self._blocks(slice(None))])
            self.index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        faiss.omp_set_num_threads(1)  # single queries: OpenMP only adds overhead

    def _load(self, directory: Path):
        import faiss

        params = directory / PARAMS_FILE
        if not (directory / INDEX_FILE).exists():
            return None
        if not params.exists() or json.loads(params.read_text()) != self.params:
            log.warning("saved HNSW index in %s was built with other settings; rebuilding", directory)
            return None
        return faiss.read_index(str(directory / INDEX_FILE))

    def save(self, directory: Path) -> None:
        import faiss

        faiss.write_index(self.index, str(directory / INDEX_FILE))
        (directory / PARAMS_FILE).write_text(json.dumps(self.params))

    def _blocks(self, rows):
        s = self.store
        for name, w in self.base.items():
            if name in _DENSE:
                block = getattr(s, name)[rows]
            elif name in _CODED:
                block = self.codebooks[name][getattr(s, name)[rows]]
            else:
                block = s.numeric[rows, NUMERIC.index(name)]
            yield w, np.atleast_2d(block)

    def query_vector(self, q: int, query_weights: Mapping[str, float]) -> np.ndarray:
        parts = [query_weights.get(name, 0.0) / np.sqrt(w) * block
                 for (w, block), name in zip(self._blocks([q]), self.base)]
        return np.hstack(parts).astype(np.float32)

    def search(self, q: int, query_weights: Mapping[str, float], n: int) -> np.ndarray:
        import faiss

        params = faiss.SearchParametersHNSW(efSearch=max(self.ef_search, n))
        _, rows = self.index.search(self.query_vector(q, query_weights), n, params=params)
        return rows[0][rows[0] >= 0]


def main() -> None:
    """Build the HNSW index once, next to the feature store, so API pods only load it."""
    from .engine import DEFAULT_WEIGHTS, validate_weights

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--index-dir", type=Path, default=index_dir())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = time.perf_counter()
    index = HnswIndex(FeatureStore.load(args.index_dir), validate_weights(DEFAULT_WEIGHTS))
    index.save(args.index_dir)
    log.info("HNSW index (%d x %d) -> %s (%.1fs)", index.index.ntotal, index.index.d, args.index_dir,
             time.perf_counter() - start)


if __name__ == "__main__":
    main()
