"""Benchmark exact search vs HNSW (Part 3).

1. Real catalogue (30k products, full scorer): latency and recall@10 of the
   HNSW + exact re-rank path against exact brute-force search, for several
   candidate-pool sizes.
2. Scale-up: synthetic catalogues of up to --scale vectors (128-d, sampled around
   the real product vectors) to show how brute force grows linearly with N while
   HNSW stays roughly logarithmic.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from similarity.ann import HnswIndex
from similarity.engine import SimilarityEngine
from similarity.features import FeatureStore

ROOT = Path(__file__).resolve().parents[1]


def percentiles(ms: list[float]) -> str:
    return f"p50 {np.percentile(ms, 50):6.2f} ms  p95 {np.percentile(ms, 95):6.2f} ms"


def timed(fn, items):
    out, ms = [], []
    for item in items:
        start = time.perf_counter()
        out.append(fn(item))
        ms.append(1000 * (time.perf_counter() - start))
    return out, ms


def real_catalogue(index_dir: Path, n_queries: int) -> None:
    store = FeatureStore.load(index_dir)
    engine = SimilarityEngine(store, cache_size=0)
    start = time.perf_counter()
    engine.ann = HnswIndex(store, engine.weights)
    build = time.perf_counter() - start
    graph_mb = engine.ann.index.ntotal * engine.ann.index.d * 4 / 2**20
    print(f"== real catalogue: {len(store)} products, fused vector {engine.ann.index.d}-d, "
          f"HNSW build {build:.1f}s, vectors {graph_mb:.0f} MB")

    queries = [str(i) for i in np.random.default_rng(1).choice(store.ids, n_queries, replace=False)]
    exact, ms = timed(lambda pid: set(engine.search(pid, 10, method="exact")[0].tolist()), queries)
    print(f"exact brute force            recall@10 1.0000  {percentiles(ms)}")
    for n in (20, 50, 100, 200, 400):
        engine.ann_candidates, engine.ann.ef_search = n, n
        approx, ms = timed(lambda pid: set(engine.search(pid, 10, method="ann")[0].tolist()), queries)
        recall = np.mean([len(a & e) / max(len(e), 1) for a, e in zip(approx, exact)])
        differ = sum(a != e for a, e in zip(approx, exact))
        print(f"hnsw, {n:3d} candidates re-ranked recall@10 {recall:.4f}  {percentiles(ms)}  "
              f"lists differing {differ}/{len(queries)}")


def scale_up(index_dir: Path, sizes: list[int], n_queries: int, dim: int = 128) -> None:
    import faiss

    store = FeatureStore.load(index_dir)
    base = np.hstack([store.text, store.image])
    centred = base - base.mean(axis=0)
    _, _, vt = np.linalg.svd(centred[np.random.default_rng(0).choice(len(base), 10_000, replace=False)],
                             full_matrices=False)
    real = centred @ vt[:dim].T
    real /= np.linalg.norm(real, axis=1, keepdims=True)
    rng = np.random.default_rng(2)
    print(f"\n== scale-up: {dim}-d vectors sampled around the real ones, single-thread search, top-10")
    for n in sizes:
        x = real[rng.integers(0, len(real), n)] + rng.normal(0, 0.35 / np.sqrt(dim), (n, dim))
        x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
        q = x[rng.choice(n, n_queries, replace=False)] + rng.normal(0, 0.05 / np.sqrt(dim), (n_queries, dim))
        q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)

        faiss.omp_set_num_threads(1)
        flat = faiss.IndexFlatIP(dim)
        flat.add(x)
        truth, ms_flat = timed(lambda v: flat.search(v[None], 10)[1][0], q)

        faiss.omp_set_num_threads(8)
        start = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 100
        hnsw.add(x)
        build = time.perf_counter() - start
        faiss.omp_set_num_threads(1)
        print(f"N={n:>9,}  brute force {percentiles(ms_flat)}   (HNSW build {build:.0f}s, 8 threads)")
        for ef in (32, 64, 128, 256):
            params = faiss.SearchParametersHNSW(efSearch=ef)
            found, ms = timed(lambda v: hnsw.search(v[None], 10, params=params)[1][0], q)
            recall = np.mean([len(set(f) & set(t)) / 10 for f, t in zip(found, truth)])
            print(f"             hnsw ef={ef:<4d} {percentiles(ms)}   recall@10 {recall:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "artifacts" / "index")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--scale", type=int, nargs="*", default=[100_000, 1_000_000])
    args = parser.parse_args()
    real_catalogue(args.index, args.queries)
    if args.scale:
        scale_up(args.index, args.scale, min(args.queries, 500))


if __name__ == "__main__":
    main()
