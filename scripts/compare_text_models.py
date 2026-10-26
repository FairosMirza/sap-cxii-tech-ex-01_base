"""Compare text representations of product names: TF-IDF vs LSA vs MiniLM (ONNX).

Neighbours are computed for a fixed random sample of products. The product itself,
other listings of the same ASIN and products with an identical name are excluded,
so the methods are judged only on non-trivial neighbours.

  category@10  share of neighbours in the same (most specific) category
  segment@10   share of neighbours for the same shopper (women/men/girls/...)
  cobuy@10     share of queries with >=1 neighbour customers also bought (weak label)
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from similarity.data import load_products, norm_text
from similarity.text import MiniLMEncoder, lsa_embeddings, tfidf_matrix

ROOT = Path(__file__).resolve().parents[1]


def unit(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def top_k(score_fn, queries, excluded, k=10, chunk=500):
    out = np.zeros((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        q = queries[s : s + chunk]
        scores = np.asarray(score_fn(q), dtype=np.float32)
        for row, qi in enumerate(q):
            scores[row, excluded[qi]] = -np.inf
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        out[s : s + chunk] = np.take_along_axis(part, order, axis=1)
    return out


def metrics(neighbours, queries, products, names):
    cat = [p.category for p in products]
    seg = [p.segment for p in products]
    rows = {"category@10": [], "segment@10": [], "cobuy@10": []}
    name_set = set(names)
    for qi, nb in zip(queries, neighbours):
        if cat[qi]:
            rows["category@10"].append(np.mean([cat[j] == cat[qi] for j in nb]))
        if seg[qi]:
            rows["segment@10"].append(np.mean([seg[j] == seg[qi] for j in nb]))
        relevant = {n for n in products[qi].co_bought if n in name_set and n != names[qi]}
        if relevant:
            rows["cobuy@10"].append(any(names[j] in relevant for j in nb))
    return {k: (float(np.mean(v)), len(v)) for k, v in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=3000)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "all-MiniLM-L6-v2")
    args = parser.parse_args()

    products = load_products()
    texts = [p.name for p in products]
    names = [norm_text(t) for t in texts]
    groups = defaultdict(list)
    for i, p in enumerate(products):
        groups["asin:" + p.asin].append(i)
        groups["name:" + names[i]].append(i)
    excluded = [np.array(sorted(set(groups["asin:" + p.asin] + groups["name:" + names[i]]))) for i, p in enumerate(products)]
    queries = np.random.default_rng(0).choice(len(products), args.queries, replace=False)

    timings, reps = {}, {}
    t = time.perf_counter(); tfidf = tfidf_matrix(texts); timings["tfidf"] = time.perf_counter() - t
    t = time.perf_counter(); reps["lsa-256"] = unit(lsa_embeddings(texts, 256)); timings["lsa-256"] = time.perf_counter() - t
    t = time.perf_counter(); reps["minilm"] = unit(MiniLMEncoder(args.model_dir).encode(texts)); timings["minilm"] = time.perf_counter() - t

    methods = {
        "tfidf": lambda q: (tfidf[q] @ tfidf.T).toarray(),
        "lsa-256": lambda q: reps["lsa-256"][q] @ reps["lsa-256"].T,
        "minilm": lambda q: reps["minilm"][q] @ reps["minilm"].T,
        "tfidf+minilm": lambda q: 0.5 * (tfidf[q] @ tfidf.T).toarray() + 0.5 * (reps["minilm"][q] @ reps["minilm"].T),
    }
    print(f"{'method':14s} {'embed s':>8s} {'category@10':>12s} {'segment@10':>11s} {'cobuy@10':>9s}")
    for name, fn in methods.items():
        m = metrics(top_k(fn, queries, excluded), queries, products, names)
        secs = timings.get(name, timings["tfidf"] + timings["minilm"])
        print(f"{name:14s} {secs:8.1f} {m['category@10'][0]:12.3f} {m['segment@10'][0]:11.3f} {m['cobuy@10'][0]:9.3f}")
    print("queries with labels:", {k: v[1] for k, v in m.items()})


if __name__ == "__main__":
    main()
