"""Offline evaluation of the full scorer: which components earn their weight?

There are no relevance labels, so two proxies are reported for a fixed sample of
queries (exact search, top 10):

  cobuy@10     share of queries with >=1 result that customers also bought
               ("other_items_customers_buy", matched by name). Not used as a feature.
  category@10  share of results in the query's category (consistency; circular when
               the category component is switched on, so read it with care)
  segment@10   share of results for the same shopper (women/men/girls/...)
  brand@10     share of results from the same brand (diversity: lower = more varied)
  price_ratio  median |log2(price_result / price_query)|; 1.0 means 2x apart
"""

import argparse
import time
from pathlib import Path

import numpy as np

from similarity.data import load_products, norm_text
from similarity.engine import DEFAULT_WEIGHTS, SimilarityEngine
from similarity.features import FeatureStore

ROOT = Path(__file__).resolve().parents[1]
STRUCTURED = {k: v for k, v in DEFAULT_WEIGHTS.items() if k not in ("text", "image")}
CONFIGS = {
    "text only": {"text": 1.0},
    "image only": {"image": 1.0},
    "structured only": STRUCTURED,
    "text + image": {"text": 0.6, "image": 0.4},
    "default - image": {k: v for k, v in DEFAULT_WEIGHTS.items() if k != "image"},
    "default - text": {k: v for k, v in DEFAULT_WEIGHTS.items() if k != "text"},
    "default - brand": {k: v for k, v in DEFAULT_WEIGHTS.items() if k != "brand"},
    "default": DEFAULT_WEIGHTS,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--index", type=Path, default=ROOT / "artifacts" / "index")
    args = parser.parse_args()

    store = FeatureStore.load(args.index)
    products = {p.uniq_id: p for p in load_products()}
    ordered = [products[pid] for pid in store.ids.tolist()]
    names = [norm_text(p.name) for p in ordered]
    rows_by_name: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        rows_by_name.setdefault(n, []).append(i)

    def relevant(q: int) -> set[int]:
        rows = {r for n in ordered[q].co_bought for r in rows_by_name.get(n, ())}
        return {r for r in rows if store.group[r] != store.group[q]}

    rng = np.random.default_rng(0)
    queries = rng.choice(len(store), args.queries, replace=False)
    labelled = [q for q in queries if relevant(q)]
    price = np.array([p.sales_price or np.nan for p in ordered])

    print(f"{len(queries)} queries ({len(labelled)} with co-purchase labels)\n")
    print(f"{'config':18s} {'cobuy@10':>9s} {'category@10':>12s} {'segment@10':>11s} {'brand@10':>9s} "
          f"{'price_ratio':>12s} {'ms/query':>9s}")
    for name, weights in CONFIGS.items():
        engine = SimilarityEngine(store, weights)
        start = time.perf_counter()
        results = {q: engine.search(str(store.ids[q]), 10)[0] for q in queries}
        ms = 1000 * (time.perf_counter() - start) / len(queries)
        hit = np.mean([bool(relevant(q) & set(results[q].tolist())) for q in labelled])
        same = {
            attr: np.mean([np.mean(getattr(store, attr)[r] == getattr(store, attr)[q]) for q, r in results.items()
                           if getattr(store, attr)[q] >= 0])
            for attr in ("category", "brand")
        }
        seg = [p.segment for p in ordered]
        segment = np.mean([np.mean([seg[j] == seg[q] for j in r]) for q, r in results.items() if seg[q]])
        ratios = np.concatenate([np.abs(np.log2(price[r] / price[q])) for q, r in results.items()])
        print(f"{name:18s} {hit:9.3f} {same['category']:12.3f} {segment:11.3f} {same['brand']:9.3f} "
              f"{np.nanmedian(ratios):12.2f} {ms:9.1f}")


if __name__ == "__main__":
    main()
