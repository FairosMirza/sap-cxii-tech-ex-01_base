"""Download the main image of every product into data/images/ (cached, safe to re-run)."""

import argparse
import logging
from pathlib import Path

from similarity.data import DATA_DIR, load_products
from similarity.images import download_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "images")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None, help="only the first N products (smoke test)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    candidates = [p.image_urls for p in load_products()[: args.limit]]
    stats = download_images(candidates, args.out, args.workers)
    logging.info("done: %s", stats)


if __name__ == "__main__":
    main()
