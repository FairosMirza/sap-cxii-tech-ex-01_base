import numpy as np
import pytest

from similarity.data import Product
from similarity.engine import SimilarityEngine
from similarity.features import build_store


def make_product(uid, asin=None, name="item", brand=None, price=None, rating=4.0, reviews=10,
                 category=None, segment=None, colors=(), weight=None):
    return Product(uniq_id=uid, asin=asin or uid, name=name, brand=brand, sales_price=price, list_price=price,
                   weight_g=weight, rating=rating, review_count=reviews, category=category, browsenode=None,
                   segment=segment, colors=tuple(colors), image_urls=())


def build_test_store(rows, meta=None):
    """rows: (product, text vector, image vector or None)."""
    products = [p for p, _, _ in rows]
    text = np.stack([t for _, t, _ in rows])
    image = np.stack([i if i is not None else np.zeros_like(t) for _, t, i in rows])
    image_known = np.array([i is not None for _, _, i in rows])
    return build_store(products, text, image, image_known, meta=meta)


def catalogue_rows():
    """A tiny catalogue with known structure: kurtas, men's tops, a dress without image/price, twins."""
    rng = np.random.default_rng(0)
    topics = rng.standard_normal((4, 16))

    def vec(topic, noise=0.1):
        return topics[topic] + noise * rng.standard_normal(16)

    kurta_text, kurta_image = vec(0), vec(0)
    twin_text, twin_image = vec(3), vec(3)
    women = {"category": "WomensKurtas", "segment": "women"}
    men = {"category": "MensTops", "segment": "men"}
    return [
        (make_product("kurta-blue", "A1", "Blue cotton kurta", "biba", 500, colors=["blue"], **women), kurta_text, kurta_image),
        (make_product("kurta-blue-dup", "A1", "Blue cotton kurta", "biba", 500, colors=["blue"], **women), kurta_text, kurta_image),
        (make_product("kurta-blue-set", "B", "Blue kurta set", "biba", 550, 4.5, colors=["blue"], **women), vec(0, 0.15), vec(0, 0.15)),
        (make_product("kurta-red", "C", "Red cotton kurta", "w", 600, colors=["red"], **women), vec(0, 0.3), vec(1)),
        (make_product("tshirt", "D", "Men's t-shirt", "puma", 800, colors=["black"], **men), vec(1), vec(1)),
        (make_product("polo", "E", "Men's polo", "puma", 900, colors=["black"], **men), vec(1, 0.3), vec(1, 0.3)),
        (make_product("dress", "F", "Girls dress", None, None, category="GirlsDresses", segment="girls"), vec(2), None),
        (make_product("twin-a", "G", "Plain socks", "x", 100, rating=3.0), twin_text, twin_image),
        (make_product("twin-b", "H", "Plain socks", "x", 100, rating=4.5), twin_text, twin_image),
    ]


@pytest.fixture(scope="session")
def store():
    return build_test_store(catalogue_rows(), meta={"source": "test"})


@pytest.fixture(scope="session")
def reordered_store():
    """Same products and size as ``store``, rows in reverse order."""
    return build_test_store(catalogue_rows()[::-1], meta={"source": "test"})


@pytest.fixture(scope="session")
def duplicate_heavy_store():
    """The query's ASIN has 12 identical listings: more than the HNSW candidates fetched below."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(16)
    rows = [(make_product(f"dup-{i}", "Q", "Same tee"), base, base) for i in range(12)]
    rows += [(make_product(f"other-{i}", f"O{i}", "Other tee"), base + rng.standard_normal(16), base) for i in range(3)]
    return build_test_store(rows)


@pytest.fixture
def engine(store):
    return SimilarityEngine(store)
