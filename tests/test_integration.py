"""End-to-end checks on the real artifacts (real data + FAISS in one process)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import INDEX_DIR, app, create_engine
from similarity.engine import find_similar_products

pytestmark = pytest.mark.skipif(not (INDEX_DIR / "features.npz").exists(),
                                reason="run `python -m similarity.build` first")


@pytest.fixture(scope="module")
def real_engine():
    return create_engine(INDEX_DIR, method="ann")


def test_part1_function():
    result = find_similar_products("26d41bdc1495de290bc8e6062d927729", 5)
    assert len(result) == 5 and all(isinstance(pid, str) for pid in result)


def test_ann_recall_against_exact(real_engine):
    ids = np.random.default_rng(7).choice(real_engine.store.ids, 200, replace=False).tolist()
    recall = np.mean([
        len(set(real_engine.find_similar(pid, 10, method="ann")) & set(real_engine.find_similar(pid, 10)))
        / 10 for pid in ids
    ])
    assert recall >= 0.98


def test_results_are_sensible(real_engine):
    store = real_engine.store
    q = real_engine.row("80e47075f39a4bdf6b83f2d7690c95fe")  # "Brown Faux Leather Jacket for Man"
    rows, _ = real_engine.search(store.ids[q], 10, method="ann")
    assert np.mean([store.catalog[r]["category"] == "MensJackets" for r in rows]) >= 0.8
    assert store.group[q] not in set(store.group[rows].tolist())


def test_api_serves_real_index(real_engine):
    app.state.engine = real_engine
    with TestClient(app) as client:
        response = client.get("/find_similar_products",
                              params={"product_id": "26d41bdc1495de290bc8e6062d927729", "num_similar": 5})
    app.state.engine = None
    assert response.status_code == 200 and len(response.json()) == 5
