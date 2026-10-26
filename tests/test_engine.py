import numpy as np
import pytest

from similarity.ann import HnswIndex
from similarity.engine import ProductNotFoundError, SimilarityEngine, validate_weights
from similarity.features import COMPONENTS, FeatureStore


def test_excludes_query_and_duplicate_listings(engine):
    results = engine.find_similar("kurta-blue", 20)
    assert "kurta-blue" not in results and "kurta-blue-dup" not in results  # same ASIN
    assert len(results) == len(set(results)) == len(engine.store) - 2


def test_closest_product_ranks_first_and_scores_descend(engine):
    rows, scores = engine.search("kurta-blue", 5)
    assert engine.store.ids[rows[0]] == "kurta-blue-set"
    assert np.all(np.diff(scores) <= 0)


def test_unknown_product_raises(engine):
    with pytest.raises(ProductNotFoundError):
        engine.find_similar("does-not-exist", 3)


def test_weights_change_the_ranking(engine):
    # the red kurta's text is about kurtas but its photo looks like the t-shirt
    assert engine.find_similar("tshirt", 1, {"text": 1}) == ["polo"]
    assert engine.find_similar("tshirt", 1, {"image": 1}) == ["kurta-red"]


def test_query_side_fallback_drops_unknown_components(engine):
    q = engine.row("dress")  # no image, brand, price or weight
    weights = engine.query_weights(q, engine.weights)
    assert not {"image", "brand", "sales_price", "price", "weight"} & set(weights)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert engine.find_similar("dress", 3)  # still answers


def test_candidate_side_unknowns_get_the_mean_vector(store):
    image_known = store.known[:, COMPONENTS.index("image")]
    np.testing.assert_allclose(store.image[~image_known][0], store.image[image_known].mean(axis=0), atol=1e-6)


def test_ties_are_broken_by_rating(engine):
    results = engine.find_similar("kurta-blue", 20, {"text": 1})  # the twins have identical text
    assert results.index("twin-b") < results.index("twin-a")  # higher rating wins, not id order


def test_explanation_adds_up_to_the_score(engine):
    rows, scores = engine.search("tshirt", 4)
    parts = engine.explain(engine.row("tshirt"), rows)
    np.testing.assert_allclose([sum(p.values()) for p in parts], scores, atol=1e-3)


@pytest.mark.parametrize("weights", [{"colour": 1}, {"text": -1}, {"text": 0, "image": 0}, {"text": float("nan")}])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        validate_weights(weights)


def test_results_are_cached_and_read_only(engine):
    first = engine.search("polo", 3)
    assert engine.search("polo", 3) is first
    with pytest.raises(ValueError):
        first[0][0] = 0


def test_ann_matches_exact_search(store):
    engine = SimilarityEngine(store)
    engine.ann = HnswIndex(store, engine.weights)
    for pid in store.ids.tolist():
        for weights in (None, {"text": 0.5, "image": 0.5}):
            assert engine.find_similar(pid, 3, weights, "ann") == engine.find_similar(pid, 3, weights, "exact")


def test_saved_hnsw_index_is_reused_only_with_matching_settings(store, tmp_path):
    weights = SimilarityEngine(store).weights
    HnswIndex(store, weights).save(tmp_path)
    loaded = HnswIndex(store, weights, index_dir=tmp_path)
    assert loaded.index.ntotal == len(store)
    assert loaded.search(0, weights, 3).tolist() == HnswIndex(store, weights).search(0, weights, 3).tolist()

    other = {**weights, "brand": 0.5}  # stale index: different weights -> rebuilt, not reused
    rebuilt = HnswIndex(store, other, index_dir=tmp_path)
    assert rebuilt.params["weights"] == {k: v for k, v in other.items() if v > 0}


def test_store_round_trip(store, tmp_path):
    store.save(tmp_path)
    loaded = SimilarityEngine(FeatureStore.load(tmp_path))
    assert loaded.store.meta == {"source": "test"}
    for pid in store.ids.tolist():
        assert loaded.find_similar(pid, 4) == SimilarityEngine(store).find_similar(pid, 4)
