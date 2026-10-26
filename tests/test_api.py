import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client(engine):
    app.state.engine = engine  # the lifespan only loads the real index when no engine is set
    with TestClient(app) as test_client:
        yield test_client
    app.state.engine = None


def test_find_similar_products(client):
    response = client.get("/find_similar_products", params={"product_id": "kurta-blue", "num_similar": 2})
    assert response.status_code == 200
    assert response.json()[0] == "kurta-blue-set" and len(response.json()) == 2


def test_weights_and_method_parameters(client):
    params = {"product_id": "tshirt", "num_similar": 1}
    assert client.get("/find_similar_products", params={**params, "weights": "image:1"}).json() == ["kurta-red"]
    assert client.get("/find_similar_products", params={**params, "method": "ann"}).status_code == 200


def test_unknown_product_is_404(client):
    response = client.get("/find_similar_products", params={"product_id": "nope", "num_similar": 3})
    assert response.status_code == 404


@pytest.mark.parametrize("params", [
    {"product_id": "kurta-blue", "num_similar": 0},
    {"product_id": "kurta-blue", "num_similar": 101},
    {"product_id": "kurta-blue", "num_similar": "ten"},
    {"num_similar": 3},
    {"product_id": "../etc/passwd", "num_similar": 3},
    {"product_id": "x" * 65, "num_similar": 3},
    {"product_id": "kurta-blue", "num_similar": 3, "weights": "text:abc"},
    {"product_id": "kurta-blue", "num_similar": 3, "weights": "colour:1"},
    {"product_id": "kurta-blue", "num_similar": 3, "weights": "text:-1"},
    {"product_id": "kurta-blue", "num_similar": 3, "method": "magic"},
])
def test_invalid_requests_are_422(client, params):
    assert client.get("/find_similar_products", params=params).status_code == 422


def test_explained_endpoint(client):
    body = client.get("/products/dress/similar", params={"num_similar": 3}).json()
    assert body["product"]["uniq_id"] == "dress"
    assert "image" not in body["weights"] and sum(body["weights"].values()) == pytest.approx(1, abs=1e-3)
    assert len(body["results"]) == 3
    assert all(sum(r["contributions"].values()) == pytest.approx(r["score"], abs=1e-3) for r in body["results"])


def test_health_and_readiness(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready").json()
    assert ready["status"] == "ready" and ready["products"] == 9


def test_not_ready_is_503(client):
    app.state.engine = None
    assert client.get("/ready").status_code == 503
    assert client.get("/find_similar_products", params={"product_id": "polo", "num_similar": 1}).status_code == 503


def test_unexpected_errors_are_500_without_details(engine, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(engine, "search", boom)
    app.state.engine = engine
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.get("/find_similar_products", params={"product_id": "polo", "num_similar": 1})
    app.state.engine = None
    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
