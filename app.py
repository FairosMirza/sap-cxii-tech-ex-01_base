"""Product similarity microservice (Part 2).

GET /find_similar_products?product_id=...&num_similar=...   -> ["uniq_id", ...]
GET /products/{product_id}/similar                          -> results with score breakdown
GET /health (liveness), GET /ready (readiness)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from similarity.engine import ProductNotFoundError, SimilarityEngine, validate_weights
from similarity.features import FeatureStore, index_dir

INDEX_DIR = index_dir()
SEARCH_METHOD = os.getenv("SEARCH_METHOD", "ann")
MAX_NUM_SIMILAR = int(os.getenv("MAX_NUM_SIMILAR", "100"))

log = logging.getLogger("uvicorn.error")  # configured by uvicorn, so startup info reaches the container log

Method = Literal["exact", "ann"]
_ID_PATTERN = r"^[A-Za-z0-9_-]+$"
_ID_DOC = "Product uniq_id, e.g. 26d41bdc1495de290bc8e6062d927729"
_NUM_DOC = f"Number of similar products to return (1-{MAX_NUM_SIMILAR})"
_WEIGHTS_DOC = "Optional weights override, e.g. 'text:0.5,image:0.3,brand:0.2' (unlisted components get 0)"


def create_engine(index_dir: Path = INDEX_DIR, method: str = SEARCH_METHOD) -> SimilarityEngine:
    if method not in ("exact", "ann"):
        raise ValueError(f"SEARCH_METHOD must be 'exact' or 'ann', got {method!r}")
    start = time.perf_counter()
    store = FeatureStore.load(index_dir)
    engine = SimilarityEngine(store)
    if method == "ann":
        from similarity.ann import HnswIndex

        engine.ann = HnswIndex(store, engine.weights, index_dir=index_dir)
    log.info("loaded %d products (%s) in %.1fs", len(store), store.meta, time.perf_counter() - start)
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "engine", None) is None:  # tests inject their own engine
        app.state.engine = create_engine()
    yield


app = FastAPI(title="Product Similarity Service", version="1.0.0", lifespan=lifespan)


class ProductInfo(BaseModel):
    uniq_id: str
    name: str
    brand: str | None
    category: str | None
    segment: str | None
    colors: list[str]
    sales_price: float | None
    rating: float | None
    image_url: str | None


class SimilarProduct(ProductInfo):
    score: float
    contributions: dict[str, float]


class SimilarProductsResponse(BaseModel):
    product: ProductInfo
    method: Method
    weights: dict[str, float]
    results: list[SimilarProduct]


def get_engine(request: Request) -> SimilarityEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="similarity index is not loaded yet")
    return engine


def parse_weights(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    try:
        weights = {}
        for item in raw.split(","):
            name, value = item.split(":", 1)
            weights[name.strip()] = float(value)
        return validate_weights(weights)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid weights: {exc}") from None


def run_search(request: Request, product_id: str, num_similar: int, weights: str | None, method: Method | None):
    engine = get_engine(request)
    parsed = parse_weights(weights)
    try:
        rows, scores = engine.search(product_id, num_similar, parsed, method or SEARCH_METHOD)
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail=f"product '{product_id}' not found") from None
    return engine, parsed, rows, scores


@app.get("/find_similar_products", response_model=list[str])
def get_similar_products(
    request: Request,
    product_id: Annotated[str, Query(min_length=1, max_length=64, pattern=_ID_PATTERN, description=_ID_DOC)],
    num_similar: Annotated[int, Query(ge=1, le=MAX_NUM_SIMILAR, description=_NUM_DOC)],
    weights: Annotated[str | None, Query(max_length=300, description=_WEIGHTS_DOC)] = None,
    method: Annotated[Method | None, Query(description="Search method; defaults to SEARCH_METHOD")] = None,
) -> list[str]:
    engine, _, rows, _ = run_search(request, product_id, num_similar, weights, method)
    return engine.store.ids[rows].tolist()


@app.get("/products/{product_id}/similar", response_model=SimilarProductsResponse)
def get_similar_products_explained(
    request: Request,
    product_id: Annotated[str, PathParam(min_length=1, max_length=64, pattern=_ID_PATTERN, description=_ID_DOC)],
    num_similar: Annotated[int, Query(ge=1, le=MAX_NUM_SIMILAR, description=_NUM_DOC)] = 10,
    weights: Annotated[str | None, Query(max_length=300, description=_WEIGHTS_DOC)] = None,
    method: Annotated[Method | None, Query(description="Search method; defaults to SEARCH_METHOD")] = None,
) -> SimilarProductsResponse:
    engine, parsed, rows, scores = run_search(request, product_id, num_similar, weights, method)
    store, q = engine.store, engine.row(product_id)

    def info(row: int) -> dict:
        return {"uniq_id": str(store.ids[row]), **store.catalog[row]}

    return SimilarProductsResponse(
        product=ProductInfo(**info(q)),
        method=method or SEARCH_METHOD,
        weights={k: round(v, 4) for k, v in engine.query_weights(q, parsed or engine.weights).items()},
        results=[
            SimilarProduct(**info(row), score=float(score), contributions=parts)
            for row, score, parts in zip(rows.tolist(), scores.tolist(), engine.explain(q, rows, parsed))
        ],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict:
    engine = get_engine(request)
    return {**engine.store.meta, "status": "ready", "products": len(engine.store), "ann": engine.ann is not None}


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
