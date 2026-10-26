# Product Similarity Search: Solution Report

This report explains what was built for the [assignment](../ASSIGNMENT.md), how it works, the decisions behind it and the evidence that it works. The [README](../README.md) covers how to run it.

**Contents**
1. [Summary](#1-summary)
2. [Requirements checklist](#2-requirements-checklist)
3. [Pipeline workflow](#3-pipeline-workflow)
4. [The data and how it was cleaned](#4-the-data-and-how-it-was-cleaned)
5. [How similarity is computed](#5-how-similarity-is-computed)
6. [Design decisions](#6-design-decisions)
7. [Results](#7-results)
8. [Verification: tests, Docker, Kubernetes, CI](#8-verification-tests-docker-kubernetes-ci)
9. [Problems found and fixed](#9-problems-found-and-fixed)
10. [Limitations and next steps](#10-limitations-and-next-steps)
11. [References](#11-references)

Appendix: [Technical details](#appendix-technical-details)

---

## 1. Summary

**Task.** Given a product's `uniq_id`, return the `num_similar` most similar products from the 30,000-product Amazon fashion catalogue. Serve this as a FastAPI microservice that can run on Kubernetes. Bonus: make search scale with a vector index. Optional: combine text and image features.

**Approach.**
- Each product is described by **10 components**: name text, image, category, brand, shopper segment, colour, sale price, list price, weight and rating.
- The similarity of two products is a **weighted average** of their per-component similarities.
- Missing data is handled explicitly, so incomplete products are neither favoured nor penalised.
- Text and image embeddings are computed **offline** with ONNX Runtime: MiniLM for names, ResNet-50 for images.
- The API serves precomputed vectors. An **HNSW index** finds candidates and an exact scorer ranks them.

**Results.**

| | |
|---|---|
| Quality | 73.5% of queries get at least one co-purchased product in the top 10. 96% of results share the query's category and 99% its shopper segment. |
| Speed | 0.34 ms per query with HNSW, which returns the same top 10 as exact search (recall@10 = 1.000). On 1M vectors: 0.21 ms vs 7.9 ms for brute force. |
| Robustness | 73 automated tests pass. The service runs on Kubernetes as 2 replicas, and a rolling restart under continuous traffic produced **0 failed requests** out of 2,990. |
| Deployment | A multi-stage Docker image (458 MB) builds in about 1 minute. It runs as a non-root user on a read-only filesystem. CI builds it, deploys it to Kubernetes and tests it on every push. |

---

## 2. Requirements checklist

| Requirement from the assignment | Status | Where | Evidence |
|---|---|---|---|
| **Part 1:** `find_similar_products(product_id, num_similar) -> List[str]` | Done | [similarity/engine.py](../similarity/engine.py) | `test_part1_function` |
| Use brand, colour, price, image, sales_price, weight, rating (others optional) | Done: all 7, plus name text, category and segment | [similarity/features.py](../similarity/features.py) | [ablation](#72-does-each-component-help) |
| Retrieve attributes → compare with all products → similarity measure → sort → top N | Done | `SimilarityEngine.scores`, `_rank` | `test_engine.py` |
| Tie-breaker (e.g. sales_price or rating) | Done: rating, then reviews, then price, then id | `_rank` | `test_ties_are_broken_by_rating` |
| A better data loader, an alternative to pandas | Done: streaming, typed, reads the zip directly | [similarity/data.py](../similarity/data.py) | `test_data.py` |
| **Part 2:** FastAPI `GET /find_similar_products` | Done | [app.py](../app.py) | `test_api.py`, Kubernetes smoke test |
| Enhance error handling with HTTP codes | Done: 404, 422, 503, 500 | [app.py](../app.py) | `test_invalid_requests_are_422` and others |
| Dockerfile deployable on Kubernetes | Done: multi-stage, non-root, probes | [Dockerfile](../Dockerfile), [k8s/](../k8s/product-similarity.yaml) | [§8.2](#82-docker), [§8.3](#83-kubernetes) |
| Optional: deploy on a local cluster | Done: kind, Kubernetes v1.37 | `make kind-deploy` | [§8.3](#83-kubernetes) |
| **Part 3 (bonus):** ANN vector search with paper references and rationale | Done: HNSW (FAISS) plus exact re-ranking | [similarity/ann.py](../similarity/ann.py) | [§7.3](#73-search-speed-part-3) |
| Multimodal: text features (TF-IDF / embeddings / transformers) | Done: TF-IDF, LSA and MiniLM compared, MiniLM chosen | [similarity/text.py](../similarity/text.py) | [§7.1](#71-text-model-tf-idf-vs-lsa-vs-minilm) |
| Multimodal: image features from a pre-trained CNN | Done: ResNet-50 | [similarity/images.py](../similarity/images.py) | [§7.2](#72-does-each-component-help) |
| Combine text and image; configurable weights; fallback for missing features | Done | `weights=` parameter, [§5.3](#53-missing-data-the-fallback-mechanism) | `test_weights_change_the_ranking`, `test_query_side_fallback_*` |
| Suggested: ANN library, dimensionality reduction, caching | Done: FAISS HNSW, PCA 2048→256, LRU cache | | |
| Explain design decisions and trade-offs | Done | [§6](#6-design-decisions), [§10](#10-limitations-and-next-steps) | |

---

## 3. Pipeline workflow

### 3.1 End to end

The pipeline has three phases:
- **Offline:** heavy work done once per catalogue (downloading images, running neural networks).
- **Build:** turns the raw data plus embeddings into a compact feature store, inside the Docker build.
- **Serve:** loads the store, builds the vector index and answers requests.

```mermaid
flowchart TB
  subgraph OFF["1. Offline (once per catalogue, developer machine)"]
    direction LR
    RAW["data/archive.zip<br/>30k products, LDJSON"] --> LOAD["data.py<br/>parse and clean"]
    LOAD --> IMG["download_images.py<br/>224 px, 3 fallback URLs"]
    IMG --> RES["ResNet-50, ONNX<br/>2048-d, then PCA to 256-d"]
    LOAD --> TXT["MiniLM-L6, ONNX<br/>384-d name embeddings"]
    RES --> EMB[("artifacts/embeddings<br/>44 MB, Git LFS")]
    TXT --> EMB
  end
  subgraph BUILD["2. Build (Docker stage 1, a few seconds)"]
    direction LR
    FS["similarity.build<br/>clean data, align embeddings,<br/>encode all 10 components"] --> STORE[("feature store<br/>56 MB")]
  end
  subgraph SERVE["3. Serve (each API pod)"]
    direction LR
    START["startup: load store,<br/>build HNSW index (1.8 s)"] --> API["FastAPI<br/>GET /find_similar_products"]
  end
  RAW --> FS
  EMB --> FS
  STORE --> START
```

| Stage | Command | Input | Output | Time | How often |
|---|---|---|---|---|---|
| Models | `make models` | Hugging Face (pinned revisions) | `models/` (ONNX) | ~1 min | once |
| Images | `make images` | image URLs | `data/images/` (235 MB) | ~5 min | once per catalogue |
| Embeddings | `make embeddings` | names, images | `artifacts/embeddings/` (44 MB) | ~8 min on CPU | once per catalogue |
| Feature store | `make index`, or Docker stage 1 | zip + embeddings | `artifacts/index/` | 2–4 s | every build |
| Serve | `make serve`, or the container | feature store | in-memory HNSW index + API | ready in ~6 s | per pod |
| Test | `make test` | code + feature store | 73 tests | ~10 s | every change |
| Deploy | `make kind-deploy` | image | 2 replicas | ~20 s rollout | per release |
| Smoke | `make kind-smoke` | running service | API checks + zero-downtime check | ~65 s | per release |

### 3.2 Life of a request

```mermaid
sequenceDiagram
  autonumber
  participant C as Client
  participant A as FastAPI (app.py)
  participant E as SimilarityEngine
  participant H as HNSW index (FAISS)
  C->>A: GET /find_similar_products?product_id=...&num_similar=5
  A->>A: validate parameters (422 if invalid)
  A->>E: search(product_id, 5, weights, method)
  E->>E: look up product (404 if unknown), check LRU cache
  E->>H: query vector built with the query's weights
  H-->>E: 100 candidate products
  E->>E: exact score, drop same-ASIN listings, tie-break
  E-->>A: top 5 rows
  A-->>C: 200 ["uniq_id", ...]
```

### 3.3 CI/CD workflow (GitHub Actions)

The workflow is [.github/workflows/ci.yml](../.github/workflows/ci.yml). It runs on every push and pull request:

```mermaid
flowchart LR
  P["push / PR"] --> T
  subgraph T["job: test"]
    direction TB
    T1["checkout + LFS pull (cached)"] --> T2["Python 3.14, pip install"] --> T3["build feature store"] --> T4["pytest: 73 tests"]
  end
  T --> D
  subgraph D["job: deploy"]
    direction TB
    D1["docker build"] --> D2["run container read-only,<br/>no capabilities: smoke test"] --> D3["create kind cluster"] --> D4["kubectl apply, rollout"] --> D5["API checks + rolling restart<br/>under load: 0 errors allowed"]
  end
```

---

## 4. The data and how it was cleaned

Exploring the data turned up several problems. Each is handled in [similarity/data.py](../similarity/data.py):

| Field | What we found | How it is handled | Known for |
|---|---|---|---|
| `weight` | 79% of rows hold the placeholder `999999999`; units are mixed (g, Grams, Kilograms, Kg) | placeholder → unknown; converted to grams | 20.8% |
| `sales_price` | string in INR; missing for 10% | parsed to a number | 90.4% |
| list price ("price") | not in the data, but `discount_percentage` is | `sales_price / (1 − discount)` | 90.4% |
| `rating` | 25% are 5.0, often based on a single review | used as a weak signal; the review count breaks ties | 100% |
| `brand` | missing for 27%; case variants (`BIBA`/`Biba`); placeholders ("generic") | normalised; inferred from the start of the name if that brand appears on ≥ 2 products | 76.5% |
| colour | the `colour` field lists the colours of *other variants* | own colour taken from the URL slug (`…_Sea-Green_X-Large`) or the name; 15 colour families | 79.3% |
| category | missing for 15% | filled from the most common category of the product's browse node (95.8% agreement) | 99.6% |
| segment | not a field | from the category prefix (`Womens…`) or the name (women/men/girls/boys/kids/baby/unisex) | 98.8% |
| images | some links are dead (404) | falls back to the product's 2nd and 3rd image | 99.6% |
| `asin` | 451 products are listed 2–3 times under different `uniq_id`s | the query's duplicates are never returned; at most one result per ASIN | |

---

## 5. How similarity is computed

### 5.1 Components and default weights

| Component | What is compared | Similarity | Weight |
|---|---|---|---|
| text | product name, embedded by MiniLM | cosine | 0.30 |
| image | main photo, embedded by ResNet-50 (PCA 256) | cosine | 0.20 |
| category | most specific category | same = 1 | 0.15 |
| brand | normalised brand | same = 1 | 0.10 |
| segment | women / men / girls / boys / kids / baby / unisex | same 1, related 0.5 | 0.05 |
| colour | colour families | overlap | 0.05 |
| sales_price | log price | 1 if equal, ≈ 0.81 at 2× apart, −1 at the extremes | 0.08 |
| price (list) | log list price | same scale | 0.02 |
| weight | log grams | same scale | 0.02 |
| rating | rating | same scale | 0.03 |

The score is a weighted average over the components the query has:

$$\text{score}(q, c) = \frac{\sum_k w_k\, s_k(q, c)}{\sum_k w_k}$$

### 5.2 Worked example

Query: **"Brown Faux Leather Jacket for Man"** (Leather Retail, ₹1,669). This product has no known weight, so the weight component is dropped and the other weights are rescaled to sum to 1. The table shows each component's contribution to the final score, taken from `GET /products/{id}/similar`:

| # | Result | Score | text | image | category | brand | segment | colour | price (sale + list) | rating |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Leather Retail Black Faux Leather Jacket (₹1,499) | 0.820 | 0.228 | 0.118 | 0.153 | 0.102 | 0.051 | 0.036 | 0.101 | 0.030 |
| 2 | Leather Retail Black Plain Faux Leather Jacket for Man (₹1,599) | 0.811 | 0.262 | 0.116 | 0.153 | 0.102 | 0.051 | 0.000 | 0.097 | 0.030 |
| 3 | Leather Retail Army Colour Faux Leather Jacket for Man (₹1,699) | 0.795 | 0.248 | 0.113 | 0.153 | 0.102 | 0.051 | 0.000 | 0.097 | 0.030 |

Result 1 ranks first despite a slightly weaker name match. Its listing also offers brown, so it scores on colour, which the other two do not.

### 5.3 Missing data (the fallback mechanism)

- **The query lacks a component.** The component is dropped and the remaining weights are rescaled. For example, a product without a photo is matched on text and attributes only.
- **A candidate lacks a component.** It gets the component's *average* vector, so it scores the *expected* similarity. Missing data neither helps nor hurts it.
- **A whole embedding file is missing.** The build falls back to TF-IDF + LSA for text and turns the image component off.

### 5.4 Ranking rules

- The query product and other listings of the same ASIN are never returned.
- At most one result per ASIN.
- Ties (equal to 6 decimals) are broken by higher rating, then more reviews, then lower price, then `uniq_id`. Results are therefore deterministic.
- Results are cached in an LRU cache, which is safe because the catalogue is static.

---

## 6. Design decisions

| Decision | Alternatives considered | Why |
|---|---|---|
| Weighted per-component similarity | One big embedding; a learned model | Explainable (each component's contribution can be shown), every weight can be tuned, and no training labels are needed |
| MiniLM for text | TF-IDF, LSA | Best category and segment agreement in [§7.1](#71-text-model-tf-idf-vs-lsa-vs-minilm); computed offline, so it adds nothing at query time |
| ONNX Runtime instead of PyTorch | PyTorch / sentence-transformers | Much smaller install, and no clash between PyTorch's and FAISS's bundled OpenMP libraries |
| ResNet-50 for images | CLIP, EfficientNet | The pre-trained CNN the assignment suggests; CLIP is the obvious next step |
| Prices encoded as angles | Feature scaling + Euclidean distance | Keeps every component an inner product, so all components fit in one vector index |
| HNSW + exact re-ranking | Brute force, IVF-PQ, Annoy, ScaNN | Top recall/speed trade-off for in-memory data, incremental inserts, few settings; the exact re-rank keeps all ranking rules in one place |
| Precomputed embeddings in Git LFS | Compute in the Docker build; artifact store | Reproducible build without 30k image downloads or ML frameworks in the image |
| Serving image without ML libraries | One image with everything | Smaller attack surface and image; offline and online code are clearly separated |
| One uvicorn worker per pod | Several workers per pod | Kubernetes scales replicas; each pod stays simple and predictable |
| Python 3.14 | Python 3.10 (in the sample Dockerfile) | 3.10 reaches end of life in October 2026 |

---

## 7. Results

### 7.1 Text model: TF-IDF vs LSA vs MiniLM

Setup: 3,000 random products. Trivial neighbours (same product, same ASIN, identical name) are excluded.

| Method | category@10 | segment@10 | co-purchase hit@10 | Embedding time |
|---|---|---|---|---|
| TF-IDF (words + character n-grams) | 0.632 | 0.858 | 0.530 | 1.9 s |
| LSA (TF-IDF → SVD 256) | 0.672 | 0.851 | 0.419 | 7.9 s |
| **MiniLM-L6-v2 (chosen)** | **0.710** | **0.888** | 0.537 | 26.7 s (offline) |

### 7.2 Does each component help?

Setup: 2,000 random queries, top 10. There are no relevance labels, so these are proxies:
- **co-purchase hit@10:** share of queries where at least one result appears in the product's "customers also bought" list. This list is never used as a feature, so the metric doesn't reuse a scoring input.
- **brand@10:** share of results from the same brand. Lower means more variety.

| Weights | co-purchase hit@10 | category@10 | segment@10 | brand@10 |
|---|---|---|---|---|
| text only | 0.690 | 0.739 | 0.899 | 0.434 |
| image only | 0.299 | 0.574 | 0.824 | 0.086 |
| structured attributes only | 0.530 | 0.991 | 0.994 | 0.445 |
| default without text | 0.585 | 0.985 | 0.991 | 0.410 |
| default without image | 0.719 | 0.963 | 0.989 | 0.469 |
| **default (all components)** | **0.735** | 0.964 | 0.987 | 0.452 |

- Text matters most: removing it costs 15 points.
- Images help a little (+1.6 points).
- The structured attributes keep results in the right category, segment and price band.

### 7.3 Search speed (Part 3)

**Real catalogue** (30k products, single queries):

| Method | recall@10 vs exact | p50 latency |
|---|---|---|
| exact (brute force) | 1.000 | 1.13 ms |
| **HNSW, 100 candidates re-ranked (default)** | **1.000** | **0.34 ms** |

**Scale test** (synthetic 128-d vectors around the real ones, single thread):

| Catalogue size | brute force | HNSW (ef=128) | recall@10 |
|---|---|---|---|
| 100k | 0.74 ms | 0.12 ms | 0.999 |
| 1M | 7.90 ms | 0.21 ms | 1.000 |

Brute-force cost grows linearly with the catalogue size, while HNSW stays nearly flat.

**Why HNSW.** HNSW ([Malkov & Yashunin](https://arxiv.org/abs/1603.09320)) builds a layered "small world" graph and searches it greedily from coarse to fine layers.
- Search cost grows roughly logarithmically with the number of items.
- It is consistently at or near the top of [ANN-Benchmarks](https://arxiv.org/abs/1807.05614) for in-memory data.
- New items can be inserted without retraining.

The other options are better suited elsewhere:
- **IVF-PQ** compresses vectors; it is the better choice above about 100M items or when memory is tight.
- **Annoy** must be rebuilt to add items and has lower recall.
- **Brute force** is fine up to about 100k items.

---

## 8. Verification: tests, Docker, Kubernetes, CI

All checks were run on 2026-10-23: macOS arm64, Docker Desktop (4 CPUs, 6 GB), kind v0.33 with Kubernetes v1.37.

### 8.1 Automated tests: 73 passing (`make test`)

| Suite | What it proves |
|---|---|
| `test_data.py` (37) | Every parsing rule: prices, units, segments, colours, image URLs, zip loading, brand and category inference |
| `test_engine.py` (15) | Exclusion and deduplication, ranking, weights, fallbacks, tie-breaking, explanations, caching, save/load, HNSW = exact |
| `test_api.py` (17) | 200 / 404 / 422 / 503 / 500 responses, the weights and method parameters, the explanation endpoint |
| `test_integration.py` (4) | On the real data: the Part 1 function, HNSW recall ≥ 0.98, sensible neighbours, the API serving the real index |

### 8.2 Docker

| Check | Result |
|---|---|
| `docker build .` with the real Dockerfile (packages from PyPI) | Passed in 1 min 12 s; the feature store is built inside the image (3.9 s) |
| Image contents | 458 MB: Debian slim + Python ≈ 160 MB, Python packages 123 MB, feature store 56 MB, code < 0.1 MB |
| Run with `--read-only --cap-drop ALL` | Ready in ~6 s, ~270 MiB memory, runs as UID 10001 |
| Endpoints | 200 for a known product; 404 unknown product; 422 for `num_similar=0`, `num_similar=1000` and bad weights |
| Clone without Git LFS | The build stops with a clear message to run `git lfs pull` |

### 8.3 Kubernetes

| Check | Result |
|---|---|
| Manifest schema validation (strict, Kubernetes 1.37) | Passed for the Deployment and the Service |
| `kubectl apply` + rollout of 2 replicas | Both ready in 20 s |
| Requests through the Service | Correct ids; 404 / 422 as expected; `/ready` reports 30,000 products and both embedding models |
| Inside a pod | `uid=10001`, writing to `/app` fails (read-only filesystem), `/tmp` is writable, scikit-learn is absent |
| **Zero downtime:** continuous requests during `kubectl rollout restart` | Without `preStop`: 12 of 2,575 requests failed. With a 5 s `preStop` sleep: **0 of 2,990** failed (p50 1.4 ms, p99 19 ms) |
| `make kind-smoke`, the repeatable version used in CI | Functional checks OK; 2,453 / 2,453 requests succeeded during the rollout |

### 8.4 Continuous integration

Every push runs the [workflow](../.github/workflows/ci.yml) shown in [§3.3](#33-cicd-workflow-github-actions): tests, Docker build, container smoke test, kind deployment and the zero-downtime test. The status badge is in the [README](../README.md).

---

## 9. Problems found and fixed

| Problem | How it was found | Fix | Outcome |
|---|---|---|---|
| 79% of weights are a fake value (`999999999`) | data profiling | treated as unknown, with the fallback | no fake values in the scores |
| The `colour` field describes other variants | data profiling | own colour read from the URL and name | colour known for 79% |
| 451 products listed twice or three times | data profiling | duplicates excluded from results | no duplicate results |
| Some image links are dead | download run | try the product's other images | images for 99.6% |
| FAISS and scikit-learn each bundle their own OpenMP library | inspecting loaded libraries | scikit-learn kept out of the API | one OpenMP runtime in production |
| Rolling restarts dropped 0.47% of requests | in-cluster load test | `preStop` sleep of 5 s | 0 failed requests |
| A tie-break test passed for the wrong reason (ids happened to sort the same way) | test review | test data where id order and rating order disagree | the test now proves the rating tie-break |
| A clone without Git LFS gives placeholder files | review of the LFS change | the build detects them and explains the fix | clear error instead of a confusing crash |

---

## 10. Limitations and next steps

- **No ground-truth labels.** Weights were set by hand and checked with proxies. With click or purchase data, the next step is to learn the weights and validate with A/B tests.
- **Generic image model.** ResNet-50 (ImageNet) sees shape and colour but not style. CLIP or a fashion-tuned model should improve results and would allow text-to-image search.
- **Static catalogue.** Updates mean a rebuild and a rolling deploy. For frequent updates, use HNSW's incremental inserts or a vector database (e.g. SAP HANA Cloud Vector Engine, pgvector).
- **Scale.** The index is rebuilt at startup (1.8 s for 30k products). For millions of products, persist the index and use compressed vectors (HNSW-SQ or IVF-PQ).
- **Production hardening.** Add authentication, TLS and rate limiting at the ingress, an HPA and a PodDisruptionBudget, and metrics and tracing.
- **Variety.** 45% of results share the query's brand. That suits "more like this"; a discovery use case would lower the brand weight, which the `weights` parameter already supports.

---

## 11. References

1. Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, IEEE TPAMI 2020. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
2. Johnson, Douze & Jégou, *Billion-scale similarity search with GPUs*, IEEE Trans. Big Data 2019. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
3. Douze et al., *The Faiss library*, 2024. [arXiv:2401.08281](https://arxiv.org/abs/2401.08281)
4. Aumüller, Bernhardsson & Faithfull, *ANN-Benchmarks*, Information Systems 2020. [arXiv:1807.05614](https://arxiv.org/abs/1807.05614)
5. Jégou, Douze & Schmid, *Product quantization for nearest neighbor search*, IEEE TPAMI 2011.
6. Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization* (ScaNN), ICML 2020. [arXiv:1908.10396](https://arxiv.org/abs/1908.10396)
7. Wang et al., *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*, NeurIPS 2020. [arXiv:2002.10957](https://arxiv.org/abs/2002.10957)
8. Reimers & Gurevych, *Sentence-BERT*, EMNLP 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
9. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
10. Deerwester et al., *Indexing by latent semantic analysis*, JASIS 1990.

---

## Appendix: Technical details

### A.1 Encodings and why

- **Text and image:** the embeddings are centred (dataset mean subtracted) and L2-normalised, so cosine similarity is a plain inner product. Centring also makes the average similarity about 0, which is what the candidate-side fallback relies on.
- **Numbers as angles:** a value is scaled to t ∈ [0, 1] between the 1st and 99th percentile and mapped to the unit vector (cos πt, sin πt). Two values then have similarity cos(π·Δt): 1 when equal, ≈ 0.81 for prices 2× apart, −1 at opposite ends.
  - Prices and weights use a log scale, so ₹100 vs ₹200 is as far apart as ₹1,000 vs ₹2,000.
  - Clipping at the percentiles stops outliers from squeezing all other values together.
  - Because each comparison is an inner product, numbers fit in the same vector index as the embeddings.
- **Segment:** each segment is a row of a positive semi-definite similarity matrix (1 = same, 0.5 = related, 0 = unrelated), factorised with an eigendecomposition. The inner product of two rows reproduces the matrix entry.
- **Colour:** an L2-normalised multi-hot vector over 15 colour families. Its inner product is the Ochiai overlap of the two colour sets.
- **Missing values:** a candidate with an unknown component gets the component's mean vector, so its score is the expected similarity. A query with an unknown component drops it, and the other weights are rescaled.

### A.2 One vector for the index

Each product is packed into one vector using the default weights $w$. The query vector carries the query's own weights $w'$, which are renormalised and may come from the request:

$$x_c = \big[\sqrt{w_k}\, v_k(c)\big]_k, \qquad y_q = \Big[\tfrac{w'_k}{\sqrt{w_k}}\, v_k(q)\Big]_k \quad\Rightarrow\quad \langle x_c, y_q\rangle = \sum_k w'_k\, s_k(q, c)$$

- **Same score as exact search:** the index ranks by exactly the exact scorer's score, and per-request weights work without rebuilding the index.
- **Random codes for one-hot components:** brand and category are one-hot in the exact scorer (6.3k + 250 dimensions). In the index they become random 64-d unit codes, whose inner product is about 0 (standard deviation 0.125) for different values. That noise only affects which candidates are fetched, because the exact scorer re-scores them.
- **Index settings:** HNSW with `M=32`, `efConstruction=200` and `efSearch=100`, using inner product. 30k products × 798 dimensions are built in 1.8 s at startup (91 MB of vectors).

### A.3 Full benchmark (`make benchmark`)

Real catalogue, single queries:

| Method | recall@10 vs exact | p50 | p95 |
|---|---|---|---|
| exact brute force (numpy) | 1.000 | 1.13 ms | 1.33 ms |
| HNSW, 20 candidates re-ranked | 0.998 | 0.20 ms | 0.28 ms |
| HNSW, 50 candidates | 0.999 | 0.23 ms | 0.35 ms |
| **HNSW, 100 candidates (default)** | **1.000** | **0.34 ms** | 0.54 ms |
| HNSW, 200 candidates | 1.000 | 0.56 ms | 0.81 ms |

Scale-up: synthetic 128-d vectors sampled around the real product vectors, single-thread search:

| N | brute force p50 | HNSW ef=64: p50 / recall@10 | HNSW ef=128: p50 / recall@10 | HNSW build (8 threads) |
|---|---|---|---|---|
| 100k | 0.74 ms | 0.06 ms / 0.998 | 0.12 ms / 0.999 | 1 s |
| 1M | 7.90 ms | 0.14 ms / 0.998 | 0.21 ms / 1.000 | 18 s |

The synthetic data is clustered, which is easy for HNSW. Recall should be measured again on real data at that scale.

### A.4 Alternatives to HNSW, dimensionality reduction, caching

- **IVF-PQ** ([Jégou et al.](https://doi.org/10.1109/TPAMI.2010.57)) compresses each vector to 16–64 bytes. It is the right choice at 100M+ vectors or tight memory, but loses some recall and needs a training step.
- **[ScaNN](https://arxiv.org/abs/1908.10396)** is very fast for maximum inner product search but is heavier to integrate and operate.
- **Annoy** is simple and memory-mappable, but has lower recall at the same speed and must be rebuilt to add items.
- **Brute force** is exact and simplest. It is fine up to roughly 100k items: about 1 ms here.
- **[FAISS](https://arxiv.org/abs/2401.08281)** provides both HNSW and the exact baseline, plus an upgrade path (HNSW with 8-bit scalar quantisation, IVF-PQ, GPU).
- **Dimensionality reduction:** PCA on the image embeddings (2048 → 256 keeps 88.4% of the variance). t-SNE was not used: it is a 2-D visualisation method, can't project new points and doesn't preserve global distances.
- **Caching:** each pod keeps an LRU cache of results. Because the catalogue is static, the top-100 neighbours of every product could also be precomputed offline (30k × 100 IDs) and served from a key-value store.

### A.5 Full ablation (`make evaluate`)

Price ratio is the median |log2(result price / query price)|; 1.0 means 2× apart.

| Weights | co-purchase hit@10 | category@10 | segment@10 | brand@10 | price ratio |
|---|---|---|---|---|---|
| text only | 0.690 | 0.739 | 0.899 | 0.434 | 0.39 |
| image only | 0.299 | 0.574 | 0.824 | 0.086 | 0.56 |
| structured attributes only | 0.530 | 0.991 | 0.994 | 0.445 | 0.18 |
| text + image | 0.722 | 0.776 | 0.912 | 0.395 | 0.38 |
| default without image | 0.719 | 0.963 | 0.989 | 0.469 | 0.27 |
| default without text | 0.585 | 0.985 | 0.991 | 0.410 | 0.28 |
| default without brand | 0.722 | 0.977 | 0.990 | 0.352 | 0.29 |
| **default** | **0.735** | 0.964 | 0.987 | 0.452 | 0.29 |

Caveats on the proxies:
- Co-purchase is not the same as similarity, because complementary items are bought together too.
- Co-purchased products are matched by name, which is ambiguous for generic names.
- Category@10 is partly circular when the category component is on.

Use the numbers to compare configurations, not as an absolute quality score.

### A.6 Text and image pipelines

- **TF-IDF:** word 1–2 grams plus character 3–5 grams, each L2-normalised and weighted equally, so "tshirt" matches "t-shirt". LSA is a truncated SVD to 256 dimensions. Averaging TF-IDF and MiniLM scores reached the best co-purchase hit (0.551), but it was not adopted: it has lower category agreement (0.689) than MiniLM alone and makes the build more complex.
- **MiniLM:** `sentence-transformers/all-MiniLM-L6-v2` exported to ONNX and run with the `tokenizers` library (max 64 tokens, mean pooling). The 30k names take 27 s on CPU.
- **Images:**
  - Each product's 224 px rendition is padded to a white square rather than cropped, because garments are often tall.
  - ImageNet normalisation, then ResNet-50 (ImageNet weights) global-pool features (2048-d) via ONNX Runtime.
  - L2-normalised and reduced with PCA to 256-d, then centred and normalised again.
  - 29,885 of 30,000 products have an image after the fallbacks; embedding takes 7.5 min on CPU.

### A.7 Other trade-offs

- **Brand vs variety:** 45% of results share the query's brand. Without the brand component this drops to 35%, at a cost of 1.3 points of co-purchase hit.
- **Rating:** it describes quality, not the product itself, so it gets a small weight and mainly breaks ties. Many 5.0 ratings come from a single review.
- **Variants:** products with the same name in another colour or size are separate products and are returned. Exact duplicates (same ASIN) are not. Grouping variants would be a product decision.
