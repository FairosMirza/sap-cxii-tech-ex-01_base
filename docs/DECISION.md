# Decision summary: why not the simplest approach?

The simplest valid solution to this exercise fits in about a hundred lines:
- one vector per product: TF-IDF of the name, one-hot brand and colour, standardised price, weight and rating
- cosine similarity against every product, with sklearn or an exact FAISS index
- return the top IDs

On clean data that is a reasonable answer, and it would cover Parts 1 and 2.

This submission goes further in three situations only:
1. The data broke the simple version.
2. The assignment asked for more: Part 3 (bonus) and the optional multimodal part.
3. A test or a load run showed a real failure.

Each decision below names the simpler alternative, the reason it was not enough, the evidence, and what it costs. Where a choice is more than this 30k-product catalogue strictly needs, it says so.

## 1. Where the simple version breaks on this dataset

These are measured facts about the supplied data, not hypotheticals ([report §4](REPORT.md#4-the-data-and-how-it-was-cleaned)). The middle column follows directly from these facts; a full simple baseline was not benchmarked end to end.

| Data fact | What the simple version does | What this solution does |
|---|---|---|
| 79% of `weight` values are the placeholder `999999999` | standardising puts most products at one extreme value; the rest crowd together | treat it as unknown and leave weight out of that query's score |
| `colour` lists the colours of *other variants* | a blue shirt is encoded as "red, black, blue" | read the item's own colour from the URL or name first (70% of products); use the variant list only when nothing else exists (10%) |
| 451 products are listed 2–3 times under different IDs | the top "similar" result is the product itself under another ID | exclude the query's ASIN and return at most one listing per ASIN |
| 27% of brands missing, with case variants (`BIBA` vs `Biba`) | missing one-hots match each other; variants count as different brands | normalise names, infer brand from the start of the name, and never let an unknown brand match |
| 15% of categories missing | category falls out of the score | fill it from the browse node (95.8% agreement) |
| Prices run from ₹39 to ₹9,988 | on a linear scale, cheap items all look the same | log scale, clipped at the 1st/99th percentile |
| The only text is the product name (no description) | TF-IDF misses synonyms in short names | MiniLM name embeddings, after a measured comparison (below) |

## 2. Decisions beyond the baseline

| Decision | Simpler alternative | Why it wasn't enough | Evidence | Cost |
|---|---|---|---|---|
| **Score = weighted average of 10 per-attribute similarities** | One concatenated vector | Weights become explicit and adjustable per request. Each attribute's contribution to a result can be shown, which makes the ranking explainable. | Worked example ([report §5.2](REPORT.md#52-worked-example)); ablation ([§7.2](REPORT.md#72-does-each-component-help)) | More code than a single vector |
| **Weights rescaled per query over the attributes it actually has** | Fill gaps with 0 or the mean and score everything | Missing data is the norm (weight 79%, brand 23%, price 10%). With zero-filling, products with missing values look similar *because* they are missing. | Tests for the query-side and candidate-side rules | Harder to explain than one formula |
| **MiniLM for names (offline, ONNX Runtime)** | TF-IDF | Compared before adopting: +7.8 points of category agreement and +3 points of segment agreement, with equal co-purchase hit rate | [§7.1](REPORT.md#71-text-model-tf-idf-vs-lsa-vs-minilm) | 27 s of offline embedding; nothing at query time |
| **ResNet-50 image features** | No images | The assignment's optional multimodal part. It uses actual pixels, not image-URL text. | +1.6 points of co-purchase hit rate. That is a small gain, and it is reported as small. | 5 min image download + 8 min embedding, offline |
| **Prices and weight encoded as angles** | Compare price differences directly at query time | Needed only so numbers can sit in the same vector index as the embeddings. Exact search alone would not need it. | The index and the exact scorer give the same numeric similarity | Less intuitive; mitigated by tests and the explanation endpoint |
| **Random 64-d codes for brand/category inside the index** | Integer labels, full one-hot vectors, or leaving them out | Integer labels imply brand 5 is "closer" to brand 6 than to brand 900. One-hot brands need 6.3k dimensions. Leaving them out loses candidates. | Final top-10 overlap with exact search: 0.9996 (4 of 1,000 lists differ) | The index score is approximate for these two attributes (documented) |
| **HNSW index + exact re-ranking** | Exact search only | Part 3 asks for vector search. Plainly: at 30k products, exact search (1.42 ms) is fast enough. | 0.33 ms vs 1.42 ms here; on 1M synthetic vectors, 0.21 ms vs 7.9 ms | Extra moving parts; 104 MB index in the image |
| **One exact scorer behind both search paths** | Return the HNSW results as they come | Tie-breaks, ASIN dedupe and missing-data rules live in one place, so both search methods behave the same | Exact fallback when HNSW candidates run short (regression test) | An exact scoring pass over the 100 candidates per query |
| **Index built at image build time, with a data fingerprint** | Build the index when the service starts | Building it at startup took 14–33 s on 1 CPU, and pods restart-looped on a busy node | Startup ~1 s; 0 restarts with 3 of 4 CPUs saturated | Larger image: 644 MB on disk, 200 MB compressed |
| **`preStop` hook and tuned probes** | Kubernetes defaults | A load test dropped 12 of 2,575 requests during a rolling restart | 0 of 2,990 afterwards | Pods take 5 s longer to shut down |
| **Embeddings precomputed and stored in Git LFS; no ML libraries in the API image** | Compute embeddings inside the container | Reproducible builds without downloading 30k images or ML frameworks. Smaller attack surface. | The API image installs only numpy, FAISS, FastAPI and uvicorn (plus their dependencies) | Reviewers need git-lfs, or the ZIP |
| **CI that deploys to a local Kubernetes cluster** | Unit tests only | Proves the Dockerfile and manifests work on every change, not once by hand | Green runs on every push to main | CI minutes |

## 3. Common pitfalls, checked deliberately

Simpler designs for this kind of exercise tend to fail in recurring ways. Each one was checked here:

- **Ranking order changed after retrieval** (for example, re-sorting the results by price): price is only the third tie-breaker; a test proves order follows the score.
- **The query removed by position instead of by ID**, so duplicates leave it in the results: exclusion uses the ID and the ASIN.
- **Category or brand integer codes treated as distances**: exact search uses equality; the index uses random codes.
- **Image URL text treated as visual features**: actual pixels go through a CNN.
- **Correlated price fields counted twice**: sale price weight 0.08, derived list price 0.02. The overlap is disclosed.
- **Variant colour lists treated as the item's colour**: the item's own colour comes first, and the fallback is documented.
- **Saved indexes validated only by row count**: the index checks a content fingerprint and rebuilds on mismatch.
- **"Bonus" code that the service never uses**: HNSW is the default serving path, and the image weight measurably changes results.

## 4. Deliberately not done

The following were left out because nothing in the assignment needed them, and each adds operational weight:
- a vector database, Redis or other external cache
- an LLM or a trained ranking model
- a free-text semantic search endpoint
- async endpoints for CPU-bound work
- multiple uvicorn workers per pod
- an Ingress or autoscaler
- t-SNE

Weights were set by hand and checked with an ablation, not tuned to the co-purchase proxy, which would overfit a weak signal.

## 5. If the scope had to be cut

The minimum sufficient version is:
- the data cleaning in §1
- the exact scorer (name text, category, brand, log price, with the missing-data rule)
- the API
- the Docker image

The rest is optional. In the order I would drop it:
1. CI deployment and the Kubernetes extras
2. The HNSW index (exact search is fast enough at this size)
3. Image features
4. MiniLM (the TF-IDF fallback still works)

## 6. Known weaknesses

- **Hand-set weights.** They come from judgement and an ablation, not from relevance labels. With click or purchase data, the next step would be learning them and validating with an A/B test.
- **Proxy metrics only.** Co-purchase is not the same as similarity, and the matching is done on names. The numbers compare configurations; they do not prove relevance.
- **Generic image model.** ImageNet ResNet-50 sees shape and colour but not style. A fashion-tuned model or CLIP would likely do better.
- **Contradictory source data.** Some listings disagree with themselves, for example a URL that says brown and a title that says black. Both colours are kept rather than guessed between.
- **More machinery than 30k products strictly need.** The index exists for the bonus and for scale. It is verified against exact search, which remains available with `method=exact`.
