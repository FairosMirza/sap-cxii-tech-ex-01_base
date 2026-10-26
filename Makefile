PY ?= .venv/bin/python
IMAGE ?= product-similarity:1.0.0
CLUSTER ?= similarity
KUBECTL ?= kubectl --context kind-$(CLUSTER)

.PHONY: setup models images embeddings index serve test compare evaluate benchmark docker-build docker-run kind-deploy kind-smoke kind-delete

setup:  ## virtualenv with the offline + test dependencies
	python3 -m venv .venv && $(PY) -m pip install -r requirements-dev.txt

models:  ## ONNX models (MiniLM, ResNet-50) into models/
	$(PY) -m scripts.download_models

images:  ## product images into data/images/ (cached)
	$(PY) -m scripts.download_images

embeddings:  ## text + image embeddings into artifacts/embeddings/
	$(PY) -m scripts.embed

index:  ## feature store served by the API into artifacts/index/
	$(PY) -m similarity.build

serve: index
	.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000

test:
	$(PY) -m pytest -q

compare:  ## TF-IDF vs LSA vs MiniLM
	$(PY) -m scripts.compare_text_models

evaluate:  ## ablation of the scoring components
	$(PY) -m scripts.evaluate

benchmark:  ## exact vs HNSW, plus synthetic scale-up
	$(PY) -m scripts.benchmark

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

kind-deploy: docker-build
	kind get clusters | grep -qx $(CLUSTER) || kind create cluster --name $(CLUSTER)
	kind load docker-image $(IMAGE) --name $(CLUSTER)
	$(KUBECTL) apply -f k8s/
	$(KUBECTL) rollout status deployment/product-similarity --timeout=180s

kind-smoke:  ## API checks + zero-downtime rolling restart under load
	KUBECTL="$(KUBECTL)" IMAGE=$(IMAGE) scripts/k8s_smoke_test.sh

kind-delete:
	kind delete cluster --name $(CLUSTER)
