# Stage 1: build the feature store from the raw catalogue and the precomputed
# embeddings (artifacts/embeddings, produced offline by scripts/embed.py).
FROM python:3.14-slim AS build
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir "$(grep '^numpy==' requirements.txt)"
COPY similarity/ similarity/
COPY data/archive.zip data/
COPY artifacts/embeddings/ artifacts/embeddings/
RUN python -m similarity.build --data data/archive.zip --out /build/index

# Stage 2: the API only. No raw data, no ONNX/scikit-learn, runs as non-root.
FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INDEX_DIR=/app/index \
    SEARCH_METHOD=ann
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --system --uid 10001 --no-create-home app
COPY similarity/ similarity/
COPY app.py .
COPY --from=build /build/index index/
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready', timeout=2)"
# One worker per container; scale with replicas (see k8s/).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
