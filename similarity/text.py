"""Text embeddings for product names: TF-IDF / LSA (scikit-learn) and MiniLM (ONNX).

Offline only. The service never imports scikit-learn or onnxruntime.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def tfidf_matrix(texts: Sequence[str]):
    """Word 1-2 grams + character 3-5 grams, each L2-normalised and weighted equally.

    Character n-grams make "tshirt", "t-shirt" and "kurti"/"kurta" overlap and
    tolerate misspelled brand names.
    """
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    return (hstack([word.fit_transform(texts), char.fit_transform(texts)]) / np.sqrt(2)).tocsr()


def lsa_embeddings(texts: Sequence[str], dim: int = 256, seed: int = 0) -> np.ndarray:
    """Latent semantic analysis: truncated SVD of the TF-IDF matrix (Deerwester et al., 1990)."""
    from sklearn.decomposition import TruncatedSVD

    return TruncatedSVD(dim, algorithm="randomized", random_state=seed).fit_transform(tfidf_matrix(texts))


class MiniLMEncoder:
    """sentence-transformers/all-MiniLM-L6-v2 with mean pooling, run by onnxruntime."""

    def __init__(self, model_dir: Path, max_length: int = 64, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or os.cpu_count() or 1
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        out = np.zeros((len(texts), 384), dtype=np.float32)
        order = np.argsort([len(t) for t in texts])  # similar lengths per batch -> less padding
        for start in range(0, len(texts), batch_size):
            rows = order[start : start + batch_size]
            encoded = self.tokenizer.encode_batch([texts[i] for i in rows])
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            hidden = self.session.run(
                None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            )[0]
            weights = mask[..., None].astype(np.float32)
            out[rows] = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        return out
