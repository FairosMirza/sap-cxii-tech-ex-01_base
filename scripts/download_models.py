"""Download the ONNX models used offline into models/ (pinned revisions, no global HF cache)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))  # keep the HF cache inside the project

from huggingface_hub import hf_hub_download  # noqa: E402

MODELS = {
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                         ["onnx/model.onnx", "tokenizer.json"]),
    "resnet50-onnx": ("Qdrant/resnet50-onnx", "43b3ed7997b4b0fc59cf2402b12f31a4ec769743", ["model.onnx"]),
}


def main() -> None:
    for name, (repo, revision, files) in MODELS.items():
        for filename in files:
            path = hf_hub_download(repo, filename, revision=revision, local_dir=ROOT / "models" / name)
            print(path)


if __name__ == "__main__":
    main()
