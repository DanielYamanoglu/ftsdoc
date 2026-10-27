"""Dataset loading and the fixed train/val/test split shared by all models."""
import json
from pathlib import Path

from .paths import DATA_FILE


def load_docs(path: Path = DATA_FILE) -> list[dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/generate_synthetic.py")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_docs(docs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """80/10/10 by document (the generator already shuffles with a fixed seed)."""
    n = len(docs)
    a, b = int(n * 0.8), int(n * 0.9)
    return docs[:a], docs[a:b], docs[b:]
