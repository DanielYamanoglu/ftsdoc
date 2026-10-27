import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def gen_module():
    spec = importlib.util.spec_from_file_location("generate_synthetic", ROOT / "scripts" / "generate_synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def sample_docs():
    with open(ROOT / "data" / "sample" / "docs.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(autouse=True)
def no_ner_model(monkeypatch):
    """Tests never load a transformer checkpoint."""
    from ftsdoc.ner.infer import NerModel
    monkeypatch.setattr(NerModel, "available", classmethod(lambda cls, model_dir=None: False))
