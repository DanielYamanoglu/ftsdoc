"""Document-type classifier: TF-IDF + logistic regression baseline."""
from __future__ import annotations

import json
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from .data import load_docs, split_docs
from .paths import MODEL_DIR, RESULTS_DIR

CLF_PATH = MODEL_DIR / "classifier.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=True)),
        ("lr", LogisticRegression(max_iter=1000, C=10.0)),
    ])


def train() -> dict:
    train_d, _, test_d = split_docs(load_docs())
    pipe = build_pipeline().fit([d["text"] for d in train_d], [d["doc_type"] for d in train_d])
    pred = pipe.predict([d["text"] for d in test_d])
    y = [d["doc_type"] for d in test_d]
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    metrics = {"n_train": len(train_d), "n_test": len(test_d), "accuracy": accuracy_score(y, pred),
               "macro_f1": report["macro avg"]["f1-score"],
               "per_class_f1": {k: v["f1-score"] for k, v in report.items() if k in set(y)}}
    CLF_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, CLF_PATH)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


_cache: Pipeline | None = None


def load_classifier() -> Pipeline:
    global _cache
    if _cache is None:
        if not CLF_PATH.exists():
            raise FileNotFoundError(f"{CLF_PATH} not found. Run: python -m ftsdoc.classify")
        _cache = joblib.load(CLF_PATH)
    return _cache


def classify(text: str) -> tuple[str, float]:
    pipe = load_classifier()
    probs = pipe.predict_proba([text])[0]
    i = int(probs.argmax())
    return str(pipe.classes_[i]), float(probs[i])


if __name__ == "__main__":
    json.dump(train(), sys.stdout, indent=2)
    print()
