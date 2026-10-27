"""Fine-tune a small transformer for token classification (BIO) on the synthetic set.

    python -m ftsdoc.ner.train --model google/bert_uncased_L-4_H-256_A-4 --epochs 12
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments, set_seed)

from ..data import load_docs, split_docs
from ..paths import MODEL_DIR, RESULTS_DIR
from .labels import ID2LABEL, LABEL2ID, LABELS

NER_DIR = MODEL_DIR / "ner"


def _sentences(docs: list[dict]) -> list[dict]:
    return [s for d in docs for s in d["sentences"]]


def _encode(tokenizer, sents: list[dict], max_len: int) -> Dataset:
    ds = Dataset.from_list(sents)

    def fn(batch):
        enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
        enc["labels"] = []
        for i, labs in enumerate(batch["labels"]):
            prev, ids = None, []
            for w in enc.word_ids(i):
                # label only the first sub-word of each word
                ids.append(-100 if w is None or w == prev else LABEL2ID[labs[w]])
                prev = w
            enc["labels"].append(ids)
        return enc

    return ds.map(fn, batched=True, remove_columns=ds.column_names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/bert_uncased_L-4_H-256_A-4")
    ap.add_argument("--epochs", type=float, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    train_d, val_d, test_d = split_docs(load_docs())
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    tr, va, te = (_encode(tok, _sentences(d), args.max_len) for d in (train_d, val_d, test_d))

    def compute(p):
        preds, labels = p
        preds = preds.argmax(-1)
        y_true = [[ID2LABEL[l] for l in row if l != -100] for row in labels]
        y_pred = [[ID2LABEL[int(q)] for q, l in zip(pr, row) if l != -100] for pr, row in zip(preds, labels)]
        return {"precision": precision_score(y_true, y_pred), "recall": recall_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred)}

    targs = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=64,
        eval_strategy="epoch", save_strategy="no", logging_steps=50, report_to="none",
        weight_decay=0.01, warmup_steps=0.1, seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=tr, eval_dataset=va,
                      data_collator=DataCollatorForTokenClassification(tok), compute_metrics=compute)
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    out = trainer.predict(te)
    preds, labels = out.predictions.argmax(-1), out.label_ids
    y_true = [[ID2LABEL[l] for l in row if l != -100] for row in labels]
    y_pred = [[ID2LABEL[int(q)] for q, l in zip(pr, row) if l != -100] for pr, row in zip(preds, labels)]
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "base_model": args.model, "epochs": args.epochs, "train_minutes": round(minutes, 2),
        "n_train_sentences": len(tr), "n_test_sentences": len(te),
        "overall": {"precision": precision_score(y_true, y_pred), "recall": recall_score(y_true, y_pred),
                    "f1": f1_score(y_true, y_pred)},
        "per_entity": {k: {m: float(v[m]) for m in ("precision", "recall", "f1-score", "support")}
                       for k, v in rep.items() if k not in ("micro avg", "macro avg", "weighted avg")},
    }
    NER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(NER_DIR))
    tok.save_pretrained(str(NER_DIR))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "ner_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_true, y_pred, digits=4))
    print(json.dumps(metrics["overall"]))


if __name__ == "__main__":
    main()
