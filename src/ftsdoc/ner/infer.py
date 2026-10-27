"""Token-classification inference: text -> entity spans."""
from __future__ import annotations

from dataclasses import dataclass

from ..paths import MODEL_DIR
from ..tokens import tokenize

NER_DIR = MODEL_DIR / "ner"


@dataclass(frozen=True)
class Entity:
    label: str
    text: str
    line: int
    start: int  # char offset in the full text
    end: int


class NerModel:
    def __init__(self, model_dir=NER_DIR) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).eval()

    @classmethod
    def available(cls, model_dir=NER_DIR) -> bool:
        return (model_dir / "config.json").exists()

    def predict(self, text: str) -> list[Entity]:
        ents: list[Entity] = []
        off = 0
        for li, line in enumerate(text.split("\n")):
            toks = tokenize(line)
            if toks:
                ents.extend(self._predict_line(line, toks, li, off))
            off += len(line) + 1
        return ents

    def _predict_line(self, line, toks, li, off) -> list[Entity]:
        words = [t[0] for t in toks]
        enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=256, return_tensors="pt")
        with self._torch.no_grad():
            logits = self.model(**enc).logits[0]
        pred = logits.argmax(-1).tolist()
        id2label = self.model.config.id2label
        word_labels = ["O"] * len(words)
        seen = set()
        for pos, w in enumerate(enc.word_ids(0)):
            if w is not None and w not in seen:
                seen.add(w)
                word_labels[w] = id2label[pred[pos]]
        return _group(line, toks, word_labels, li, off)


def _group(line, toks, labels, li, off) -> list[Entity]:
    out, cur = [], None
    for (tok, s, e), lab in zip(toks, labels):
        if lab == "O":
            cur = _close(out, cur, line, li, off)
            continue
        prefix, typ = lab.split("-", 1)
        if prefix == "B" or cur is None or cur[0] != typ:
            cur = _close(out, cur, line, li, off)
            cur = [typ, s, e]
        else:
            cur[2] = e
    _close(out, cur, line, li, off)
    return out


def _close(out, cur, line, li, off):
    if cur:
        out.append(Entity(cur[0], line[cur[1]:cur[2]], li, cur[1] + off, cur[2] + off))
    return None
