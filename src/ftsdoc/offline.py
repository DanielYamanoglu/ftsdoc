"""Offline backend: fill the schemas from NER entities plus regex fallbacks (no API keys)."""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel

from .ner.infer import Entity
from .schemas import SCHEMAS

_REGEX = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+\w"),
    "INVOICE_NO": re.compile(r"\bINV-\d{4}-\d{5}\b"),
    "PART_NO": re.compile(r"\b[A-Z]{2}-\d{3,4}-[A-Z]\b"),
    "MONEY": re.compile(r"(?:\$|EUR |GBP )\d{1,3}(?:,\d{3})*(?:\.\d{2})?"),
    "DATE": re.compile(
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\.\d{2}\.\d{4}|(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December) \d{1,2}, \d{4}|\d{1,2} (?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December) \d{4})\b"),
}
_SPEC_LINE = re.compile(r"^([A-Z][A-Za-z ]{2,30}):\s+(.+)$")
_NOT_SPEC = {"model", "manufacturer", "revision date", "list price", "support"}


def regex_entities(text: str) -> list[Entity]:
    out, off = [], 0
    for li, line in enumerate(text.split("\n")):
        for label, rx in _REGEX.items():
            for m in rx.finditer(line):
                out.append(Entity(label, m.group(), li, off + m.start(), off + m.end()))
        off += len(line) + 1
    return out


def merge(primary: list[Entity], extra: list[Entity]) -> list[Entity]:
    """Keep NER entities; add regex entities that do not overlap any of them."""
    kept = list(primary)
    for e in extra:
        if not any(e.start < p.end and p.start < e.end for p in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.start)


def _first(ents: list[Entity], label: str, line_hint: Optional[str] = None, lines: list[str] | None = None):
    for e in ents:
        if e.label == label and (line_hint is None or lines[e.line].lower().lstrip().startswith(line_hint)):
            return e.text
    return None


def _uniq(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def fill(doc_type: str, text: str, ents: list[Entity]) -> BaseModel:
    lines = text.split("\n")
    texts = lambda label: _uniq([e.text for e in ents if e.label == label])  # noqa: E731
    first = lambda label, hint=None: _first(ents, label, hint, lines)  # noqa: E731

    if doc_type == "invoice":
        total = next((e.text for e in ents if e.label == "MONEY" and "total" in lines[e.line].lower()), None)
        return SCHEMAS[doc_type](
            invoice_number=first("INVOICE_NO"), issue_date=first("DATE", "issue"), due_date=first("DATE", "due"),
            vendor=first("ORG", "from"), customer=first("ORG", "bill"), contact_email=first("EMAIL"), total=total)
    if doc_type == "cv":
        return SCHEMAS[doc_type](name=first("PERSON"), email=first("EMAIL"), organizations=texts("ORG"),
                                 skills=texts("SKILL"))
    if doc_type == "contract":
        return SCHEMAS[doc_type](
            effective_date=first("DATE"), parties=texts("ORG"), fee=first("MONEY"), notice_email=first("EMAIL"),
            signatories=texts("PERSON"))
    if doc_type == "tech_spec":
        specs = {}
        for line in lines:
            m = _SPEC_LINE.match(line.strip())
            if m and m.group(1).lower() not in _NOT_SPEC:
                specs[m.group(1)] = m.group(2).strip()
        return SCHEMAS[doc_type](
            part_number=first("PART_NO"), manufacturer=first("ORG"), revision_date=first("DATE"),
            list_price=first("MONEY"), support_email=first("EMAIL"), specs=specs)
    raise ValueError(f"Unknown doc_type {doc_type!r}")
