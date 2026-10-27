import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from ftsdoc import classify as clf
from ftsdoc.__main__ import main
from ftsdoc.extract import build_chain, pick_backend
from ftsdoc.offline import fill, regex_entities
from ftsdoc.schemas import Invoice

INVOICE = """INVOICE
Invoice No: INV-2025-73304
Issue Date: 2025-03-13
Due Date: 2025-04-12

From: Garnet Labs
Bill To: Altavue Systems
Contact: billing@example.org

Total Due: EUR 81,443.03
"""


def test_regex_entities_find_core_fields():
    labels = {(e.label, e.text) for e in regex_entities(INVOICE)}
    assert ("INVOICE_NO", "INV-2025-73304") in labels
    assert ("EMAIL", "billing@example.org") in labels
    assert ("MONEY", "EUR 81,443.03") in labels


def test_fill_invoice_uses_line_context():
    inv = fill("invoice", INVOICE, regex_entities(INVOICE))
    assert inv.invoice_number == "INV-2025-73304"
    assert inv.issue_date == "2025-03-13" and inv.due_date == "2025-04-12"
    assert inv.total == "EUR 81,443.03"


def test_tech_spec_key_values():
    text = "Model: KX-1234-A\nOperating voltage: 12 V\nWeight: 2.0 kg\nSupport: a@example.com\n"
    spec = fill("tech_spec", text, regex_entities(text))
    assert spec.part_number == "KX-1234-A"
    assert spec.specs == {"Operating voltage": "12 V", "Weight": "2.0 kg"}


@pytest.fixture
def trained_classifier(monkeypatch, sample_docs, tmp_path):
    pipe = clf.build_pipeline().set_params(tfidf__min_df=1)
    pipe.fit([d["text"] for d in sample_docs], [d["doc_type"] for d in sample_docs])
    monkeypatch.setattr(clf, "_cache", pipe)
    return pipe


def test_classifier_on_sample(trained_classifier, sample_docs):
    label, conf = clf.classify(sample_docs[0]["text"])
    assert label == sample_docs[0]["doc_type"] and 0 < conf <= 1


def test_chain_offline_end_to_end(trained_classifier, tmp_path):
    p = tmp_path / "inv.txt"
    p.write_text(INVOICE, encoding="utf-8")
    res = build_chain("offline").invoke(p)
    assert res.doc_type == "invoice" and res.backend == "offline"
    assert res.data.invoice_number == "INV-2025-73304"


class StubLLM:
    """Mimics a LangChain chat model's with_structured_output without any network."""

    def with_structured_output(self, schema):
        return RunnableLambda(lambda _: schema(invoice_number="INV-STUB"))


def test_chain_llm_backend_with_stub(tmp_path):
    p = tmp_path / "inv.txt"
    p.write_text(INVOICE, encoding="utf-8")
    res = build_chain("llm", llm=StubLLM(), doc_type="invoice").invoke(p)
    assert res.backend == "llm" and isinstance(res.data, Invoice) and res.data.invoice_number == "INV-STUB"


def test_pick_backend(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert pick_backend("auto") == "offline"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert pick_backend("auto") == "llm"


def test_cli_json(tmp_path, capsys, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    p = tmp_path / "inv.txt"
    p.write_text(INVOICE, encoding="utf-8")
    assert main(["extract", str(p), "--json", "--doc-type", "invoice"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["data"]["vendor"] is None  # no NER model in tests: regex-only path
    assert out["data"]["total"] == "EUR 81,443.03"


def test_cli_missing_file(capsys):
    assert main(["extract", "nope.txt", "--doc-type", "cv"]) == 1


def test_api_extract(monkeypatch, tmp_path):
    from ftsdoc import api
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(api, "_chain", build_chain("offline", doc_type="invoice"))
    client = TestClient(api.app)
    r = client.post("/extract", files={"file": ("inv.txt", INVOICE.encode(), "text/plain")})
    assert r.status_code == 200 and r.json()["data"]["invoice_number"] == "INV-2025-73304"
    assert client.post("/extract", files={"file": ("x.exe", b"x")}).status_code == 415


def test_committed_examples_are_valid_json():
    for p in (Path(__file__).resolve().parents[1] / "examples").glob("*.json"):
        assert json.loads(p.read_text())["doc_type"] in {"cv", "invoice", "contract", "tech_spec"}
