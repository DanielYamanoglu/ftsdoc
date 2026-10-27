from ftsdoc.ner.labels import LABELS
from ftsdoc.tokens import tokenize


def test_tokenize_splits_punctuation_and_offsets():
    text = "Invoice #INV-2024-00001 (due 2024-01-01), thanks."
    toks = tokenize(text)
    assert [t for t, _, _ in toks][:3] == ["Invoice", "#", "INV-2024-00001"]
    assert all(text[s:e] == t for t, s, e in toks)
    assert toks[-1][0] == "."


def test_generator_is_deterministic(gen_module):
    a = gen_module.Gen(1).invoice().text()
    b = gen_module.Gen(1).invoice().text()
    assert a == b


def test_bio_labels_are_valid_and_cover_all_spans(gen_module):
    g = gen_module.Gen(3)
    for dt in gen_module.DOC_TYPES:
        for _ in range(20):
            b = getattr(g, dt)()
            sents = gen_module.to_sentences(b)
            labels = [l for s in sents for l in s["labels"]]
            assert set(labels) <= set(LABELS)
            n_begin = sum(l.startswith("B-") for l in labels)
            assert n_begin == len(b.spans), "every entity span must produce exactly one B- token"


def test_sample_has_all_doc_types(sample_docs):
    assert {d["doc_type"] for d in sample_docs} == {"cv", "invoice", "contract", "tech_spec"}
