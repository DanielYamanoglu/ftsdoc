import docx
import pytest
from reportlab.pdfgen import canvas

from ftsdoc.ingest import load_and_split, load_documents, load_text


def test_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello\nworld", encoding="utf-8")
    assert load_text(p) == "hello\nworld"


def test_docx(tmp_path):
    p = tmp_path / "a.docx"
    d = docx.Document()
    d.add_paragraph("Invoice No: INV-2024-00001")
    d.save(str(p))
    assert "INV-2024-00001" in load_text(p)


def test_pdf(tmp_path):
    p = tmp_path / "a.pdf"
    c = canvas.Canvas(str(p))
    c.drawString(72, 700, "Total Due: $120.00")
    c.save()
    assert "Total Due" in load_text(p)


def test_split_makes_chunks(tmp_path):
    p = tmp_path / "long.txt"
    p.write_text("word " * 1000, encoding="utf-8")
    assert len(load_and_split(p, chunk_size=500, chunk_overlap=50)) > 5


def test_unsupported_extension(tmp_path):
    p = tmp_path / "a.xyz"
    p.write_text("x")
    with pytest.raises(ValueError):
        load_documents(p)
