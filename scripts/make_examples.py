"""Render sample documents as .txt/.pdf/.docx and run the pipeline on them.

    python scripts/make_examples.py
"""
import json
from pathlib import Path

import docx
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from ftsdoc.extract import build_chain

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "txt"
OUT = ROOT / "examples"
PICK = {"cv": ("cv_0000", ".txt"), "invoice": ("invoice_0001", ".pdf"),
        "contract": ("contract_0002", ".docx"), "tech_spec": ("tech_spec_0001", ".txt")}


def write_pdf(text: str, path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in text.split("\n"):
        c.drawString(50, y, line)
        y -= 16
    c.save()


def write_docx(text: str, path: Path) -> None:
    d = docx.Document()
    for line in text.split("\n"):
        d.add_paragraph(line)
    d.save(str(path))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    chain = build_chain("offline")
    for kind, (stem, ext) in PICK.items():
        text = (SAMPLE / f"{stem}.txt").read_text(encoding="utf-8")
        target = OUT / f"{kind}_example{ext}"
        {".txt": lambda t, p: p.write_text(t, encoding="utf-8"), ".pdf": write_pdf, ".docx": write_docx}[ext](text, target)
        res = chain.invoke(target)
        (OUT / f"{kind}_example.json").write_text(json.dumps(res.model_dump(), indent=2, ensure_ascii=False) + "\n")
        print(f"{target.name}: {res.doc_type} ({res.classifier_confidence:.3f})")


if __name__ == "__main__":
    main()
