import argparse
import json
import sys

from .extract import extract


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ftsdoc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="extract structured data from a .txt/.pdf/.docx file")
    e.add_argument("path")
    e.add_argument("--json", action="store_true", help="print JSON (default)")
    e.add_argument("--backend", choices=["auto", "llm", "offline"], default="auto")
    e.add_argument("--doc-type", choices=["invoice", "cv", "contract", "tech_spec"], help="skip classification")
    s = sub.add_parser("serve", help="run the FastAPI server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("ftsdoc.api:app", host=a.host, port=a.port)
        return 0
    try:
        res = extract(a.path, backend=a.backend, doc_type=a.doc_type)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
