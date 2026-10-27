"""FastAPI service: POST /extract with a multipart file upload."""
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from .extract import build_chain
from .ingest import SUPPORTED

app = FastAPI(title="ftsdoc", version="0.1.0")
_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        _chain = build_chain("auto")
    return _chain


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(415, f"Unsupported file type {suffix!r}; expected one of {sorted(SUPPORTED)}")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"upload{suffix}"
        p.write_bytes(await file.read())
        try:
            result = _get_chain().invoke(p)
        except Exception as exc:  # surface a clean 422 rather than a stack trace
            raise HTTPException(422, f"Could not process document: {type(exc).__name__}") from exc
    return result.model_dump()
