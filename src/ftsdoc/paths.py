import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = Path(os.environ.get("FTSDOC_DATA", ROOT / "data" / "generated" / "docs.jsonl"))
MODEL_DIR = Path(os.environ.get("FTSDOC_MODEL_DIR", ROOT / "models"))
RESULTS_DIR = ROOT / "results"
