"""Compatibility command; prefer python -m backend.evaluation --method bm25."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(default_method="bm25"))
