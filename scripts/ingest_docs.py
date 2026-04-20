"""Ingest documents from a source directory into Chroma + BM25.

Usage:
    python scripts/ingest_docs.py --source-dir data/raw_docs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Repo root on sys.path so `python scripts/foo.py` works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ingestion import DocumentIngestor  # noqa: E402
from src.utils.config import get_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = get_config()
    ingestor = DocumentIngestor(cfg)
    stats = ingestor.ingest(Path(args.source_dir))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
