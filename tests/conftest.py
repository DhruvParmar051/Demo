"""Shared pytest fixtures for AegisRAG tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.schema import ChunkRecord  # noqa: E402


@pytest.fixture(scope="session")
def mini_corpus() -> list[ChunkRecord]:
    """Load the 10-chunk fixture corpus."""
    path = Path(__file__).parent / "fixtures" / "mini_corpus.json"
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    return [ChunkRecord.from_dict(r) for r in records]
