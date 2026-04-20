"""Thin facade exposing a ``run_ingestion`` orchestrator for ``run.py``.

Delegates to :class:`src.data.ingestion.DocumentIngestor`. This module
exists so the CLI entry point can import a single well-named function
without having to wire up the class itself.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from src.data.chunker import RecursiveChunker
from src.data.ingestion import DocumentIngestor
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def run_ingestion(
    source_dir: str | Path,
    vector_db_path: str | Path | None = None,
    bm25_index_path: str | Path | None = None,
    chunk_size: int = 256,
    chunk_overlap: int = 64,
    min_chunk_size: int = 30,
    dedup_threshold: float = 0.85,
    embedding_model: str | None = None,
    supported_extensions: Iterable[str] | None = None,
    domain: str = "",
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the full ingestion pipeline with CLI-friendly kwargs.

    Constructs the vector store, BM25 index, and chunker from the
    provided parameters (falling back to config defaults where any
    argument is ``None``) and invokes
    :meth:`DocumentIngestor.ingest`.

    Returns
    -------
    dict
        Ingestion stats produced by :class:`DocumentIngestor`.
    """
    vs_kwargs: dict[str, Any] = {}
    if vector_db_path is not None:
        vs_kwargs["persist_directory"] = str(vector_db_path)
    if embedding_model:
        vs_kwargs["embedding_model_name"] = embedding_model
    vector_store = ChromaVectorStore(**vs_kwargs)
    bm25 = BM25Index()
    chunker = RecursiveChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )

    ingestor = DocumentIngestor(
        vector_store=vector_store,
        bm25_index=bm25,
        chunker=chunker,
        dedup_threshold=dedup_threshold,
        domain=domain,
    )
    if supported_extensions is not None:
        ingestor.supported_ext = {str(e).lower() for e in supported_extensions}

    stats = ingestor.ingest(Path(source_dir), limit=limit, save_bm25=True)
    if bm25_index_path:
        try:
            bm25.save(Path(bm25_index_path))
        except Exception as exc:
            logger.warning("Could not save BM25 to %s: %s", bm25_index_path, exc)
    return stats
