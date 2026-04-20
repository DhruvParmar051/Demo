"""
AegisRAG - Document Ingestion Pipeline

Walks a source directory, parses each supported file, chunks the text,
deduplicates near-duplicate chunks via MinHash LSH, then upserts the
survivors into :class:`ChromaVectorStore` and builds a persisted
:class:`BM25Index`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.data.chunker import RecursiveChunker
from src.data.parsers import get_parser
from src.data.schema import ChunkRecord
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """End-to-end ingestion: parse, chunk, dedup, index.

    Parameters
    ----------
    vector_store : ChromaVectorStore, optional
        Reuse an existing store; otherwise one is constructed from config.
    bm25_index : BM25Index, optional
        Reuse an existing index; otherwise a fresh one is created.
    chunker : RecursiveChunker, optional
        Reuse an existing chunker; otherwise one is created from config.
    dedup_threshold : float, optional
        Jaccard threshold for MinHash LSH deduplication (default 0.85).
    num_perm : int, optional
        Number of MinHash permutations (default 128).
    domain : str
        Domain tag attached to every chunk.
    seed : int
        Deterministic seed (default 42).
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        bm25_index: BM25Index | None = None,
        chunker: RecursiveChunker | None = None,
        dedup_threshold: float = 0.85,
        num_perm: int = 128,
        domain: str = "",
        seed: int = 42,
    ) -> None:
        set_seed(seed)
        self.seed = seed

        cfg = get_config()
        self.cfg = cfg

        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.chunker = chunker or RecursiveChunker()

        self.dedup_threshold = float(dedup_threshold)
        self.num_perm = int(num_perm)
        self.domain = domain

        self.supported_ext = {e.lower() for e in cfg.data.supported_extensions}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        source_dir: Path,
        limit: int | None = None,
        save_bm25: bool = True,
    ) -> dict[str, Any]:
        """Ingest every supported file under ``source_dir`` recursively.

        Parameters
        ----------
        source_dir : Path
            Root directory of raw documents.
        limit : int or None
            Optional cap on number of documents (handy for dev).
        save_bm25 : bool
            If True, persist the BM25 index to
            ``cfg.data.bm25_index_path`` on completion.

        Returns
        -------
        dict
            ``{"docs_seen": int, "chunks_produced": int, "dupes_removed": int}``
        """
        source_dir = Path(source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Source dir not found: {source_dir}")

        files = self._discover_files(source_dir, limit=limit)
        logger.info("Discovered %d ingestable files under %s", len(files), source_dir)

        all_chunks: list[ChunkRecord] = []
        docs_seen = 0

        for fp in files:
            try:
                chunks = self._process_file(fp)
            except Exception as exc:
                logger.exception("Failed to ingest %s: %s", fp, exc)
                continue

            docs_seen += 1
            all_chunks.extend(chunks)

        logger.info(
            "Parsed+chunked %d docs into %d raw chunks", docs_seen, len(all_chunks)
        )

        deduped, dupes_removed = self._deduplicate(all_chunks)
        logger.info(
            "Dedup complete: %d unique chunks, %d near-duplicates removed",
            len(deduped),
            dupes_removed,
        )

        if deduped:
            self.vector_store.add_chunks(deduped)
            self.bm25_index.build_index(deduped)
            if save_bm25:
                self.bm25_index.save(self.cfg.resolve_path(self.cfg.data.bm25_index_path))

        stats = {
            "docs_seen": docs_seen,
            "chunks_produced": len(deduped),
            "dupes_removed": dupes_removed,
        }
        logger.info("Ingestion stats: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _discover_files(
        self, source_dir: Path, limit: int | None = None
    ) -> list[Path]:
        files: list[Path] = []
        for fp in sorted(source_dir.rglob("*")):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in self.supported_ext:
                continue
            files.append(fp)
        if limit is not None:
            files = files[:limit]
        return files

    # ------------------------------------------------------------------
    # Per-file processing
    # ------------------------------------------------------------------

    def _process_file(self, path: Path) -> list[ChunkRecord]:
        """Parse and chunk a single file."""
        suffix = path.suffix.lower()
        if suffix not in self.supported_ext:
            logger.debug("Skipping unsupported file: %s", path)
            return []

        try:
            parser = get_parser(path)
        except ValueError as exc:
            logger.warning("No parser for %s: %s", path, exc)
            return []

        blocks = parser.parse(path)
        if not blocks:
            return []

        doc_id = ChunkRecord.generate_doc_id(str(path.resolve()))
        source = str(path)
        chunks: list[ChunkRecord] = []
        running_idx = 0

        for block in blocks:
            text = (block.get("text") or "").strip()
            if not text:
                continue
            page = block.get("page_number")
            block_chunks = self.chunker.chunk(
                text=text,
                doc_id=doc_id,
                source=source,
                page_number=page,
                domain=self.domain,
                chunk_index_offset=running_idx,
            )
            chunks.extend(block_chunks)
            running_idx += len(block_chunks)

        return chunks

    # ------------------------------------------------------------------
    # Deduplication via MinHash LSH
    # ------------------------------------------------------------------

    def _deduplicate(
        self, chunks: list[ChunkRecord]
    ) -> tuple[list[ChunkRecord], int]:
        """Remove near-duplicate chunks using datasketch MinHashLSH.

        Returns ``(unique_chunks, num_dupes_removed)``. Chunks are
        processed in their original order so the first occurrence is
        retained.
        """
        if not chunks:
            return [], 0

        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            logger.warning(
                "datasketch not installed; skipping deduplication. "
                "Install with: pip install datasketch"
            )
            return chunks, 0

        lsh = MinHashLSH(threshold=self.dedup_threshold, num_perm=self.num_perm)
        unique: list[ChunkRecord] = []
        dupes = 0

        for chunk in chunks:
            mh = MinHash(num_perm=self.num_perm, seed=self.seed)
            for token in self._shingles(chunk.text):
                mh.update(token.encode("utf-8"))

            matches = lsh.query(mh)
            if matches:
                dupes += 1
                continue

            lsh.insert(chunk.chunk_id, mh)
            unique.append(chunk)

        return unique, dupes

    @staticmethod
    def _shingles(text: str, n: int = 5) -> list[str]:
        """Produce word-level n-gram shingles for MinHash."""
        words = text.lower().split()
        if len(words) < n:
            return [" ".join(words)] if words else []
        return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
