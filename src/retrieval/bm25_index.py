"""
AegisRAG - BM25 Sparse Retrieval Index

Provides keyword-based retrieval using BM25Okapi from rank_bm25.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.data.schema import ChunkRecord

logger = logging.getLogger(__name__)

# Pre-compiled regex for tokenisation
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = _PUNCT_RE.sub("", text.lower())
    return text.split()


class BM25Index:
    """In-memory BM25Okapi index over ``ChunkRecord`` objects.

    Build with :meth:`build_index`, query with :meth:`query`, and
    persist / restore with :meth:`save` / :meth:`load`.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[ChunkRecord] = []
        self._id_to_chunk: dict[str, ChunkRecord] = {}
        self._corpus_tokens: list[list[str]] = []

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[ChunkRecord]) -> None:
        """Tokenize all chunks and build the BM25Okapi index.

        Parameters
        ----------
        chunks : list[ChunkRecord]
            The chunks to index.  Previous index state is replaced.
        """
        if not chunks:
            logger.warning("build_index called with empty chunk list")
            return

        self._chunks = list(chunks)
        self._id_to_chunk = {c.chunk_id: c for c in self._chunks}
        self._corpus_tokens = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

        logger.info("BM25 index built with %d documents", len(self._chunks))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self, query_text: str, top_k: int = 20
    ) -> list[tuple[ChunkRecord, float]]:
        """Return the *top_k* chunks ranked by BM25 score.

        Parameters
        ----------
        query_text : str
            Raw query string.
        top_k : int
            Number of results to return.

        Returns
        -------
        list of (ChunkRecord, float)
            Chunks paired with their BM25 scores (higher is better).
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("BM25 query called but index is empty")
            return []

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices (descending score)
        top_k = min(top_k, len(self._chunks))
        top_indices = scores.argsort()[::-1][:top_k]

        results: list[tuple[ChunkRecord, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break  # remaining scores are zero or negative
            results.append((self._chunks[idx], score))

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize the full index state to disk via pickle.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``data/bm25_index.pkl``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "chunks": self._chunks,
            "corpus_tokens": self._corpus_tokens,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("BM25 index saved to %s (%d docs)", path, len(self._chunks))

    def load(self, path: str | Path) -> None:
        """Load index state from a pickle file and rebuild BM25.

        Parameters
        ----------
        path : str or Path
            Path to a previously saved index file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301

        self._chunks = state["chunks"]
        self._corpus_tokens = state["corpus_tokens"]
        self._id_to_chunk = {c.chunk_id: c for c in self._chunks}
        self._bm25 = BM25Okapi(self._corpus_tokens)

        logger.info("BM25 index loaded from %s (%d docs)", path, len(self._chunks))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return len(self._chunks)
