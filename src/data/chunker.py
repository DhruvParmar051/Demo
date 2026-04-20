"""
AegisRAG - Recursive Token Chunker

Splits raw document text into overlapping token-bounded chunks. Uses
``tiktoken`` for tokenisation (cl100k_base) so chunk sizes align with
common transformer tokenizers. Character-level span offsets into the
original text are preserved for citation purposes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.data.schema import ChunkRecord
from src.utils.config import get_config

logger = logging.getLogger(__name__)


# Boundaries preferred when splitting text, from strongest to weakest.
_BREAK_PATTERNS = [
    re.compile(r"\n{2,}"),   # paragraph break
    re.compile(r"(?<=[.!?])\s+"),  # sentence break
    re.compile(r"\n"),       # line break
    re.compile(r"\s+"),      # whitespace
]


class RecursiveChunker:
    """Token-bounded recursive text splitter with overlap and span tracking.

    Parameters are sourced from ``cfg.retrieval`` when not supplied:
    ``chunk_size`` (256), ``chunk_overlap`` (64), ``min_chunk_size`` (30).
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        cfg = get_config()
        self.chunk_size = int(
            chunk_size if chunk_size is not None else cfg.retrieval.chunk_size
        )
        self.chunk_overlap = int(
            chunk_overlap
            if chunk_overlap is not None
            else cfg.retrieval.chunk_overlap
        )
        self.min_chunk_size = int(
            min_chunk_size
            if min_chunk_size is not None
            else cfg.retrieval.min_chunk_size
        )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )

        # tiktoken is a hard dependency here; we need token-accurate sizing.
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "RecursiveChunker requires 'tiktoken'. "
                "Install with: pip install tiktoken"
            ) from exc

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            logger.warning(
                "Could not load tiktoken encoding %s, falling back to cl100k_base",
                encoding_name,
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        doc_id: str,
        source: str,
        page_number: int | None = None,
        domain: str = "",
        chunk_index_offset: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkRecord]:
        """Split ``text`` into :class:`ChunkRecord` objects.

        Parameters
        ----------
        text : str
            Raw text to split.
        doc_id : str
            Stable id of the source document.
        source : str
            Origin path or URL of the source document.
        page_number : int or None
            Optional page number (PDF only).
        domain : str
            Business/topic tag copied onto every chunk.
        chunk_index_offset : int
            Starting ``chunk_index`` value (useful when the same ``doc_id``
            is chunked across multiple pages).
        metadata : dict or None
            Extra metadata to attach verbatim to each chunk.

        Returns
        -------
        list[ChunkRecord]
            Sequential chunks, each with ``span_start`` / ``span_end``
            character offsets into the original ``text``.
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        tokens = self.encoding.encode(text)
        if len(tokens) <= self.chunk_size:
            # Fits in a single chunk; emit directly (still enforce min size).
            if len(tokens) < self.min_chunk_size:
                return []
            return [
                self._make_chunk(
                    text=text,
                    doc_id=doc_id,
                    source=source,
                    page_number=page_number,
                    domain=domain,
                    chunk_index=chunk_index_offset,
                    token_count=len(tokens),
                    span_start=0,
                    span_end=len(text),
                    metadata=metadata,
                )
            ]

        # Produce overlapping windows.
        chunks: list[ChunkRecord] = []
        step = self.chunk_size - self.chunk_overlap
        n = len(tokens)
        chunk_idx = chunk_index_offset

        # We need (token_slice -> character_slice) mapping. Recompute the
        # decoded substring's character offset by decoding incrementally.
        # tiktoken.decode is lossless for complete token sequences, but
        # boundaries may not align to our break patterns exactly; we use
        # character-level search to refine boundaries.

        cursor_char = 0
        for start in range(0, n, step):
            end = min(start + self.chunk_size, n)
            token_slice = tokens[start:end]
            chunk_text = self.encoding.decode(token_slice)

            # Locate chunk_text in the original string starting from cursor.
            span_start, span_end = self._locate_span(text, chunk_text, cursor_char)

            # Try to refine span_start/span_end to natural boundaries when
            # possible (improves citation readability).
            refined_text, span_start, span_end = self._refine_boundaries(
                text, span_start, span_end
            )

            token_count = len(self.encoding.encode(refined_text))
            if token_count < self.min_chunk_size:
                # Advance cursor but skip emitting tiny tail chunks.
                cursor_char = max(cursor_char, span_end - self._overlap_chars())
                if end >= n:
                    break
                continue

            chunks.append(
                self._make_chunk(
                    text=refined_text,
                    doc_id=doc_id,
                    source=source,
                    page_number=page_number,
                    domain=domain,
                    chunk_index=chunk_idx,
                    token_count=token_count,
                    span_start=span_start,
                    span_end=span_end,
                    metadata=metadata,
                )
            )
            chunk_idx += 1

            # Move cursor forward leaving overlap room.
            cursor_char = max(cursor_char, span_end - self._overlap_chars())

            if end >= n:
                break

        logger.debug(
            "Chunked doc_id=%s into %d chunks (page=%s, %d tokens)",
            doc_id,
            len(chunks),
            page_number,
            n,
        )
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _overlap_chars(self) -> int:
        """Rough character equivalent of the token overlap (4 chars/token)."""
        return max(1, self.chunk_overlap * 4)

    @staticmethod
    def _locate_span(
        full_text: str, needle: str, start_from: int
    ) -> tuple[int, int]:
        """Find ``needle`` in ``full_text`` near ``start_from``.

        tiktoken round-trips *most* text exactly, but some whitespace and
        special characters normalise. We fall back to nearest-match on a
        prefix/suffix if exact match fails.
        """
        if not needle:
            return start_from, start_from

        idx = full_text.find(needle, max(0, start_from - 16))
        if idx >= 0:
            return idx, idx + len(needle)

        # Try prefix match (first ~48 chars of needle)
        probe = needle[: min(48, len(needle))]
        idx = full_text.find(probe, max(0, start_from - 16))
        if idx >= 0:
            return idx, idx + len(needle)

        # Fall back to cursor; span length derived from needle length.
        return start_from, start_from + len(needle)

    def _refine_boundaries(
        self, full_text: str, start: int, end: int
    ) -> tuple[str, int, int]:
        """Nudge ``[start, end)`` toward natural boundaries (paragraph,
        sentence, line, space) when the shift is small (< 64 chars).
        """
        max_shift = 64
        n = len(full_text)
        end = min(end, n)
        start = max(0, start)

        # Refine start: move forward to next natural boundary
        for pattern in _BREAK_PATTERNS:
            m = pattern.search(full_text, start, min(start + max_shift, n))
            if m and m.end() < end:
                start = m.end()
                break

        # Refine end: move backward to previous natural boundary
        search_begin = max(end - max_shift, start + 1)
        for pattern in _BREAK_PATTERNS:
            matches = list(pattern.finditer(full_text, search_begin, end))
            if matches:
                end = matches[-1].end()
                break

        return full_text[start:end].strip(), start, end

    @staticmethod
    def _make_chunk(
        text: str,
        doc_id: str,
        source: str,
        page_number: int | None,
        domain: str,
        chunk_index: int,
        token_count: int,
        span_start: int,
        span_end: int,
        metadata: dict[str, Any],
    ) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=ChunkRecord.generate_chunk_id(doc_id, chunk_index),
            doc_id=doc_id,
            text=text,
            source=source,
            page_number=page_number,
            chunk_index=chunk_index,
            token_count=token_count,
            span_start=span_start,
            span_end=span_end,
            section_title="",
            domain=domain,
            metadata=dict(metadata),
        )
