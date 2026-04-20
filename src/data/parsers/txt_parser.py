"""
AegisRAG - Plain-text / Markdown Parser

Handles ``.txt`` and ``.md`` files. Uses ``chardet`` to detect the file
encoding so we can gracefully ingest legacy documents that are not
UTF-8.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TXTParser:
    """Parse a plain-text or Markdown file into a single text block.

    Output schema: ``[{"text": str, "page_number": None}]``.
    The full document is returned as one block; chunking into smaller
    pieces is handled downstream by :class:`RecursiveChunker`.
    """

    SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}

    def __init__(self, fallback_encoding: str = "utf-8") -> None:
        self.fallback_encoding = fallback_encoding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, path: Path) -> list[dict[str, Any]]:
        """Parse a ``.txt`` or ``.md`` file.

        Parameters
        ----------
        path : Path
            Path to the text file.

        Returns
        -------
        list of dict
            ``[{"text": str, "page_number": None}]`` if the file has
            content, otherwise ``[]``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        logger.info("Parsing text file: %s", path)

        raw = path.read_bytes()
        if not raw:
            return []

        encoding = self._detect_encoding(raw)
        try:
            text = raw.decode(encoding, errors="replace")
        except (LookupError, UnicodeDecodeError):
            logger.warning(
                "Could not decode %s as %s, falling back to %s",
                path.name,
                encoding,
                self.fallback_encoding,
            )
            text = raw.decode(self.fallback_encoding, errors="replace")

        text = text.strip()
        if not text:
            return []

        return [{"text": text, "page_number": None}]

    # ------------------------------------------------------------------
    # Encoding detection
    # ------------------------------------------------------------------

    def _detect_encoding(self, raw: bytes) -> str:
        """Return best-guess encoding for ``raw`` using chardet if available."""
        try:
            import chardet
        except ImportError:
            logger.debug(
                "chardet not installed; using fallback encoding %s",
                self.fallback_encoding,
            )
            return self.fallback_encoding

        # Detect on a reasonable sample (full file for small, prefix for big)
        sample = raw if len(raw) <= 65536 else raw[:65536]
        detection = chardet.detect(sample) or {}
        encoding = detection.get("encoding") or self.fallback_encoding
        confidence = detection.get("confidence", 0.0) or 0.0

        if confidence < 0.5:
            logger.debug(
                "chardet confidence %.2f for %s; using fallback %s",
                confidence,
                encoding,
                self.fallback_encoding,
            )
            return self.fallback_encoding

        return encoding
