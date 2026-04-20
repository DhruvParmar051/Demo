"""AegisRAG document parsers.

Exports the per-format parser classes plus a :func:`get_parser` factory
that dispatches by file extension.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from src.data.parsers.docx_parser import DOCXParser
from src.data.parsers.pdf_parser import PDFParser
from src.data.parsers.txt_parser import TXTParser

__all__ = ["DOCXParser", "PDFParser", "TXTParser", "Parser", "get_parser"]


@runtime_checkable
class Parser(Protocol):
    """Minimal protocol every parser implementation must satisfy."""

    def parse(self, path: Path) -> list[dict]: ...


_PDF_EXT = {".pdf"}
_DOCX_EXT = {".docx"}
_TXT_EXT = {".txt", ".md", ".markdown"}


def get_parser(path: Path | str) -> Parser:
    """Return a parser instance appropriate for ``path``'s extension.

    Raises ``ValueError`` for unsupported extensions.
    """
    suffix = Path(path).suffix.lower()
    if suffix in _PDF_EXT:
        return PDFParser()
    if suffix in _DOCX_EXT:
        return DOCXParser()
    if suffix in _TXT_EXT:
        return TXTParser()
    raise ValueError(f"Unsupported file extension: {suffix!r} for path {path}")
