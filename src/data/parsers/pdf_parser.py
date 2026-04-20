"""
AegisRAG - PDF Parser

Extracts text blocks and tables from PDF documents. Primary backend is
``pdfplumber`` (good table support). If unavailable, falls back to
``pdfminer.six`` for plain-text extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PDFParser:
    """Parse a PDF file into a list of text blocks with page numbers.

    The return schema is a list of ``{"text": str, "page_number": int}``
    dicts. Tables are flattened by joining cells with tab separators and
    rows with newline separators, then appended after the page's prose.
    """

    def __init__(self) -> None:
        self._backend = self._select_backend()

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_backend() -> str:
        """Return 'pdfplumber' if importable, else 'pdfminer'."""
        try:
            import pdfplumber  # noqa: F401

            return "pdfplumber"
        except ImportError:
            pass
        try:
            from pdfminer.high_level import extract_text  # noqa: F401

            logger.warning(
                "pdfplumber not installed; falling back to pdfminer.six "
                "(tables will not be extracted). Install pdfplumber for "
                "best results: pip install pdfplumber"
            )
            return "pdfminer"
        except ImportError as exc:
            raise ImportError(
                "PDFParser requires either 'pdfplumber' or 'pdfminer.six'. "
                "Install with: pip install pdfplumber"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, path: Path) -> list[dict[str, Any]]:
        """Parse a PDF file and return a list of text blocks.

        Parameters
        ----------
        path : Path
            Absolute or relative path to the PDF file.

        Returns
        -------
        list of dict
            Each dict has keys ``text`` (str) and ``page_number`` (int,
            1-indexed).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        logger.info("Parsing PDF (%s backend): %s", self._backend, path)

        if self._backend == "pdfplumber":
            return self._parse_pdfplumber(path)
        return self._parse_pdfminer(path)

    # ------------------------------------------------------------------
    # pdfplumber backend
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_table(table: list[list[str | None]]) -> str:
        """Convert a pdfplumber table (list of rows) to a text block."""
        rows: list[str] = []
        for row in table:
            cells = ["" if c is None else str(c).strip() for c in row]
            rows.append("\t".join(cells))
        return "\n".join(rows)

    def _parse_pdfplumber(self, path: Path) -> list[dict[str, Any]]:
        import pdfplumber

        blocks: list[dict[str, Any]] = []
        with pdfplumber.open(str(path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                # Plain prose
                try:
                    text = page.extract_text() or ""
                except Exception as exc:  # defensive: corrupted pages
                    logger.warning(
                        "pdfplumber failed on page %d of %s: %s",
                        page_idx,
                        path.name,
                        exc,
                    )
                    text = ""

                text = text.strip()
                if text:
                    blocks.append({"text": text, "page_number": page_idx})

                # Tables flattened to text
                try:
                    tables = page.extract_tables() or []
                except Exception as exc:
                    logger.warning(
                        "Table extraction failed on page %d of %s: %s",
                        page_idx,
                        path.name,
                        exc,
                    )
                    tables = []

                for table in tables:
                    flat = self._flatten_table(table).strip()
                    if flat:
                        blocks.append(
                            {
                                "text": f"[TABLE]\n{flat}",
                                "page_number": page_idx,
                            }
                        )

        logger.info("PDF parsed: %d blocks extracted from %s", len(blocks), path.name)
        return blocks

    # ------------------------------------------------------------------
    # pdfminer fallback
    # ------------------------------------------------------------------

    def _parse_pdfminer(self, path: Path) -> list[dict[str, Any]]:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        from io import StringIO

        blocks: list[dict[str, Any]] = []
        # pdfminer does not easily expose per-page text without more
        # machinery. We iterate page-by-page using page_numbers.
        try:
            import pdfminer.pdfpage as pdfpage

            with open(path, "rb") as fh:
                num_pages = sum(1 for _ in pdfpage.PDFPage.get_pages(fh))
        except Exception as exc:
            logger.warning("Could not count PDF pages, defaulting to 1: %s", exc)
            num_pages = 1

        for page_idx in range(num_pages):
            out = StringIO()
            with open(path, "rb") as fh:
                extract_text_to_fp(
                    fh,
                    out,
                    laparams=LAParams(),
                    page_numbers=[page_idx],
                )
            text = out.getvalue().strip()
            if text:
                blocks.append({"text": text, "page_number": page_idx + 1})

        logger.info("PDF parsed: %d blocks extracted from %s", len(blocks), path.name)
        return blocks
