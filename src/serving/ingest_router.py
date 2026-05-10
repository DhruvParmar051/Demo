"""Deployment-time document ingestion endpoint.

This router is intentionally separate from ``src/serving/app.py`` so it can be
mounted opt-in with a single line::

    from src.serving.ingest_router import build_ingest_router
    app.include_router(build_ingest_router(config))

At deployment time, end-users upload PDF / DOCX / TXT / MD files. The upload
is parsed, chunked, deduplicated, embedded, and indexed into the live
ChromaDB + BM25 stores used by the M5 pipeline. No synthetic training data
is generated on upload -- chunks alone are enough to power evidence-grounded
answers.
"""

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)

_ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}
_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB per file is plenty for policy docs.


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in _ALLOWED_SUFFIXES else ""


def _persist_upload(tmp_dir: Path, filename: str, content: bytes) -> Path:
    """Write uploaded bytes to disk under tmp_dir, preserving extension."""
    suffix = _safe_suffix(filename)
    if not suffix:
        raise ValueError(f"Unsupported file type: {filename}")
    # Use the original stem so the resulting doc_id is human-recognizable.
    stem = Path(filename).stem or "upload"
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in ("-", "_"))[:64] or "upload"
    dst = tmp_dir / f"{safe_stem}{suffix}"
    # If the stem collides, append a counter.
    counter = 1
    while dst.exists():
        dst = tmp_dir / f"{safe_stem}_{counter}{suffix}"
        counter += 1
    dst.write_bytes(content)
    return dst


def build_ingest_router(config: Any | None = None):
    """Construct the ingestion APIRouter.

    ``config`` is passed through to DocumentIngestor. When None, the ingestor
    will fall back to its own default config resolution.
    """
    from fastapi import APIRouter, File, Header, HTTPException, UploadFile

    router = APIRouter(tags=["ingest"])

    @router.post("/ingest", response_model=None)
    async def ingest_files(
        files: List[UploadFile] = File(...),
        collection_id: str | None = Header(None, alias="X-Collection-ID"),
    ) -> dict:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        started = time.time()
        tmp_dir = Path(tempfile.mkdtemp(prefix="aegis_upload_"))
        saved_paths: List[Path] = []
        rejected: List[dict] = []

        try:
            for upload in files:
                raw = await upload.read()
                if not raw:
                    rejected.append({"filename": upload.filename, "reason": "empty"})
                    continue
                if len(raw) > _MAX_FILE_BYTES:
                    rejected.append({
                        "filename": upload.filename,
                        "reason": f"exceeds {_MAX_FILE_BYTES} bytes",
                    })
                    continue
                if not _safe_suffix(upload.filename or ""):
                    rejected.append({"filename": upload.filename, "reason": "unsupported type"})
                    continue
                try:
                    dst = _persist_upload(tmp_dir, upload.filename or "upload", raw)
                except ValueError as exc:
                    rejected.append({"filename": upload.filename, "reason": str(exc)})
                    continue
                saved_paths.append(dst)

            if not saved_paths:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "No valid files.", "rejected": rejected},
                )

            # Lazy import so the router can be constructed in environments
            # where heavy deps aren't installed yet (tests, CI).
            from src.data.ingestion import DocumentIngestor  # noqa: WPS433
            from src.retrieval.bm25_index import BM25Index
            from src.utils.config import get_config as _get_config

            # Load existing BM25 index so new chunks are APPENDED, not replacing
            # the full corpus. Without this, every upload would overwrite the index
            # with only the new document's chunks.
            _cfg = config or _get_config()
            _bm25_path = Path(_cfg.resolve_path(_cfg.data.bm25_index_path))
            _bm25 = BM25Index()
            if _bm25_path.exists():
                try:
                    _bm25.load(str(_bm25_path))
                    logger.info(
                        "Loaded existing BM25 index (%d docs) for incremental update.",
                        _bm25.size,
                    )
                except Exception as _exc:
                    logger.warning("Could not load BM25 index: %s — starting fresh.", _exc)

            ingestor = DocumentIngestor(
                collection_name=collection_id or _cfg.data.vector_db_collection,
                bm25_index=_bm25,
            )
            stats = ingestor.ingest(Path(tmp_dir))

            # DocumentIngestor's return shape varies slightly across versions;
            # normalize to a stable contract.
            chunks_added = 0
            doc_ids: List[str] = []
            if isinstance(stats, dict):
                chunks_added = int(
                    stats.get("chunks_added")
                    or stats.get("chunks_produced")
                    or stats.get("num_chunks")
                    or 0
                )
                doc_ids = list(stats.get("doc_ids") or [])

            elapsed_ms = int((time.time() - started) * 1000)
            response = {
                "status": "ok",
                "files_accepted": [p.name for p in saved_paths],
                "files_rejected": rejected,
                "chunks_added": chunks_added,
                "doc_ids": doc_ids,
                "latency_ms": elapsed_ms,
            }
            if isinstance(stats, dict):
                # Surface any extra fields (dedup_ratio, num_docs, etc.) without
                # clobbering our normalized keys.
                for k, v in stats.items():
                    response.setdefault(k, v)
            return response

        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Ingest failed")
            raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @router.get("/ingest/health")
    async def ingest_health() -> dict:
        return {
            "status": "ok",
            "allowed_suffixes": sorted(_ALLOWED_SUFFIXES),
            "max_file_bytes": _MAX_FILE_BYTES,
        }

    return router
