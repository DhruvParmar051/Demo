"""Synthetic data generation orchestrator.

Usage
-----
    python scripts/generate_data.py --type qa
    python scripts/generate_data.py --type all --output-dir data/synthetic

Pipeline
--------
1. Load already-ingested ``ChunkRecord`` objects from the ChromaDB store.
2. Run :class:`QAGenerator` over the chunks -> ``qa_pairs.jsonl``.
3. Run downstream generators (preference, confidence, alpha, decomposition)
   that consume those QA pairs. ``confidence`` and ``alpha`` additionally
   require the live ``HybridRetriever`` so they can grade retrieval quality.

Each downstream generator is best-effort: a failure in one is logged and
tracked in the returned summary but does not abort the rest of the run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.schema import ChunkRecord, QAPair  # noqa: E402
from src.utils.config import get_config  # noqa: E402

logger = logging.getLogger(__name__)


_TYPES = ("qa", "preference", "confidence", "alpha", "decomp", "all")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_chunks(limit: int | None = None) -> list[ChunkRecord]:
    """Load every chunk currently in the ChromaDB collection."""
    from src.retrieval.vector_store import ChromaVectorStore  # noqa: WPS433

    store = ChromaVectorStore()
    raw = store.collection.get(include=["documents", "metadatas"])
    chunks: list[ChunkRecord] = []
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    for cid, text, meta in zip(ids, docs, metas):
        meta = meta or {}
        chunks.append(
            ChunkRecord(
                chunk_id=str(cid),
                doc_id=str(meta.get("doc_id", cid)),
                text=str(text or ""),
                source=str(meta.get("source", "")),
                domain=str(meta.get("domain", "")),
                page_number=meta.get("page_number"),
                span_start=int(meta.get("span_start", 0) or 0),
                span_end=int(meta.get("span_end", len(text or "")) or 0),
                chunk_index=int(meta.get("chunk_index", 0) or 0),
            )
        )
        if limit is not None and len(chunks) >= limit:
            break
    logger.info("Loaded %d chunks from vector store", len(chunks))
    return chunks


def _load_qa_pairs(path: Path) -> list[QAPair]:
    """Parse a QA-pair JSONL produced earlier in the pipeline."""
    if not path.exists():
        logger.warning("QA pair file missing: %s", path)
        return []
    pairs: list[QAPair] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            try:
                pairs.append(QAPair(**obj))
            except TypeError:
                # Tolerate extra keys persisted by older versions.
                known = {
                    k: v for k, v in obj.items() if k in QAPair.__dataclass_fields__
                }
                pairs.append(QAPair(**known))
    logger.info("Loaded %d QA pairs from %s", len(pairs), path)
    return pairs


def _build_retriever() -> Any:
    """Construct a ``HybridRetriever`` bound to the live store + BM25 index."""
    from src.retrieval.bm25_index import BM25Index  # noqa: WPS433
    from src.retrieval.retriever import HybridRetriever  # noqa: WPS433
    from src.retrieval.vector_store import ChromaVectorStore  # noqa: WPS433

    cfg = get_config()
    bm25 = BM25Index()
    bm25_path = getattr(cfg.paths, "bm25_index", None)
    if bm25_path:
        try:
            bm25.load(bm25_path)
        except Exception as exc:
            logger.warning("Could not load BM25 index: %s", exc)
    return HybridRetriever(vector_store=ChromaVectorStore(), bm25_index=bm25)


# ---------------------------------------------------------------------------
# Per-dataset runners
# ---------------------------------------------------------------------------


def _run_qa(out_dir: Path) -> dict[str, Any]:
    from src.data.qa_generator import QAGenerator  # noqa: WPS433

    chunks = _load_chunks()
    if not chunks:
        return {"status": "skipped", "reason": "no_chunks_in_store", "count": 0}
    out_path = out_dir / "qa_pairs.jsonl"
    pairs = QAGenerator().generate(chunks=chunks, output_path=out_path)
    return {"status": "ok", "count": len(pairs), "output": str(out_path)}


def _run_preference(out_dir: Path) -> dict[str, Any]:
    from src.data.preference_generator import PreferenceGenerator  # noqa: WPS433

    qa = _load_qa_pairs(out_dir / "qa_pairs.jsonl")
    if not qa:
        return {"status": "skipped", "reason": "no_qa_pairs", "count": 0}
    out_path = out_dir / "preferences.jsonl"
    triplets = PreferenceGenerator().generate(qa_pairs=qa, output_path=out_path)
    return {"status": "ok", "count": len(triplets), "output": str(out_path)}


def _run_confidence(out_dir: Path) -> dict[str, Any]:
    from src.data.confidence_label_generator import (  # noqa: WPS433
        ConfidenceLabelGenerator,
    )

    qa = _load_qa_pairs(out_dir / "qa_pairs.jsonl")
    if not qa:
        return {"status": "skipped", "reason": "no_qa_pairs", "count": 0}
    retriever = _build_retriever()
    out_path = out_dir / "confidence_labels.jsonl"
    labels = ConfidenceLabelGenerator(retriever=retriever).generate(
        qa_pairs=qa, output_path=out_path
    )
    return {"status": "ok", "count": len(labels), "output": str(out_path)}


def _run_alpha(out_dir: Path) -> dict[str, Any]:
    from src.data.alpha_label_generator import AlphaLabelGenerator  # noqa: WPS433

    qa = _load_qa_pairs(out_dir / "qa_pairs.jsonl")
    if not qa:
        return {"status": "skipped", "reason": "no_qa_pairs", "count": 0}
    retriever = _build_retriever()
    out_path = out_dir / "alpha_labels.jsonl"
    labels = AlphaLabelGenerator(retriever=retriever).generate(
        qa_pairs=qa, output_path=out_path
    )
    return {"status": "ok", "count": len(labels), "output": str(out_path)}


def _run_decomp(out_dir: Path) -> dict[str, Any]:
    from src.data.decomp_label_generator import DecompLabelGenerator  # noqa: WPS433

    qa = _load_qa_pairs(out_dir / "qa_pairs.jsonl")
    if not qa:
        return {"status": "skipped", "reason": "no_qa_pairs", "count": 0}
    out_path = out_dir / "decomp_labels.jsonl"
    labels = DecompLabelGenerator().generate(qa_pairs=qa, output_path=out_path)
    return {"status": "ok", "count": len(labels), "output": str(out_path)}


_DISPATCH: dict[str, Callable[[Path], dict[str, Any]]] = {
    "qa": _run_qa,
    "preference": _run_preference,
    "confidence": _run_confidence,
    "alpha": _run_alpha,
    "decomp": _run_decomp,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(type_: str = "all", output_dir: str | Path = "data/synthetic") -> dict[str, dict]:
    """Programmatic entry point (mirrors the CLI)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    targets = list(_DISPATCH) if type_ == "all" else [type_]
    for t in targets:
        logger.info("Running generator: %s", t)
        try:
            results[t] = _DISPATCH[t](out_dir)
        except Exception as exc:
            logger.exception("Generator %s failed", t)
            results[t] = {"status": "error", "error": str(exc)}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=_TYPES, default="all")
    parser.add_argument("--output-dir", default="data/synthetic")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    results = run(type_=args.type, output_dir=args.output_dir)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
