"""Facade: ``run_benchmark`` for the CLI ``benchmark`` subcommand.

Samples queries from the synthetic test set (or a small canned list),
runs the pipeline ``n`` times, and reports p50/p95/p99 latency plus
throughput (queries/sec) and mean confidence. Uses ``time.perf_counter``
and is deliberately single-threaded so numbers match production-serving
latency rather than saturated throughput.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_DEFAULT_QUERIES: tuple[str, ...] = (
    "How do I reset my password?",
    "What is the refund policy for a cancelled subscription?",
    "Can I transfer my license to another user?",
    "How do I enable two-factor authentication?",
    "What are the supported payment methods?",
    "How long does shipping take?",
    "How do I update my billing address?",
    "What happens if my payment fails?",
    "How do I cancel my account?",
    "Is my data encrypted at rest?",
)


def _load_queries(config: Any | None, n: int) -> list[str]:
    """Pull queries from ``data/synthetic/qa_pairs.jsonl`` when available."""
    candidate_paths: list[Path] = []
    if config is not None:
        syn_dir = getattr(getattr(config, "data", None), "synthetic_dir", None)
        if syn_dir:
            candidate_paths.append(Path(syn_dir) / "qa_pairs.jsonl")
    candidate_paths.append(Path("data/synthetic/qa_pairs.jsonl"))

    for path in candidate_paths:
        if not path.exists():
            continue
        queries: list[str] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    q = obj.get("query") if isinstance(obj, dict) else None
                    if q:
                        queries.append(str(q))
                    if len(queries) >= n:
                        break
            if queries:
                return queries
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)

    # Fallback: cycle canned queries.
    return [_DEFAULT_QUERIES[i % len(_DEFAULT_QUERIES)] for i in range(n)]


def _build_pipeline(model_tag: str, cfg: Any) -> Any:
    tag = model_tag.lower()
    if tag == "b1":
        from src.models.baselines import BaselineB1
        return BaselineB1()
    if tag == "b2":
        from src.models.baselines import BaselineB2
        return BaselineB2()
    if tag == "b3":
        from src.models.baselines import BaselineB3
        return BaselineB3()
    from src.models.m5_pipeline import M5Pipeline
    return M5Pipeline.from_tag(tag, cfg)


def run_benchmark(
    model_tag: str,
    n: int = 100,
    config: Any | None = None,
) -> dict[str, Any]:
    """Run ``n`` single-threaded queries and return latency statistics."""
    pipeline = _build_pipeline(model_tag, config)
    run = getattr(pipeline, "run", None)
    if not callable(run):
        raise TypeError(f"Pipeline {model_tag} has no .run() method")

    queries = _load_queries(config, n)

    # If CUDA is in play, we need torch.cuda.synchronize() around each query
    # or wall-clock timing will report the enqueue time, not the real latency.
    _sync = None
    try:  # noqa: SIM105
        import torch  # type: ignore

        if torch.cuda.is_available():
            _sync = torch.cuda.synchronize
    except Exception:
        _sync = None

    latencies_ms: list[float] = []
    confidences: list[float] = []

    wall_start = time.perf_counter()
    for i, q in enumerate(queries):
        if _sync is not None:
            _sync()
        t0 = time.perf_counter()
        try:
            resp = run(q)
            if _sync is not None:
                _sync()
        except Exception as exc:
            logger.warning("Benchmark query %d failed: %s", i, exc)
            continue
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        conf = getattr(resp, "confidence", None)
        if conf is None and isinstance(resp, dict):
            conf = resp.get("confidence")
        if conf is not None:
            try:
                confidences.append(float(conf))
            except (TypeError, ValueError):
                pass
    wall_elapsed = time.perf_counter() - wall_start

    if not latencies_ms:
        raise RuntimeError("All benchmark queries failed; no latencies collected")

    latencies_ms.sort()

    def _pct(p: float) -> float:
        k = max(0, min(len(latencies_ms) - 1, int(round(p / 100.0 * (len(latencies_ms) - 1)))))
        return latencies_ms[k]

    return {
        "model_tag": model_tag,
        "n_queries": float(len(latencies_ms)),
        "avg_latency_ms": float(statistics.fmean(latencies_ms)),
        "p50_latency_ms": _pct(50),
        "p95_latency_ms": _pct(95),
        "p99_latency_ms": _pct(99),
        "throughput_qps": len(latencies_ms) / wall_elapsed if wall_elapsed > 0 else 0.0,
        "avg_confidence": float(statistics.fmean(confidences)) if confidences else 0.0,
    }
