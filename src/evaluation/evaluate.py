"""Facade: ``run_evaluation`` wiring pipelines into :class:`Evaluator`.

Resolves each model tag to a concrete pipeline, delegates to
:class:`src.evaluation.evaluator.Evaluator`, and writes a combined
comparison report via :mod:`src.evaluation.report`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


def _build_pipeline(tag: str, cfg: Any) -> Any:
    """Resolve a model tag to a callable pipeline."""
    tag = tag.lower()
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


def _pipeline_callable(pipeline: Any):
    """Return a ``(query) -> QueryResponse`` callable from a pipeline obj."""
    run = getattr(pipeline, "run", None)
    if callable(run):
        return run
    if callable(pipeline):
        return pipeline
    raise TypeError(f"Pipeline {type(pipeline)} is neither callable nor has .run()")


def _resolve_test_set_path(test_dir: str | Path) -> Path:
    """Find a usable test set file under ``test_dir``.

    Prefers ``test.jsonl`` then ``qa_pairs.jsonl`` then the first
    ``*.jsonl`` / ``*.json`` entry.
    """
    p = Path(test_dir)
    if p.is_file():
        return p
    for name in ("test.jsonl", "qa_pairs.jsonl", "test.json", "qa_pairs.json"):
        candidate = p / name
        if candidate.exists():
            return candidate
    for candidate in sorted(p.glob("*.jsonl")):
        return candidate
    for candidate in sorted(p.glob("*.json")):
        return candidate
    raise FileNotFoundError(f"No test set file found under {p}")


def run_evaluation(
    models: list[str],
    test_dir: str | Path = "data/test",
    output_dir: str | Path = "report",
    config: Any | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate every tag in ``models`` and write a combined report.

    Parameters
    ----------
    models : list[str]
        Model tags, e.g. ``["b1", "b3", "m5"]``.
    test_dir : str or Path
        Directory containing the test set, or a direct JSONL path.
    output_dir : str or Path
        Destination for per-model JSON and the final comparison report.
    config : object, optional
        System config; passed through to M-series pipelines.

    Returns
    -------
    dict
        ``{model_tag: aggregate_metrics}``.
    """
    test_path = _resolve_test_set_path(test_dir)
    evaluator = Evaluator(test_set_path=test_path, output_dir=output_dir)

    pipelines: dict[str, Any] = {}
    for tag in models:
        try:
            pipe_obj = _build_pipeline(tag, config)
            pipelines[tag] = _pipeline_callable(pipe_obj)
        except Exception as exc:
            logger.exception("Failed to build pipeline for %s: %s", tag, exc)

    if not pipelines:
        logger.error("No pipelines could be built; evaluation aborted")
        return {}

    summary = evaluator.evaluate_all(pipelines)

    # Generate human-readable comparison artifacts (markdown + figs).
    try:
        from src.evaluation.report import generate_report

        summary_path = Path(output_dir) / "summary.json"
        if summary_path.exists():
            generate_report(summary_path, Path(output_dir))
    except Exception as exc:
        logger.warning("Report generation failed: %s", exc)

    return summary
