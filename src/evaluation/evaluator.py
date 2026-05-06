"""
AegisRAG - Evaluator orchestrator.

The ``Evaluator`` class loads a test set (JSON/JSONL of gold annotations),
runs one or more pipelines on it, collects per-query metrics, computes
aggregates, and persists results. It also offers paired-bootstrap
significance testing and a simple ablation harness.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from src.data.schema import Citation, QueryResponse, ToolCall
from src.evaluation.fcrs import compute_fcrs
from src.evaluation.metrics import (
    answer_quality,
    cgal_efficiency,
    citation_f1,
    decomposition_accuracy,
    escalation_f1,
    grounding_score,
    recall_at_k,
    tool_accuracy,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers for loading test sets / adapting pipeline outputs
# ----------------------------------------------------------------------


def _load_test_set(path: Path) -> list[dict[str, Any]]:
    """Load a test set from JSON (list) or JSONL.

    Args:
        path: Path to a ``.json`` or ``.jsonl`` file.

    Returns:
        List of gold-annotation dicts. Each dict is expected to contain
        at least ``query``; the remaining keys are consumed by metric
        functions (e.g. ``key_points``, ``needed_tool`` for FCRS).
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            data = json.load(fh)
        else:
            data = [json.loads(line) for line in fh if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Test set at {path} is not a list")
    return data


def _to_query_response(obj: Any) -> QueryResponse:
    """Coerce a pipeline return value into a QueryResponse.

    Args:
        obj: Either a ``QueryResponse`` already or a dict compatible with
            the QueryResponse schema.

    Returns:
        A QueryResponse instance.
    """
    if isinstance(obj, QueryResponse):
        return obj
    if isinstance(obj, dict):
        citations = [
            Citation(**c) if isinstance(c, dict) else c
            for c in obj.get("citations", [])
        ]
        tool_calls = [
            ToolCall(**t) if isinstance(t, dict) else t
            for t in obj.get("tool_calls", [])
        ]
        payload = {k: v for k, v in obj.items() if k not in {"citations", "tool_calls"}}
        return QueryResponse(citations=citations, tool_calls=tool_calls, **payload)
    raise TypeError(f"Cannot coerce {type(obj)} to QueryResponse")


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------


class Evaluator:
    """Orchestrator for running pipelines over a test set and scoring them.

    Attributes:
        test_set_path: Path to the JSON/JSONL test set.
        output_dir: Directory into which per-model JSON results are written.
        test_set: Cached list of gold annotations.
        seed: RNG seed for reproducible bootstrap resampling.
    """

    def __init__(
        self,
        test_set_path: str | Path,
        output_dir: str | Path,
        seed: int = 42,
    ) -> None:
        """Initialize the evaluator.

        Args:
            test_set_path: Path to the test set file.
            output_dir: Directory for result JSONs (created if missing).
            seed: Seed used for numpy RNG in bootstrap sampling.
        """
        self.test_set_path = Path(test_set_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_set: list[dict[str, Any]] = _load_test_set(self.test_set_path)
        self.seed = seed

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, model_tag: str, pipeline: Any) -> dict[str, Any]:
        """Run a pipeline over the full test set and score every query.

        ``pipeline`` must be callable as ``pipeline(query)`` and return
        either a ``QueryResponse`` or a dict coercible to one.

        Args:
            model_tag: Identifier used when persisting the result JSON.
            pipeline: A callable of signature ``(query: str) -> QueryResponse``.

        Returns:
            A dict with keys ``model_tag``, ``per_query`` (list of per-query
            metric dicts), and ``aggregate`` (dict of metric-name -> float).
        """
        per_query: list[dict[str, Any]] = []
        pred_escalated: list[bool] = []
        gold_escalated: list[bool] = []
        pred_decomp: list[bool] = []
        gold_decomp: list[bool] = []

        n_total = len(self.test_set)
        for idx, item in enumerate(self.test_set):
            query = item["query"]
            logger.info("[%s] query %d/%d: %s", model_tag, idx + 1, n_total, query[:60])
            try:
                raw = pipeline(query)
                response = _to_query_response(raw)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Pipeline failed on query %d: %s", idx, exc)
                response = QueryResponse(answer="", model_tag=model_tag)

            # Field-name normalisation: test set uses "answer_with_citations"
            # and "citations"; fall back to legacy "gold_answer"/"gold_citations".
            raw_gold_answer = item.get("gold_answer") or item.get("answer_with_citations", "")
            # Strip inline citation markers [hex:start-end] before scoring.
            import re as _re
            gold_answer = _re.sub(r"\[[0-9a-f]{8,}:\d+-\d+\]", "", raw_gold_answer).strip()
            gold_citations = item.get("citations", item.get("gold_citations", []))
            needed_tool = item.get("needed_tool")
            gold_escalate = bool(item.get("should_escalate", False))

            row: dict[str, Any] = {
                "query": query,
                "model_tag": model_tag,
                "answer": response.answer,
                "latency_ms": response.latency_ms,
                "ttft_ms": response.ttft_ms,
                "cgal_iterations": response.cgal_iterations,
                "confidence": response.confidence,
            }

            row["grounding"] = grounding_score(response.answer, response.citations)
            row["citation_f1"] = citation_f1(response.citations, gold_citations)

            gold_chunk_ids = item.get("gold_chunk_ids", [])
            if gold_chunk_ids:
                # Try chunk_id first (M5 path); fall back to doc_id comparison
                # for baseline pipelines that only populate doc_id on citations.
                # IMPORTANT: when falling back to doc_ids, compare against gold
                # doc_ids (not gold chunk_ids) to avoid a doc_id vs chunk_id
                # format mismatch that would always produce recall = 0.
                chunk_ids = [c.chunk_id for c in response.citations if c.chunk_id]
                if chunk_ids and any(rid in gold_chunk_ids for rid in chunk_ids):
                    row["recall_at_20"] = recall_at_k(chunk_ids, gold_chunk_ids, k=20)
                else:
                    gold_doc_ids_for_recall = [
                        c["doc_id"]
                        for c in gold_citations
                        if isinstance(c, dict) and c.get("doc_id")
                    ]
                    if gold_doc_ids_for_recall:
                        retrieved_doc_ids = [
                            c.doc_id for c in response.citations if c.doc_id
                        ]
                        row["recall_at_20"] = recall_at_k(
                            retrieved_doc_ids, gold_doc_ids_for_recall, k=20
                        )
                    else:
                        row["recall_at_20"] = float("nan")
            else:
                # Fall back: use doc_ids from citations against gold citation doc_ids.
                gold_doc_ids_for_recall = [
                    c["doc_id"] for c in gold_citations
                    if isinstance(c, dict) and c.get("doc_id")
                ]
                if gold_doc_ids_for_recall:
                    retrieved_doc_ids = [c.doc_id for c in response.citations]
                    row["recall_at_20"] = recall_at_k(
                        retrieved_doc_ids, gold_doc_ids_for_recall, k=20
                    )
                else:
                    row["recall_at_20"] = float("nan")

            if gold_answer:
                try:
                    row["bertscore_f1"] = answer_quality(response.answer, gold_answer)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("BERTScore failed on query %d: %s", idx, exc)
                    row["bertscore_f1"] = float("nan")
            else:
                row["bertscore_f1"] = float("nan")

            if needed_tool:
                row["tool_accuracy"] = tool_accuracy(response.tool_calls, needed_tool)
            elif response.tool_calls:
                # No tool was required but the model called one anyway — false positive.
                row["tool_accuracy"] = {"name_match": 0.0, "arg_f1": 0.0}
            else:
                # Correct: no tool needed and none was called.
                row["tool_accuracy"] = {"name_match": 1.0, "arg_f1": 1.0}

            # Enrich gold dict with fields FCRS needs but test set omits.
            gold_doc_ids = [
                c["doc_id"] for c in gold_citations
                if isinstance(c, dict) and c.get("doc_id")
            ]
            # Build key_points from gold answer sentences (strip citation markers).
            gold_sentences = [
                s.strip() for s in _re.split(r"(?<=[.!?])\s+", gold_answer) if s.strip()
            ]
            fcrs_gold = dict(item)
            fcrs_gold.setdefault("doc_ids", gold_doc_ids)
            fcrs_gold.setdefault("key_points", gold_sentences)

            try:
                row["fcrs"] = compute_fcrs(response, fcrs_gold)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FCRS failed on query %d: %s", idx, exc)
                row["fcrs"] = {
                    "fcrs": float("nan"),
                    "completeness": float("nan"),
                    "citation_coverage": float("nan"),
                    "tool_appropriateness": float("nan"),
                    "escalation_accuracy": float("nan"),
                }

            pred_escalated.append(response.ticket_id is not None)
            gold_escalated.append(gold_escalate)
            if "is_multi_part" in item:
                pred_decomp.append(bool(response.decomposed))
                gold_decomp.append(bool(item["is_multi_part"]))

            per_query.append(row)

        # Aggregate
        aggregate: dict[str, float] = {}
        aggregate["grounding"] = self._nan_mean([r["grounding"] for r in per_query])
        aggregate["citation_f1"] = self._nan_mean(
            [r["citation_f1"]["f1"] for r in per_query]
        )
        aggregate["citation_precision"] = self._nan_mean(
            [r["citation_f1"]["precision"] for r in per_query]
        )
        aggregate["citation_recall"] = self._nan_mean(
            [r["citation_f1"]["recall"] for r in per_query]
        )
        aggregate["bertscore_f1"] = self._nan_mean(
            [r["bertscore_f1"] for r in per_query]
        )
        aggregate["recall_at_20"] = self._nan_mean(
            [r["recall_at_20"] for r in per_query]
        )
        aggregate["tool_name_match"] = self._nan_mean(
            [r["tool_accuracy"]["name_match"] for r in per_query]
        )
        aggregate["tool_arg_f1"] = self._nan_mean(
            [r["tool_accuracy"]["arg_f1"] for r in per_query]
        )
        aggregate["fcrs"] = self._nan_mean([r["fcrs"]["fcrs"] for r in per_query])
        aggregate["completeness"] = self._nan_mean(
            [r["fcrs"]["completeness"] for r in per_query]
        )
        aggregate["citation_coverage"] = self._nan_mean(
            [r["fcrs"]["citation_coverage"] for r in per_query]
        )
        aggregate["tool_appropriateness"] = self._nan_mean(
            [r["fcrs"]["tool_appropriateness"] for r in per_query]
        )
        aggregate["escalation_accuracy_per_query"] = self._nan_mean(
            [r["fcrs"]["escalation_accuracy"] for r in per_query]
        )

        latencies = [r["latency_ms"] for r in per_query if r["latency_ms"] is not None]
        if latencies:
            aggregate["latency_p50"] = float(np.percentile(latencies, 50))
            aggregate["latency_p95"] = float(np.percentile(latencies, 95))
            aggregate["latency_mean"] = float(np.mean(latencies))

        ttfts = [r["ttft_ms"] for r in per_query if r["ttft_ms"] is not None]
        if ttfts:
            aggregate["ttft_p50"] = float(np.percentile(ttfts, 50))
            aggregate["ttft_p95"] = float(np.percentile(ttfts, 95))

        # Responses for CGAL efficiency (reconstruct minimal QueryResponse).
        dummy_responses = [
            QueryResponse(
                answer=r["answer"], cgal_iterations=int(r["cgal_iterations"])
            )
            for r in per_query
        ]
        aggregate["cgal_efficiency"] = cgal_efficiency(dummy_responses)

        if pred_escalated:
            aggregate["escalation_f1"] = escalation_f1(
                pred_escalated, gold_escalated
            )["f1"]
        if pred_decomp:
            aggregate["decomposition_f1"] = decomposition_accuracy(
                pred_decomp, gold_decomp
            )["f1"]

        result = {
            "model_tag": model_tag,
            "n_queries": len(per_query),
            "per_query": per_query,
            "aggregate": aggregate,
        }
        self.save_results(result, self.output_dir / f"{model_tag}.json")
        return result

    def evaluate_all(
        self, pipelines: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Evaluate multiple pipelines and return a tag -> aggregate map.

        Args:
            pipelines: Dict mapping model-tag strings to callable pipelines.

        Returns:
            Dict ``{model_tag: aggregate_metrics_dict}``.
        """
        out: dict[str, dict[str, float]] = {}
        full_results: dict[str, dict[str, Any]] = {}
        for tag, pipe in pipelines.items():
            logger.info("Evaluating model: %s", tag)
            res = self.evaluate(tag, pipe)
            out[tag] = res["aggregate"]
            full_results[tag] = res
        # Persist the combined summary.
        self.save_results(
            {"summary": out}, self.output_dir / "summary.json"
        )
        return out

    # ------------------------------------------------------------------
    # Statistical testing
    # ------------------------------------------------------------------

    def bootstrap_significance(
        self,
        metric_values_a: Iterable[float],
        metric_values_b: Iterable[float],
        n: int = 1000,
    ) -> float:
        """Paired-bootstrap two-sided p-value for mean(a) - mean(b).

        Args:
            metric_values_a: Per-query metric values for model A.
            metric_values_b: Per-query metric values for model B
                (same length and ordering as A).
            n: Number of bootstrap resamples.

        Returns:
            Two-sided p-value for H0: mean(a) == mean(b).
        """
        a = np.asarray(list(metric_values_a), dtype=float)
        b = np.asarray(list(metric_values_b), dtype=float)
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: a={a.shape} b={b.shape}")
        if a.size == 0:
            return 1.0

        # Drop NaNs pairwise.
        mask = ~(np.isnan(a) | np.isnan(b))
        a = a[mask]
        b = b[mask]
        if a.size == 0:
            return 1.0

        diff = a - b
        observed = float(diff.mean())
        centered = diff - observed

        rng = np.random.default_rng(self.seed)
        n_samples = a.size
        count = 0
        for _ in range(n):
            idx = rng.integers(0, n_samples, size=n_samples)
            resample_mean = float(centered[idx].mean())
            if abs(resample_mean) >= abs(observed):
                count += 1
        return count / n

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, results: dict[str, Any], path: Path) -> None:
        """Serialize a results dict to JSON.

        Args:
            results: Results dictionary.
            path: Destination file path (parent dirs created).

        Returns:
            None.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=_json_default)

    # ------------------------------------------------------------------
    # Ablations
    # ------------------------------------------------------------------

    def ablation_study(
        self,
        full_pipeline: Any,
        components_to_ablate: list[str],
    ) -> dict[str, dict[str, float]]:
        """Disable each named component and measure FCRS delta vs full.

        The ``full_pipeline`` is expected to expose ``toggle(name, enabled)``
        or an ``ablate(name)`` context manager. If neither exists, this
        method falls back to setting ``pipeline.<name>_enabled = False``
        via ``copy.copy`` (shallow).

        Args:
            full_pipeline: The unablated pipeline callable/object.
            components_to_ablate: List of component flag names to disable
                one at a time (e.g. ``["decomposition", "reranker"]``).

        Returns:
            Dict mapping component-name -> {``fcrs``, ``delta`` (vs full)}.
        """
        baseline_res = self.evaluate("full", full_pipeline)
        baseline_fcrs = baseline_res["aggregate"].get("fcrs", float("nan"))

        out: dict[str, dict[str, float]] = {
            "full": {"fcrs": baseline_fcrs, "delta": 0.0}
        }

        for comp in components_to_ablate:
            ablated = _ablate_pipeline(full_pipeline, comp)
            tag = f"ablate_{comp}"
            res = self.evaluate(tag, ablated)
            fcrs_val = res["aggregate"].get("fcrs", float("nan"))
            delta = fcrs_val - baseline_fcrs if not np.isnan(baseline_fcrs) else float("nan")
            out[comp] = {"fcrs": float(fcrs_val), "delta": float(delta)}

        self.save_results(
            {"ablation": out}, self.output_dir / "ablation.json"
        )
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nan_mean(values: Iterable[float]) -> float:
        """Return mean of values, ignoring NaNs.

        Args:
            values: Iterable of floats.

        Returns:
            Mean of non-NaN values, or NaN if all values are NaN.
        """
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            return float("nan")
        mask = ~np.isnan(arr)
        if not mask.any():
            return float("nan")
        return float(arr[mask].mean())


# ----------------------------------------------------------------------
# Free-standing helpers
# ----------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON default for non-serializable objects (Paths, numpy scalars).

    Args:
        obj: Object to serialize.

    Returns:
        A JSON-compatible representation.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _ablate_pipeline(pipeline: Any, component: str) -> Callable[[str], QueryResponse]:
    """Return a wrapped pipeline with one component disabled.

    Tries in order: ``pipeline.toggle(component, False)`` via shallow copy,
    then setting ``pipeline.<component>_enabled = False``, then finally
    returns the original pipeline unchanged (logging a warning).

    Args:
        pipeline: The unablated pipeline object/callable.
        component: Name of the component to disable.

    Returns:
        A callable pipeline (possibly the same object) with the named
        component disabled.
    """
    try:
        ablated = copy.copy(pipeline)
    except TypeError:
        ablated = pipeline

    flag_name = f"{component}_enabled"
    if hasattr(ablated, "toggle") and callable(getattr(ablated, "toggle")):
        try:
            ablated.toggle(component, False)
            return ablated
        except Exception as exc:  # noqa: BLE001
            logger.warning("toggle(%s, False) failed: %s", component, exc)

    if hasattr(ablated, flag_name):
        try:
            setattr(ablated, flag_name, False)
            return ablated
        except Exception as exc:  # noqa: BLE001
            logger.warning("setattr %s failed: %s", flag_name, exc)

    logger.warning(
        "Could not ablate component '%s'; running pipeline unchanged", component
    )
    return ablated
