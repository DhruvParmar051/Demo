"""
AegisRAG - Report generation.

Reads a results JSON (produced by :class:`Evaluator`) and writes:

* ``report.md`` -- human-readable Markdown comparison table.
* ``results_table.tex`` -- LaTeX table for the final paper.
* ``radar_<tag>.png`` -- one radar plot per model.
* ``latency_violin.png`` -- latency distribution violin plot.
* ``reliability.png`` -- calibration reliability diagram (if confidences
  and accuracies are present in the results file).
* ``summary.json`` -- distilled summary with bootstrap significance flags.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.calibration import reliability_diagram

logger = logging.getLogger(__name__)


# Metrics to include in the main comparison table / radar chart.
_CORE_METRICS: list[str] = [
    "fcrs",
    "grounding",
    "citation_f1",
    "bertscore_f1",
    "recall_at_20",
    "completeness",
    "citation_coverage",
    "tool_appropriateness",
    "tool_name_match",
    "cgal_efficiency",
    "latency_p50",
    "latency_p95",
]


_RADAR_METRICS: list[str] = [
    "fcrs",
    "grounding",
    "citation_f1",
    "bertscore_f1",
    "completeness",
    "tool_appropriateness",
]


def _load_results(results_path: Path) -> dict[str, Any]:
    """Load the full results JSON produced by the Evaluator.

    The file may be either a single-model result or a dict ``{"summary": ...}``
    produced by ``evaluate_all``. This loader normalizes both forms into
    ``{tag: {"aggregate": {...}, "per_query": [...]}}``.

    Args:
        results_path: Path to a JSON produced by the Evaluator.

    Returns:
        Normalized dict keyed by model tag. Each value contains at least
        ``aggregate`` and optionally ``per_query``.
    """
    results_path = Path(results_path)
    with open(results_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # If this is a summary-only file, load individual model JSONs alongside.
    out: dict[str, dict[str, Any]] = {}
    if "summary" in raw and isinstance(raw["summary"], dict):
        for tag, agg in raw["summary"].items():
            per_path = results_path.parent / f"{tag}.json"
            if per_path.exists():
                with open(per_path, "r", encoding="utf-8") as fh:
                    out[tag] = json.load(fh)
            else:
                out[tag] = {"model_tag": tag, "aggregate": agg, "per_query": []}
    elif "model_tag" in raw:
        out[raw["model_tag"]] = raw
    else:
        # Assume raw is already {tag: result-dict} OR {tag: flat-aggregate-dict}.
        # evaluate_all.py writes all_results.json as {tag: flat-aggregate-dict}
        # (not wrapped under "aggregate"). Normalise both forms and sideload the
        # per-model JSON (which contains the full per_query list) when available.
        for tag, v in raw.items():
            if not isinstance(v, dict):
                continue
            per_path = results_path.parent / f"{tag}.json"
            if per_path.exists():
                with open(per_path, "r", encoding="utf-8") as fh:
                    out[tag] = json.load(fh)
            elif "aggregate" in v:
                out[tag] = v
            else:
                # Flat aggregate dict — wrap it so downstream consumers work.
                out[tag] = {"model_tag": tag, "aggregate": v, "per_query": []}
    return out


def _fmt(v: Any) -> str:
    """Format a metric value for table display.

    Args:
        v: The value (int, float, or other).

    Returns:
        A short string representation.
    """
    if v is None:
        return "-"
    if isinstance(v, float):
        if np.isnan(v):
            return "-"
        return f"{v:.4f}"
    return str(v)


def _write_markdown_table(
    results: dict[str, dict[str, Any]],
    metrics: list[str],
    path: Path,
) -> None:
    """Write a Markdown comparison table.

    Args:
        results: Normalized ``{tag: result-dict}`` mapping.
        metrics: Metric names (columns).
        path: Output ``.md`` path.

    Returns:
        None.
    """
    tags = list(results.keys())
    header = "| model | " + " | ".join(metrics) + " |"
    sep = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"
    lines = ["# AegisRAG Evaluation Report", "", header, sep]
    for tag in tags:
        agg = results[tag].get("aggregate", {})
        row = [tag] + [_fmt(agg.get(m)) for m in metrics]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _write_latex_table(
    results: dict[str, dict[str, Any]],
    metrics: list[str],
    path: Path,
) -> None:
    """Write a LaTeX booktabs-style table.

    Args:
        results: Normalized ``{tag: result-dict}`` mapping.
        metrics: Metric names (columns).
        path: Output ``.tex`` path.

    Returns:
        None.
    """
    col_spec = "l" + "r" * len(metrics)
    header = " & ".join(["Model"] + [_latex_safe(m) for m in metrics]) + " \\\\"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{AegisRAG evaluation results across baselines and improved"
        " models. Higher is better except for latency and CGAL efficiency.}",
        "\\label{tab:aegis-results}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for tag, res in results.items():
        agg = res.get("aggregate", {})
        row = [tag] + [_fmt(agg.get(m)) for m in metrics]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _latex_safe(text: str) -> str:
    """Escape underscores for LaTeX.

    Args:
        text: Raw text.

    Returns:
        Text with underscores escaped.
    """
    return text.replace("_", "\\_")


def _radar_chart(
    results: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    """Write one radar chart per model using matplotlib.

    Args:
        results: Normalized ``{tag: result-dict}`` mapping.
        output_dir: Directory in which to save the PNGs.

    Returns:
        None.
    """
    import matplotlib  # lazy

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = _RADAR_METRICS
    n_axes = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    for tag, res in results.items():
        agg = res.get("aggregate", {})
        values = []
        for m in metrics:
            v = agg.get(m, 0.0)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 0.0
            values.append(max(0.0, min(1.0, float(v))))
        values += values[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
        ax.plot(angles, values, linewidth=2, color="steelblue")
        ax.fill(angles, values, alpha=0.25, color="steelblue")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Model: {tag}", pad=14)
        fig.tight_layout()
        fig.savefig(output_dir / f"radar_{tag}.png", dpi=150)
        plt.close(fig)


def _latency_violin(
    results: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    """Violin plot of per-query latency distributions, one per model.

    Args:
        results: Normalized ``{tag: result-dict}`` mapping.
        output_dir: Directory to save ``latency_violin.png``.

    Returns:
        None.
    """
    import matplotlib  # lazy

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data: list[list[float]] = []
    labels: list[str] = []
    for tag, res in results.items():
        per_q = res.get("per_query", [])
        lat = [
            float(q.get("latency_ms", 0.0))
            for q in per_q
            if q.get("latency_ms") is not None
        ]
        if lat:
            data.append(lat)
            labels.append(tag)
    if not data:
        logger.info("No latency data; skipping violin plot.")
        return

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(data)), 5))
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    bodies = parts.get("bodies", []) if isinstance(parts, dict) else []
    for pc in list(bodies):  # type: ignore[arg-type]
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-query latency distribution by model")
    fig.tight_layout()
    fig.savefig(output_dir / "latency_violin.png", dpi=150)
    plt.close(fig)


def _maybe_reliability(
    results: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    """If per-query confidences + correctness are present, plot reliability.

    Args:
        results: Normalized ``{tag: result-dict}`` mapping.
        output_dir: Directory to save ``reliability_<tag>.png``.

    Returns:
        None.
    """
    for tag, res in results.items():
        per_q = res.get("per_query", [])
        confs: list[float] = []
        accs: list[float] = []
        for q in per_q:
            c = q.get("confidence")
            bs = q.get("bertscore_f1")
            if c is None or bs is None:
                continue
            if isinstance(bs, float) and np.isnan(bs):
                continue
            confs.append(float(c))
            accs.append(1.0 if float(bs) >= 0.70 else 0.0)
        if len(confs) < 10:
            continue
        reliability_diagram(
            np.array(confs),
            np.array(accs),
            output_dir / f"reliability_{tag}.png",
            title=f"Reliability ({tag})",
        )


def _bootstrap_flags(
    results: dict[str, dict[str, Any]], metric: str = "fcrs", n: int = 1000
) -> dict[str, float]:
    """Compute paired-bootstrap p-values for each model vs the alphabetically
    first model, on the per-query FCRS scalar.

    Args:
        results: Normalized ``{tag: result-dict}`` mapping.
        metric: Per-query metric key (looked up inside the per-query FCRS
            dict).
        n: Bootstrap sample count.

    Returns:
        Dict ``{tag: p_value}`` for non-reference tags.
    """
    tags = list(results.keys())
    if len(tags) < 2:
        return {}
    ref = tags[0]
    ref_per = results[ref].get("per_query", [])
    ref_vals = np.array(
        [q.get("fcrs", {}).get(metric, np.nan) for q in ref_per], dtype=float
    )

    out: dict[str, float] = {}
    rng = np.random.default_rng(42)
    for tag in tags[1:]:
        other_per = results[tag].get("per_query", [])
        if len(other_per) != len(ref_per):
            continue
        other_vals = np.array(
            [q.get("fcrs", {}).get(metric, np.nan) for q in other_per], dtype=float
        )
        mask = ~(np.isnan(ref_vals) | np.isnan(other_vals))
        a = other_vals[mask]
        b = ref_vals[mask]
        if a.size == 0:
            out[tag] = 1.0
            continue
        diff = a - b
        observed = float(diff.mean())
        centered = diff - observed
        count = 0
        for _ in range(n):
            idx = rng.integers(0, a.size, size=a.size)
            if abs(float(centered[idx].mean())) >= abs(observed):
                count += 1
        out[tag] = count / n
    return out


def generate_report(results_path: Path, output_dir: Path) -> None:
    """Generate all report artifacts from a single results JSON.

    Args:
        results_path: Path to the JSON file produced by the Evaluator
            (single-model or ``summary.json`` form).
        output_dir: Directory to write the Markdown/LaTeX/PNG artifacts.

    Returns:
        None.
    """
    results_path = Path(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(results_path)
    if not results:
        logger.warning("No results found in %s", results_path)
        return

    _write_markdown_table(results, _CORE_METRICS, output_dir / "report.md")
    _write_latex_table(results, _CORE_METRICS, output_dir / "results_table.tex")
    _radar_chart(results, output_dir)
    _latency_violin(results, output_dir)
    _maybe_reliability(results, output_dir)

    # Distilled summary with bootstrap p-values.
    summary = {
        tag: res.get("aggregate", {}) for tag, res in results.items()
    }
    p_values = _bootstrap_flags(results)
    summary_out = {
        "per_model": summary,
        "bootstrap_p_values_vs_reference": p_values,
        "significance_flag_alpha_0.05": {
            tag: (p is not None and p < 0.05) for tag, p in p_values.items()
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_out, fh, indent=2)
