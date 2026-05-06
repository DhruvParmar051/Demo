"""
AegisRAG - Evaluation metric functions.

Individual metric functions used across baselines and improved models.
All heavy imports (bert_score, torch) are lazy so importing this module
remains cheap even in restricted environments.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

from src.data.schema import Citation, QueryResponse, ToolCall


# ----------------------------------------------------------------------
# Token normalization helpers
# ----------------------------------------------------------------------


_ALNUM_RE = re.compile(r"[a-z0-9]+")


def _normalize_tokens(text: str) -> list[str]:
    """Lowercase text and extract alphanumeric tokens.

    Args:
        text: Raw input string.

    Returns:
        List of lowercase alphanumeric-only tokens.
    """
    if not text:
        return []
    return _ALNUM_RE.findall(text.lower())


# ----------------------------------------------------------------------
# Retrieval recall
# ----------------------------------------------------------------------


def recall_at_k(
    retrieved_ids: list[str],
    gold_ids: list[str],
    k: int = 20,
) -> float:
    """Fraction of gold chunk IDs found in the top-k retrieved chunks.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (ranked best-first).
        gold_ids: Gold chunk IDs that should be retrieved.
        k: Cutoff depth.

    Returns:
        Recall@k in [0, 1]. Returns 0.0 if ``gold_ids`` is empty.
    """
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(gold_ids)) / len(gold_ids)


# ----------------------------------------------------------------------
# Grounding
# ----------------------------------------------------------------------


def grounding_score(answer: str, cited_spans: list[Citation]) -> float:
    """Fraction of answer tokens supported by the concatenated cited text.

    Tokens are normalized (lowercased, alphanumeric only) before matching.

    Args:
        answer: The generated answer string.
        cited_spans: List of Citation objects whose ``cited_text`` field is
            concatenated to form the support corpus.

    Returns:
        A float in [0, 1] giving the fraction of answer tokens present in
        the cited support corpus. Returns 0.0 if the answer has no tokens.
    """
    answer_tokens = _normalize_tokens(answer)
    if not answer_tokens:
        return 0.0

    support_text = " ".join(c.cited_text or "" for c in cited_spans)
    support_tokens = set(_normalize_tokens(support_text))
    if not support_tokens:
        return 0.0

    matched = sum(1 for t in answer_tokens if t in support_tokens)
    return matched / len(answer_tokens)


# ----------------------------------------------------------------------
# Citation F1
# ----------------------------------------------------------------------


def _span_overlap(
    a_start: int, a_end: int, b_start: int, b_end: int
) -> float:
    """Compute IoU-style overlap between two spans.

    Args:
        a_start: Start of span A.
        a_end: End of span A (exclusive).
        b_start: Start of span B.
        b_end: End of span B (exclusive).

    Returns:
        Overlap ratio (intersection / union) in [0, 1]. Returns 0.0 if
        either span is degenerate (start >= end).
    """
    if a_end <= a_start or b_end <= b_start:
        return 0.0
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def citation_f1(
    pred: list[Citation], gold: list[dict[str, Any]]
) -> dict[str, float]:
    """Compute precision / recall / F1 for predicted citations.

    A predicted citation is counted as correct if there exists a gold
    citation with the same ``doc_id`` and span IoU overlap >= 0.5. Each
    gold citation can be matched at most once (greedy best-overlap match).

    Args:
        pred: List of predicted Citation objects.
        gold: List of gold citation dicts with keys
            ``doc_id``, ``span_start``, ``span_end``.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``.
    """
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gold: set[int] = set()
    tp = 0
    for p in pred:
        best_idx = -1
        best_overlap = 0.0
        pred_degenerate = p.span_start >= p.span_end
        for g_idx, g in enumerate(gold):
            if g_idx in matched_gold:
                continue
            if g.get("doc_id") != p.doc_id:
                continue
            if pred_degenerate:
                # Predicted span is degenerate (e.g. 0-0 from inline markers);
                # doc_id match is sufficient — treat as full overlap.
                ov = 1.0
            else:
                ov = _span_overlap(
                    p.span_start,
                    p.span_end,
                    int(g.get("span_start", 0)),
                    int(g.get("span_end", 0)),
                )
            if ov > best_overlap:
                best_overlap = ov
                best_idx = g_idx
        if best_idx >= 0 and best_overlap >= 0.5:
            matched_gold.add(best_idx)
            tp += 1

    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ----------------------------------------------------------------------
# Answer quality (BERTScore)
# ----------------------------------------------------------------------


_BERTSCORE_CACHE: dict[str, Any] = {}


def _get_bertscorer() -> Any:
    """Return a cached BERTScorer instance (roberta-large loads once per process)."""
    if "scorer" not in _BERTSCORE_CACHE:
        from bert_score import BERTScorer
        import inspect as _inspect
        import logging
        # Suppress bert_score's verbose load reports (UNEXPECTED/MISSING key tables).
        for _noisy in ("bert_score", "transformers.modeling_utils"):
            logging.getLogger(_noisy).setLevel(logging.ERROR)
        _sig = _inspect.signature(BERTScorer.__init__)
        _kwargs: dict[str, Any] = {"lang": "en", "rescale_with_baseline": False}
        if "verbose" in _sig.parameters:
            _kwargs["verbose"] = False
        _BERTSCORE_CACHE["scorer"] = BERTScorer(**_kwargs)
    return _BERTSCORE_CACHE["scorer"]


def answer_quality(pred: str, gold: str) -> float:
    """BERTScore F1 between a predicted answer and a gold reference.

    Args:
        pred: Predicted answer string.
        gold: Gold reference answer string.

    Returns:
        BERTScore F1 in [0, 1]. Returns 0.0 if either input is empty.
    """
    if not pred or not gold:
        return 0.0
    scorer = _get_bertscorer()
    _, _, f1 = scorer.score(
        [pred],
        [gold],
        verbose=False,
    )
    return float(f1[0].item())


# ----------------------------------------------------------------------
# Tool accuracy
# ----------------------------------------------------------------------


def _arg_field_f1(pred_args: dict[str, Any], gold_args: dict[str, Any]) -> float:
    """Compute field-level F1 between two argument dicts.

    Args:
        pred_args: Predicted arguments dict.
        gold_args: Gold arguments dict.

    Returns:
        F1 score over (key, normalized-value) pairs.
    """
    def _norm(v: Any) -> str:
        return str(v).strip().lower()

    pred_items = {(k, _norm(v)) for k, v in (pred_args or {}).items()}
    gold_items = {(k, _norm(v)) for k, v in (gold_args or {}).items()}
    if not pred_items and not gold_items:
        return 1.0
    if not pred_items or not gold_items:
        return 0.0
    tp = len(pred_items & gold_items)
    precision = tp / len(pred_items)
    recall = tp / len(gold_items)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def tool_accuracy(
    pred_tool_calls: list[ToolCall], gold_tool: str
) -> dict[str, float]:
    """Evaluate tool-call correctness.

    Computes exact-name match (does the predicted sequence contain a call
    to ``gold_tool``?) and an argument-level F1 (over fields) for the
    matching call. If no prediction invokes ``gold_tool``, both metrics
    are 0.0.

    Args:
        pred_tool_calls: List of predicted ToolCall objects.
        gold_tool: Name of the expected tool.

    Returns:
        Dict with keys ``name_match``, ``arg_f1``.
    """
    if not gold_tool:
        return {"name_match": 1.0, "arg_f1": 1.0}

    matching = [tc for tc in pred_tool_calls if tc.tool_name == gold_tool]
    name_match = 1.0 if matching else 0.0
    if not matching:
        return {"name_match": 0.0, "arg_f1": 0.0}

    # Use the last matching call's args (most recent iteration).
    arg_f1 = _arg_field_f1(matching[-1].args or {}, {})
    # If no gold args supplied, treat name match as sufficient.
    return {"name_match": name_match, "arg_f1": arg_f1}


# ----------------------------------------------------------------------
# Escalation F1
# ----------------------------------------------------------------------


def escalation_f1(
    pred_escalated: list[bool], gold_escalated: list[bool]
) -> dict[str, float]:
    """Compute precision / recall / F1 for binary escalation decisions.

    Args:
        pred_escalated: Predicted escalation flags.
        gold_escalated: Gold escalation flags.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``.
    """
    if len(pred_escalated) != len(gold_escalated):
        raise ValueError(
            f"Length mismatch: pred={len(pred_escalated)} "
            f"gold={len(gold_escalated)}"
        )
    tp = sum(1 for p, g in zip(pred_escalated, gold_escalated) if p and g)
    fp = sum(1 for p, g in zip(pred_escalated, gold_escalated) if p and not g)
    fn = sum(1 for p, g in zip(pred_escalated, gold_escalated) if not p and g)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ----------------------------------------------------------------------
# Consistency
# ----------------------------------------------------------------------


def consistency(responses: list[QueryResponse]) -> float:
    """Fraction of repeated-run answers that are identical to the mode answer.

    Given multiple responses to the same query, return the fraction of
    those responses whose (normalized) answer equals the most common
    answer in the batch.

    Args:
        responses: List of QueryResponse objects from repeated runs.

    Returns:
        Consistency score in [0, 1]. Returns 1.0 for a single response.
    """
    if not responses:
        return 0.0
    if len(responses) == 1:
        return 1.0
    normalized = [" ".join(_normalize_tokens(r.answer)) for r in responses]
    counts = Counter(normalized)
    _, top_n = counts.most_common(1)[0]
    return top_n / len(responses)


# ----------------------------------------------------------------------
# CGAL efficiency
# ----------------------------------------------------------------------


def cgal_efficiency(responses: list[QueryResponse]) -> float:
    """Mean number of CGAL iterations per response.

    Lower is better: fewer iterations means higher confidence on the
    first pass.

    Args:
        responses: List of QueryResponse objects.

    Returns:
        Mean of ``cgal_iterations`` across responses. Returns 0.0 for an
        empty list.
    """
    if not responses:
        return 0.0
    return float(np.mean([r.cgal_iterations for r in responses]))


# ----------------------------------------------------------------------
# Decomposition accuracy
# ----------------------------------------------------------------------


def decomposition_accuracy(
    pred: list[bool], gold: list[bool]
) -> dict[str, float]:
    """Compute P/R/F1 for multi-part query detection.

    Args:
        pred: Predicted ``is_multi_part`` flags.
        gold: Gold ``is_multi_part`` flags.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``.
    """
    if len(pred) != len(gold):
        raise ValueError(
            f"Length mismatch: pred={len(pred)} gold={len(gold)}"
        )
    tp = sum(1 for p, g in zip(pred, gold) if p and g)
    fp = sum(1 for p, g in zip(pred, gold) if p and not g)
    fn = sum(1 for p, g in zip(pred, gold) if not p and g)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}
