"""
AegisRAG - First-Contact Resolution Score (FCRS).

FCRS is AegisRAG's novel composite metric for customer-support RAG
systems. It combines answer completeness (vs gold key points), citation
coverage of factual claims, tool appropriateness, and escalation
accuracy into a single [0, 1] score.

FCRS = 0.35 * completeness
     + 0.25 * citation_coverage
     + 0.20 * tool_appropriateness
     + 0.20 * escalation_accuracy
"""

from __future__ import annotations

import re
from typing import Any

from src.data.schema import QueryResponse


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(r"\d")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]+\b")


# Cached BERTScore callable
_BERTSCORE_CACHE: dict[str, Any] = {}


def _bertscore_fn() -> Any:
    """Lazy-load ``bert_score.score``.

    Returns:
        The ``bert_score.score`` function.
    """
    if "score" not in _BERTSCORE_CACHE:
        from bert_score import score as _score  # lazy import

        _BERTSCORE_CACHE["score"] = _score
    return _BERTSCORE_CACHE["score"]


def _completeness(answer: str, key_points: list[str]) -> float:
    """Average max-BERTScore F1 of the answer vs each gold key point.

    Args:
        answer: Predicted answer string.
        key_points: List of gold key-point strings the answer should cover.

    Returns:
        Mean of per-keypoint BERTScore F1 values in [0, 1]. Returns 1.0 if
        there are no key points; 0.0 if the answer is empty.
    """
    if not key_points:
        return 1.0
    if not answer:
        return 0.0
    score = _bertscore_fn()
    preds = [answer] * len(key_points)
    refs = list(key_points)
    _, _, f1 = score(
        preds, refs, lang="en", rescale_with_baseline=False, verbose=False
    )
    vals = [float(x.item()) for x in f1]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _is_factual_sentence(sentence: str) -> bool:
    """Heuristic: a sentence is factual if it contains a number or proper noun.

    Args:
        sentence: A single sentence string.

    Returns:
        True when the sentence contains a digit or a capitalized token
        (other than a single leading word), else False.
    """
    if not sentence or not sentence.strip():
        return False
    if _NUMBER_RE.search(sentence):
        return True
    # Strip the first token (often sentence-initial capital) before
    # searching for proper nouns.
    tokens = sentence.strip().split()
    tail = " ".join(tokens[1:]) if len(tokens) > 1 else ""
    if tail and _PROPER_NOUN_RE.search(tail):
        return True
    return False


def _citation_coverage(
    answer: str, response: QueryResponse, gold_doc_ids: set[str]
) -> float:
    """Fraction of factual sentences with a valid supporting citation.

    A sentence is "covered" if at least one of ``response.citations`` has
    a ``doc_id`` present in ``gold_doc_ids``.

    Args:
        answer: Predicted answer string.
        response: The full QueryResponse (for its citation list).
        gold_doc_ids: Set of doc_ids considered valid support.

    Returns:
        Coverage fraction in [0, 1]. Returns 1.0 if no factual sentences
        are detected.
    """
    if not answer:
        return 0.0
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()]
    factual = [s for s in sentences if _is_factual_sentence(s)]
    if not factual:
        return 1.0
    pred_doc_ids = {c.doc_id for c in response.citations}
    valid_citation = bool(pred_doc_ids & gold_doc_ids)
    # All factual sentences share the same citation pool; either the
    # response has a valid citation or it does not.
    return 1.0 if valid_citation else 0.0


def _tool_appropriateness(
    response: QueryResponse, needed_tool: str | None
) -> float:
    """Score tool usage: 1.0 correct, 0.5 unnecessary, 0.0 missing.

    Args:
        response: The QueryResponse whose tool_calls are inspected.
        needed_tool: Name of the tool the query required, or None if
            no tool was needed.

    Returns:
        1.0 if the correct tool was used (or none was needed and none
        was used); 0.5 if a tool was used when none was needed; 0.0 if
        a needed tool was missing or the wrong tool was used.
    """
    used_tools = {tc.tool_name for tc in response.tool_calls}
    # Filter out trivial direct-answer sentinels from the "real" tool set.
    real_tools = used_tools - {"AnswerDirect"}

    if not needed_tool:
        # No tool required.
        return 1.0 if not real_tools else 0.5

    if needed_tool in used_tools:
        return 1.0
    return 0.0


def _escalation_accuracy(
    response: QueryResponse, should_escalate: bool
) -> float:
    """1.0 if ticket-id presence matches should_escalate, else 0.0.

    Args:
        response: The QueryResponse (``ticket_id`` indicates escalation).
        should_escalate: Whether the query should have been escalated.

    Returns:
        1.0 on agreement, 0.0 otherwise.
    """
    escalated = response.ticket_id is not None
    return 1.0 if escalated == bool(should_escalate) else 0.0


def compute_fcrs(response: QueryResponse, gold: dict[str, Any]) -> dict[str, float]:
    """Compute the FCRS composite and its four sub-scores.

    Expected keys on ``gold``:
        - ``key_points``: list[str] -- required gold key points.
        - ``doc_ids``: list[str] (optional) -- valid citation doc ids.
        - ``needed_tool``: str | None (optional) -- expected tool name.
        - ``should_escalate``: bool (optional) -- whether to escalate.

    Args:
        response: The full QueryResponse produced by a pipeline.
        gold: Gold annotation dict for the query.

    Returns:
        Dict with keys ``fcrs``, ``completeness``, ``citation_coverage``,
        ``tool_appropriateness``, ``escalation_accuracy``.
    """
    key_points: list[str] = list(gold.get("key_points") or [])
    gold_doc_ids: set[str] = set(gold.get("doc_ids") or [])
    needed_tool: str | None = gold.get("needed_tool")
    should_escalate: bool = bool(gold.get("should_escalate", False))

    completeness = _completeness(response.answer, key_points)
    citation_coverage = _citation_coverage(
        response.answer, response, gold_doc_ids
    )
    tool_appropriateness = _tool_appropriateness(response, needed_tool)
    escalation_accuracy = _escalation_accuracy(response, should_escalate)

    fcrs = (
        0.35 * completeness
        + 0.25 * citation_coverage
        + 0.20 * tool_appropriateness
        + 0.20 * escalation_accuracy
    )

    return {
        "fcrs": float(fcrs),
        "completeness": float(completeness),
        "citation_coverage": float(citation_coverage),
        "tool_appropriateness": float(tool_appropriateness),
        "escalation_accuracy": float(escalation_accuracy),
    }
