"""
AegisRAG - Self-Consistency Scoring

Given N candidate ``{query, answer}`` dicts generated for the same chunk,
picks the highest-quality candidate using cheap heuristics — no extra model
calls required:

  - Token overlap with the source chunk text (grounding proxy)
  - Answer length within a sweet-spot window (15–80 words)
  - Query length check (rejects degenerate one-word queries)
  - Citation presence / absence based on task type
"""

from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[[A-Za-z0-9_\-]+:\d+\-\d+\]")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


# ---------------------------------------------------------------------------
# Internal scoring primitives
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    """Return the lowercase word token set for *text*."""
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def _overlap(a: str, b: str) -> float:
    """Fraction of *a*-tokens that also appear in *b*."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta))


def _length_score(n_words: int, lo: int = 15, hi: int = 80) -> float:
    """Return 1.0 for answers in the [lo, hi] word range, decaying outside it."""
    if n_words < lo:
        return n_words / lo
    if n_words > hi:
        return max(0.0, 1.0 - (n_words - hi) / hi)
    return 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_candidate(cand: dict, chunk_text: str, needs_citation: bool = True) -> float:
    """Return a scalar quality score in roughly [0, 4].

    Parameters
    ----------
    cand : dict
        Candidate with ``"query"`` and ``"answer"`` keys.
    chunk_text : str
        Source chunk the answer should be grounded in.
    needs_citation : bool
        Whether a ``[doc_id:start-end]`` citation marker is expected.
    """
    query = str(cand.get("query", "")).strip()
    answer = str(cand.get("answer", "")).strip()
    if not query or not answer:
        return 0.0

    s = 0.0
    s += _overlap(answer, chunk_text)                          # grounding
    s += _length_score(len(answer.split()))                    # length sweet-spot
    s += 1.0 if 4 <= len(query.split()) <= 40 else 0.3        # non-degenerate query

    has_cite = bool(_CITATION_RE.search(answer))
    s += 1.0 if (has_cite == needs_citation) else -0.5        # citation presence

    return s


def pick_best(
    candidates: list[dict],
    chunk_text: str,
    needs_citation: bool = True,
) -> dict | None:
    """Return the highest-scoring candidate from *candidates*, or ``None`` if all empty.

    Parameters
    ----------
    candidates : list[dict]
        List of ``{query, answer}`` dicts to rank.
    chunk_text : str
        Source chunk used for grounding overlap scoring.
    needs_citation : bool
        Passed through to :func:`score_candidate`.
    """
    scored = [
        (score_candidate(c, chunk_text, needs_citation=needs_citation), c)
        for c in candidates
        if isinstance(c, dict)
    ]
    scored = [(s, c) for s, c in scored if s > 0.0]
    if not scored:
        return None
    return max(scored, key=lambda x: x[0])[1]
