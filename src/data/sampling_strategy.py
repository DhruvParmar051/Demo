"""Self-consistency scoring for QA generation.

Given N candidate ``{query, answer}`` dicts generated for the same chunk,
pick the highest-scoring candidate via cheap heuristics (no extra model
calls):

  * citation present in answer
  * token overlap with the chunk text (grounding proxy)
  * answer length within a sweet-spot window
  * query length not degenerate
"""

from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[[A-Za-z0-9_\-]+:\d+\-\d+\]")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta))


def _length_score(n_words: int, lo: int = 15, hi: int = 80) -> float:
    if n_words < lo:
        return n_words / lo
    if n_words > hi:
        return max(0.0, 1.0 - (n_words - hi) / hi)
    return 1.0


def score_candidate(cand: dict, chunk_text: str, needs_citation: bool = True) -> float:
    """Return a scalar quality score in roughly [0, 3]."""
    query = str(cand.get("query", "")).strip()
    answer = str(cand.get("answer", "")).strip()
    if not query or not answer:
        return 0.0

    ans_words = answer.split()
    s = 0.0
    # 1. grounding
    s += _overlap(answer, chunk_text)
    # 2. length sweet-spot
    s += _length_score(len(ans_words))
    # 3. query non-degenerate
    s += 1.0 if 4 <= len(query.split()) <= 40 else 0.3
    # 4. citation presence
    has_cite = bool(_CITATION_RE.search(answer))
    if needs_citation:
        s += 1.0 if has_cite else -0.5
    else:
        s += 1.0 if not has_cite else -0.5
    return s


def pick_best(
    candidates: list[dict],
    chunk_text: str,
    needs_citation: bool = True,
) -> dict | None:
    """Return the highest-scoring candidate, or None if all empty."""
    scored = [
        (score_candidate(c, chunk_text, needs_citation=needs_citation), c)
        for c in candidates
        if isinstance(c, dict)
    ]
    scored = [(s, c) for s, c in scored if s > 0.0]
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]
