"""Facade: ``run_decomposition_test`` for ``python run.py test-decomp``.

Runs the query-decomposition splitter on a mix of multi-part and
single-part sample queries, recording per-query latency, detected
multi-part flag, and resulting sub-queries. Used for spot-checking the
decomposer after training or prompt changes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.decomposer.splitter import QuerySplitter

logger = logging.getLogger(__name__)


_MULTI_PART_SAMPLES: tuple[str, ...] = (
    "What is the refund policy and how long does it take to receive the money back?",
    "How do I update my billing address and also change my payment method?",
    "What are your business hours and do you offer same-day shipping?",
    "Can I cancel my subscription anytime and will I get a prorated refund?",
    "How do I enable 2FA and what backup options are available if I lose my phone?",
    "What is the password reset process and how long is the reset link valid for?",
    "Where can I find the API key and how do I rotate it?",
    "What data do you collect and how long is it retained?",
    "How do I contact support and what are the response SLAs?",
    "What payment methods do you accept and are international cards supported?",
)

_SINGLE_PART_SAMPLES: tuple[str, ...] = (
    "How do I reset my password?",
    "What is your refund policy?",
    "Can I transfer my license to another user?",
    "What are the supported payment methods?",
    "How long does shipping take?",
    "How do I update my billing address?",
    "What happens if my payment fails?",
    "How do I cancel my account?",
    "Is my data encrypted at rest?",
    "Do you support SSO?",
)


def _sample_queries(n: int) -> list[tuple[str, bool]]:
    """Return ``[(query, gold_is_multi_part)]`` of total length ``n``."""
    pairs: list[tuple[str, bool]] = []
    half = max(1, n // 2)
    for i in range(half):
        pairs.append((_MULTI_PART_SAMPLES[i % len(_MULTI_PART_SAMPLES)], True))
    for i in range(n - half):
        pairs.append((_SINGLE_PART_SAMPLES[i % len(_SINGLE_PART_SAMPLES)], False))
    return pairs


def _build_generator() -> Any:
    """Construct a minimal generator that QuerySplitter can drive."""
    try:
        from src.models.generator import Generator  # lazy import

        return Generator()
    except Exception as exc:
        logger.warning(
            "Could not instantiate real Generator (%s); using heuristic stub", exc
        )

    class _HeuristicStub:
        """Stub generator: emits one sub-query per 'and'/'also' clause.

        Enough to exercise the splitter's plumbing in smoke-tests when
        the real LLM is unavailable (e.g. CPU-only CI).
        """

        def generate(self, prompt: str, **_: Any) -> str:
            # Very crude: just surface the original query back. The real
            # splitter's regex heuristics take over from there.
            return prompt.splitlines()[-1] if prompt else ""

    return _HeuristicStub()


def run_decomposition_test(
    n: int = 20,
    config: Any | None = None,  # noqa: ARG001 -- kept for CLI symmetry
) -> list[dict[str, Any]]:
    """Decompose ``n`` sample queries and return per-sample results."""
    splitter = QuerySplitter(generator=_build_generator())
    queries = _sample_queries(n)

    results: list[dict[str, Any]] = []
    for original, gold in queries:
        t0 = time.perf_counter()
        try:
            subs = splitter.split(original)
        except Exception as exc:
            logger.warning("Split failed for %r: %s", original, exc)
            subs = [original]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # A query is considered decomposed when the splitter produces
        # more than one sub-query.
        is_multi = len(subs) > 1
        results.append(
            {
                "original": original,
                "sub_queries": list(subs),
                "is_multi_part": is_multi,
                "gold_multi_part": gold,
                "correct": is_multi == gold,
                "time_ms": dt_ms,
            }
        )
    return results
