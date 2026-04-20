"""
AegisRAG - Result Merger

Combines the :class:`QueryResponse` objects produced by running each
atomic sub-query through the CGAL loop.  The merger:

* concatenates answers with ``For [sub_query]:`` framing;
* merges and deduplicates citations (by ``chunk_id``);
* concatenates all tool calls in order;
* takes the *min* confidence across sub-queries (worst-case reporting);
* takes the *max* ``cgal_iterations`` across sub-queries;
* OR-s escalation (any escalated sub-query escalates the merged response);
* averages ``alpha`` across sub-queries that have one;
* joins ``sub_queries`` for audit trail;
* sums ``latency_ms`` (sequential path) or takes the max (parallel path).
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterable

from src.data.schema import Citation, QueryResponse, ToolCall

logger = logging.getLogger(__name__)


class ResultMerger:
    """Merge a list of sub-query :class:`QueryResponse` objects into one.

    Parameters
    ----------
    parallel : bool
        When True, ``latency_ms`` is computed as the max of sub-latencies
        (assuming the caller ran sub-queries concurrently).  When False,
        latencies are summed.
    """

    def __init__(self, parallel: bool = False) -> None:
        self.parallel = bool(parallel)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(
        self,
        sub_responses: list[QueryResponse],
        original_query: str,
    ) -> QueryResponse:
        """Merge sub-responses into a single :class:`QueryResponse`.

        Parameters
        ----------
        sub_responses : list[QueryResponse]
            One per sub-query, in original decomposition order.
        original_query : str
            The full user query, recorded for audit/trace purposes.

        Returns
        -------
        QueryResponse
            A single response covering all sub-queries.
        """
        if not sub_responses:
            logger.warning(
                "ResultMerger.merge called with empty sub_responses list "
                "for query: %s",
                original_query[:80],
            )
            return QueryResponse(
                answer="",
                session_id=str(uuid.uuid4()),
            )

        if len(sub_responses) == 1:
            # Nothing to merge, but flag as decomposed-but-single so the
            # caller knows the pipeline went through this path.
            only = sub_responses[0]
            only.decomposed = True
            if not only.sub_queries:
                only.sub_queries = [original_query]
            return only

        answer = self._merge_answers(sub_responses)
        citations = self._merge_citations(r.citations for r in sub_responses)
        tool_calls = self._merge_tool_calls(r.tool_calls for r in sub_responses)
        confidence = min(r.confidence for r in sub_responses)
        iterations = max(r.cgal_iterations for r in sub_responses)
        escalated = any(r.ticket_id for r in sub_responses)
        ticket_id = next(
            (r.ticket_id for r in sub_responses if r.ticket_id), None
        )
        alpha = self._average_alpha(sub_responses)
        verify_verdict = self._combine_verdicts(
            r.verify_verdict for r in sub_responses
        )

        latencies = [float(r.latency_ms) for r in sub_responses if r.latency_ms]
        if not latencies:
            total_latency = 0.0
        elif self.parallel:
            total_latency = max(latencies)
        else:
            total_latency = sum(latencies)

        ttfts = [r.ttft_ms for r in sub_responses if r.ttft_ms is not None]
        ttft = min(ttfts) if ttfts else None

        sub_queries: list[str] = []
        for r in sub_responses:
            if r.sub_queries:
                sub_queries.extend(r.sub_queries)
        if not sub_queries:
            sub_queries = [
                r.answer[:120] if r.answer else "(sub-query)"
                for r in sub_responses
            ]

        model_tag = next(
            (r.model_tag for r in sub_responses if r.model_tag), ""
        )

        merged = QueryResponse(
            answer=answer,
            citations=citations,
            tool_calls=tool_calls,
            confidence=float(confidence),
            cgal_iterations=int(iterations),
            ticket_id=ticket_id if escalated else None,
            latency_ms=float(total_latency),
            ttft_ms=ttft,
            session_id=str(uuid.uuid4()),
            decomposed=True,
            sub_queries=sub_queries,
            alpha=alpha,
            verify_verdict=verify_verdict,
            model_tag=model_tag,
        )
        return merged

    # ------------------------------------------------------------------
    # Internal merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_answers(sub_responses: list[QueryResponse]) -> str:
        """Concatenate sub-answers with ``For [sub_query]:`` framing."""
        parts: list[str] = []
        for r in sub_responses:
            header = "(sub-query)"
            if r.sub_queries:
                header = r.sub_queries[0]
            body = r.answer.strip() if r.answer else ""
            parts.append(f"For [{header}]:\n{body}")
        return "\n\n".join(parts)

    @staticmethod
    def _merge_citations(
        citation_iters: Iterable[list[Citation]],
    ) -> list[Citation]:
        """Flatten citation lists, dedupe by ``chunk_id``, preserve order."""
        seen: set[str] = set()
        out: list[Citation] = []
        for citations in citation_iters:
            for c in citations:
                if c.chunk_id in seen:
                    continue
                seen.add(c.chunk_id)
                out.append(c)
        return out

    @staticmethod
    def _merge_tool_calls(
        tool_call_iters: Iterable[list[ToolCall]],
    ) -> list[ToolCall]:
        """Concatenate tool-call lists in sub-query order."""
        out: list[ToolCall] = []
        for tc_list in tool_call_iters:
            out.extend(tc_list)
        return out

    @staticmethod
    def _average_alpha(sub_responses: list[QueryResponse]) -> float | None:
        """Average alpha across sub-responses that reported one."""
        alphas = [r.alpha for r in sub_responses if r.alpha is not None]
        if not alphas:
            return None
        return float(sum(alphas) / len(alphas))

    @staticmethod
    def _combine_verdicts(verdicts: Iterable[str | None]) -> str | None:
        """Combine NLI verdicts conservatively.

        * If any sub-response is ``contradiction`` -> overall ``contradiction``.
        * Else if any is ``neutral`` -> overall ``neutral``.
        * Else if any is ``entailment`` -> overall ``entailment``.
        * Otherwise -> ``None``.
        """
        seen = {v.lower() for v in verdicts if v}
        if not seen:
            return None
        if any("contradict" in v for v in seen):
            return "contradiction"
        if "neutral" in seen:
            return "neutral"
        if any("entail" in v for v in seen):
            return "entailment"
        # Fall back to whichever distinct verdict is present.
        return next(iter(seen))
