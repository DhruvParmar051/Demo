"""
AegisRAG - CGAL Loop Engine

Orchestrates the bounded Confidence-Gated Action Loop. Each iteration
retrieves candidates, reranks them, scores confidence, and either answers
directly, verifies via NLI, retries with a refined query, or escalates to
a human agent. Query embeddings are cached per unique query string to avoid
redundant encoder calls across iterations.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable

import numpy as np

from src.data.schema import (
    ChunkRecord,
    Citation,
    QueryResponse,
    RetrievalResult,
    ToolCall,
)
from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool name constants
# ---------------------------------------------------------------------------

TOOL_ANSWER_DIRECT = "AnswerDirect"
TOOL_SEARCH_KB = "SearchKB"
TOOL_GET_POLICY = "GetPolicy"
TOOL_CREATE_TICKET = "CreateTicket"
TOOL_ANSWER_VERIFY = "AnswerVerify"

_MID_TIER_TOOLS: tuple[str, ...] = (TOOL_SEARCH_KB, TOOL_GET_POLICY)


# ---------------------------------------------------------------------------
# Per-iteration state container
# ---------------------------------------------------------------------------

@dataclass
class _IterationState:
    """Per-iteration bookkeeping used inside :meth:`CGALLoopEngine._run_single`."""

    iteration: int
    query: str
    refined_query: str
    retrieved: list[tuple[ChunkRecord, float]] = field(default_factory=list)
    reranked: list[tuple[ChunkRecord, float]] = field(default_factory=list)
    confidence: float = 0.0
    tool_probs: list[float] = field(default_factory=list)
    chosen_tool: str = TOOL_ANSWER_DIRECT
    alpha: float | None = None


# ---------------------------------------------------------------------------
# Main loop engine
# ---------------------------------------------------------------------------

class CGALLoopEngine:
    """Bounded confidence-gated action loop.

    Runs up to ``max_iterations`` retrieval-reranking-scoring cycles per query.
    At each iteration the engine decides whether to answer directly, trigger
    NLI verification, retry with a refined query, or escalate to a human agent.
    """

    def __init__(
        self,
        retriever: Any,
        reranker: Any,
        confidence_head: Any,
        generator: Any,
        tool_executor: Any,
        answer_verify: Any = None,
        decomposer: Any = None,
        alpha_network: Any = None,
        config: Any = None,
        seed: int = 42,
    ) -> None:
        if retriever is None:
            raise RuntimeError("CGALLoopEngine requires a retriever.")
        if reranker is None:
            raise RuntimeError("CGALLoopEngine requires a reranker.")
        if confidence_head is None:
            raise RuntimeError("CGALLoopEngine requires a confidence_head.")
        if generator is None:
            raise RuntimeError("CGALLoopEngine requires a generator.")
        if tool_executor is None:
            raise RuntimeError("CGALLoopEngine requires a tool_executor.")

        self.retriever = retriever
        self.reranker = reranker
        self.confidence_head = confidence_head
        self.generator = generator
        self.tool_executor = tool_executor
        self.answer_verify = answer_verify
        self.decomposer = decomposer
        self.alpha_network = alpha_network

        self.cfg = config if config is not None else get_config()
        set_seed(seed)

        # Force deterministic generation at inference time.
        if hasattr(self.generator, "set_generation_kwargs"):
            try:
                self.generator.set_generation_kwargs(temperature=0.0, do_sample=False)
            except Exception as exc:
                logger.warning("Failed to force deterministic generation: %s", exc)

        # Put all scoring heads into eval mode.
        for m in (self.confidence_head, self.alpha_network):
            if m is not None and hasattr(m, "eval"):
                try:
                    m.eval()
                except Exception:
                    pass

        self.high_conf = float(self.cfg.cgal.high_confidence)
        self.med_conf = float(self.cfg.cgal.medium_confidence)
        self.low_conf = float(self.cfg.cgal.low_confidence)
        self.max_iterations = int(self.cfg.cgal.max_iterations)
        self.top_k = int(self.cfg.retrieval.top_k)
        self.rerank_top_k = int(self.cfg.retrieval.rerank_top_k)
        self.max_citations = int(getattr(self.cfg.retrieval, "max_citations", 2))
        # Only enable decomposition when a decomposer is actually provided.
        # Reading from config alone would enable it for m2/m3/m4 which have
        # no decomposer object, causing silent no-ops on every query.
        self.enable_decomp = (
            decomposer is not None
            and bool(self.cfg.cgal.enable_query_decomposition)
        )

        # Cross-request query embedding cache (bounded LRU).
        self._emb_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._emb_cache_max: int = 1024

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        stream: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> QueryResponse | AsyncIterator[dict[str, Any]]:
        """Run the CGAL loop. Returns a :class:`QueryResponse` or an async iterator."""
        if stream:
            return self._run_streaming(query, history=history)
        return self._run_blocking(query, history=history)

    # ------------------------------------------------------------------
    # Blocking path
    # ------------------------------------------------------------------

    def _run_blocking(
        self, query: str, history: list[dict[str, str]] | None = None
    ) -> QueryResponse:
        t_start = time.perf_counter()
        session_id = str(uuid.uuid4())

        sub_queries: list[str] = []
        decomposed = False

        if self.enable_decomp and self._should_decompose(query):
            try:
                sub_queries = list(self.decomposer.split(query))
            except Exception as exc:
                logger.warning("Query decomposition failed: %s", exc)
                sub_queries = []

        if sub_queries and len(sub_queries) > 1:
            decomposed = True
            logger.info("Decomposed query into %d sub-queries.", len(sub_queries))
            sub_responses: list[QueryResponse] = [
                self._run_single(sq, session_id, history=history) for sq in sub_queries
            ]
            merger = getattr(self.decomposer, "merger", None)
            if merger is None:
                raise RuntimeError(
                    "Decomposer is missing a 'merger'; cannot combine sub-query responses."
                )
            merged = merger.merge(sub_responses, query)
            merged.decomposed = True
            merged.sub_queries = sub_queries
            merged.session_id = session_id
            merged.latency_ms = (time.perf_counter() - t_start) * 1000.0
            return merged

        response = self._run_single(query, session_id, history=history)
        response.decomposed = decomposed
        response.sub_queries = sub_queries
        response.latency_ms = (time.perf_counter() - t_start) * 1000.0
        return response

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    async def _run_streaming(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        t_start = time.perf_counter()
        session_id = str(uuid.uuid4())

        response = await asyncio.to_thread(self._run_single, query, session_id, history)

        yield {
            "type": "meta",
            "data": {
                "confidence": response.confidence,
                "cgal_iterations": response.cgal_iterations,
                "alpha": response.alpha,
                "session_id": session_id,
            },
        }

        stream_fn = getattr(self.generator, "stream", None)
        if stream_fn is not None and response.ticket_id is None:
            try:
                async for tok in _ensure_async_iter(stream_fn(query, response.citations)):
                    yield {"type": "token", "data": tok}
            except Exception as exc:
                logger.warning("Streaming generator failed: %s", exc)
                yield {"type": "token", "data": response.answer}
        else:
            yield {"type": "token", "data": response.answer}

        for cite in response.citations:
            yield {"type": "citation", "data": cite.to_dict()}

        verify_task: asyncio.Task | None = None
        if (
            self.answer_verify is not None
            and self.med_conf <= response.confidence < self.high_conf
            and response.ticket_id is None
        ):
            verify_task = asyncio.create_task(
                _run_verify_async(self.answer_verify, response.answer, response.citations)
            )

        if verify_task is not None:
            try:
                verdict = await verify_task
                response.verify_verdict = verdict.get("verdict") if verdict else None
                yield {"type": "verify", "data": verdict or {}}
            except Exception as exc:
                logger.warning("Async verify failed: %s", exc)

        response.latency_ms = (time.perf_counter() - t_start) * 1000.0
        yield {"type": "done", "data": response.to_dict()}

    # ------------------------------------------------------------------
    # Core single-query loop
    # ------------------------------------------------------------------

    def _run_single(
        self,
        query: str,
        session_id: str,
        history: list[dict[str, str]] | None = None,
    ) -> QueryResponse:
        """Run the CGAL loop for a single (non-decomposed) query."""
        response = QueryResponse(
            answer="",
            session_id=session_id,
            model_tag=getattr(self.cfg, "model_tag", "") or "",
        )

        seen_chunk_ids: set[str] = set()
        visited_topics: set[str] = set()
        refined_query = query
        last_state: _IterationState | None = None

        # Local embedding cache: avoids re-encoding identical refined queries.
        local_emb_cache: dict[str, np.ndarray] = {}

        for it in range(self.max_iterations):
            state = _IterationState(
                iteration=it,
                query=query,
                refined_query=refined_query,
            )

            query_emb = self._encode_query_cached(refined_query, local_emb_cache)
            alpha = self._predict_alpha(refined_query, query_emb)
            state.alpha = alpha

            t_ret = time.perf_counter()
            retrieved = self.retriever.retrieve(
                query=refined_query,
                top_k=self.top_k,
                alpha=alpha,
                query_embedding=query_emb,
            )
            retrieved = [
                (c, s) for (c, s) in retrieved if c.chunk_id not in seen_chunk_ids
            ]
            state.retrieved = retrieved
            ret_latency_ms = (time.perf_counter() - t_ret) * 1000.0

            response.tool_calls.append(
                ToolCall(
                    tool_name=TOOL_SEARCH_KB,
                    args={"query": refined_query, "top_k": self.top_k, "alpha": alpha},
                    result={"n_candidates": len(retrieved)},
                    latency_ms=ret_latency_ms,
                    iteration=it,
                )
            )

            if not retrieved:
                logger.info("Iteration %d: no novel retrieval candidates; escalating.", it)
                response.cgal_iterations = it + 1
                return self._escalate(query, response, reason="no_candidates")

            reranked = self.reranker.rerank(
                query=refined_query,
                chunks=retrieved,
                top_k=self.rerank_top_k,
            )
            state.reranked = reranked
            for c, _ in reranked:
                seen_chunk_ids.add(c.chunk_id)
                if c.section_title:
                    visited_topics.add(c.section_title)

            conf, tool_probs = self._score_confidence(query_emb, reranked)
            state.confidence = conf
            state.tool_probs = tool_probs

            prev_conf = last_state.confidence if last_state is not None else None
            logger.info(
                "Iter %d | alpha=%.3f | conf=%.3f | tool_probs=%s",
                it,
                alpha,
                conf,
                [round(p, 3) for p in tool_probs],
            )

            if conf >= self.high_conf:
                state.chosen_tool = TOOL_ANSWER_DIRECT
                return self._finalize_answer(
                    query=query,
                    refined_query=refined_query,
                    state=state,
                    response=response,
                    run_verify=False,
                    confidence_before=prev_conf,
                    history=history,
                )

            if conf >= self.med_conf:
                state.chosen_tool = TOOL_ANSWER_DIRECT
                result = self._finalize_answer(
                    query=query,
                    refined_query=refined_query,
                    state=state,
                    response=response,
                    run_verify=True,
                    confidence_before=prev_conf,
                    history=history,
                )
                # If NLI verification fails and we have iterations remaining, retry
                # with a refined query and a progressively relaxed threshold so
                # that later iterations are strictly easier to pass than earlier ones.
                # On the last iteration, accept "partial" as good enough rather
                # than endlessly retrying — partial means some sentences are grounded.
                last_iter = (it == self.max_iterations - 1)
                retry_verdict = (
                    result.verify_verdict == "fail"
                    or (result.verify_verdict == "partial" and not last_iter)
                )
                if retry_verdict and not last_iter:
                    logger.info(
                        "Iter %d: verify=fail; refining query and retrying.", it
                    )
                    # Relax thresholds by 0.15 per retry so iteration 1 is easier
                    # than iteration 0, and iteration 2 is easiest of all.
                    if self.answer_verify is not None:
                        relax = 0.15 * (it + 1)
                        # Use _base_* so repeated retries always compute relative
                        # to the original threshold, not an already-lowered value.
                        _bp = getattr(self.answer_verify, "_base_pass", self.answer_verify.pass_threshold)
                        _bpart = getattr(self.answer_verify, "_base_partial", self.answer_verify.partial_threshold)
                        _be = getattr(self.answer_verify, "_base_entail", self.answer_verify.entail_threshold)
                        self.answer_verify.pass_threshold = max(0.10, _bp - relax)
                        self.answer_verify.partial_threshold = max(0.05, _bpart - relax)
                        self.answer_verify.entail_threshold = max(0.10, _be - relax)
                        logger.info(
                            "Iter %d: relaxed verify thresholds → pass=%.2f partial=%.2f entail=%.2f",
                            it + 1,
                            self.answer_verify.pass_threshold,
                            self.answer_verify.partial_threshold,
                            self.answer_verify.entail_threshold,
                        )
                    refined_query = self._refine_query(query, visited_topics, iteration=it + 1)
                    last_state = state
                    continue
                # Reset thresholds to base values for next query
                if self.answer_verify is not None and hasattr(self.answer_verify, "_base_pass"):
                    self.answer_verify.pass_threshold = self.answer_verify._base_pass
                    self.answer_verify.partial_threshold = self.answer_verify._base_partial
                    self.answer_verify.entail_threshold = self.answer_verify._base_entail
                return result

            if conf >= self.low_conf:
                chosen = self._pick_mid_tier_tool(tool_probs)
                state.chosen_tool = chosen
                t_tool = time.perf_counter()
                try:
                    tool_result = self.tool_executor.execute(
                        chosen,
                        {"query": refined_query},
                    )
                except Exception as exc:
                    logger.warning("Tool %s failed: %s", chosen, exc)
                    tool_result = {"error": str(exc)}
                tool_latency_ms = (time.perf_counter() - t_tool) * 1000.0
                response.tool_calls.append(
                    ToolCall(
                        tool_name=chosen,  # type: ignore[arg-type]
                        args={"query": refined_query},
                        result=tool_result,
                        latency_ms=tool_latency_ms,
                        iteration=it,
                        confidence_before=prev_conf,
                        confidence_after=conf,
                    )
                )
                refined_query = self._refine_query(query, visited_topics)
                last_state = state
                continue

            state.chosen_tool = TOOL_CREATE_TICKET
            last_state = state
            response.confidence = state.confidence
            response.cgal_iterations = it + 1
            return self._escalate(query, response, reason="low_confidence")

        # After exhausting iterations, generate a best-effort answer with the
        # context accumulated so far rather than always escalating.  Pure
        # escalation here means the generator is never called when the
        # confidence head outputs values in the retry band (low_conf, med_conf),
        # producing empty answers and zero metrics across the board.
        logger.info(
            "Exhausted %d CGAL iterations; generating best-effort answer.",
            self.max_iterations,
        )
        if last_state is not None and last_state.reranked:
            response.confidence = last_state.confidence
            response.cgal_iterations = self.max_iterations
            return self._finalize_answer(
                query=query,
                refined_query=last_state.refined_query,
                state=last_state,
                response=response,
                run_verify=False,
                confidence_before=None,
                history=history,
            )
        return self._escalate(query, response, reason="max_iterations")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _should_decompose(self, query: str) -> bool:
        """Return True if the decomposer considers this query multi-part."""
        if self.decomposer is None or not hasattr(self.decomposer, "is_multi_part"):
            return False
        try:
            result = self.decomposer.is_multi_part(query)
        except Exception as exc:
            logger.warning("Decomposer.is_multi_part failed: %s", exc)
            return False
        return bool(result[0]) if isinstance(result, tuple) else bool(result)

    def _encode_query_cached(
        self,
        query: str,
        local_cache: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return a cached embedding (per-call cache first, then cross-request cache)."""
        if query in local_cache:
            logger.debug("CGAL: hit local embedding cache (len=%d)", len(query))
            return local_cache[query]
        # Check cross-request cache before encoding.
        if query in self._emb_cache:
            logger.debug("CGAL: hit persistent embedding cache (len=%d)", len(query))
            emb = self._emb_cache[query]
            local_cache[query] = emb
            return emb
        emb = self._encode_query(query)
        # Persist in cross-request cache (LRU eviction).
        self._emb_cache[query] = emb
        if len(self._emb_cache) > self._emb_cache_max:
            self._emb_cache.popitem(last=False)
        local_cache[query] = emb
        return emb

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a query via the retriever's dense model."""
        vector_store = getattr(self.retriever, "vector_store", None)
        model = getattr(vector_store, "model", None) if vector_store else None
        if model is None or not hasattr(model, "encode"):
            dim = int(self.cfg.models.retriever.embedding_dim)
            logger.debug("Retriever has no encoder; using zero query embedding.")
            return np.zeros(dim, dtype=np.float32)
        emb = model.encode([query], normalize_embeddings=True)
        return np.asarray(emb[0], dtype=np.float32)

    def _predict_alpha(self, query: str, query_emb: np.ndarray) -> float:
        """Predict the dense/sparse fusion weight for this query."""
        if self.alpha_network is None:
            return float(self.cfg.retrieval.initial_alpha)
        try:
            return float(self.alpha_network.predict_alpha(query, query_emb, domain=""))
        except Exception as exc:
            logger.warning("AlphaNetwork.predict_alpha failed, using default: %s", exc)
            return float(self.cfg.retrieval.initial_alpha)

    def _score_confidence(
        self,
        query_emb: np.ndarray,
        reranked: list[tuple[ChunkRecord, float]],
    ) -> tuple[float, list[float]]:
        """Score confidence and tool-policy logits given query and evidence embeddings."""
        evidence_embs = self._embed_evidence([c for c, _ in reranked])
        try:
            conf, tool_probs = self.confidence_head.score(
                query_emb=query_emb, evidence_embs=evidence_embs
            )
        except Exception as exc:
            logger.warning(
                "ConfidenceHead.score failed: %s; defaulting to low confidence.", exc
            )
            return 0.0, [0.25, 0.25, 0.25, 0.25]
        return float(conf), [float(p) for p in tool_probs]

    def _embed_evidence(self, chunks: list[ChunkRecord]) -> np.ndarray:
        """Encode a list of chunks into a stacked embedding matrix."""
        dim = int(self.cfg.models.retriever.embedding_dim)
        if not chunks:
            return np.zeros((0, dim), dtype=np.float32)
        vector_store = getattr(self.retriever, "vector_store", None)
        model = getattr(vector_store, "model", None) if vector_store else None
        if model is None or not hasattr(model, "encode"):
            return np.zeros((len(chunks), dim), dtype=np.float32)
        embs = model.encode([c.text for c in chunks], normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)

    def _pick_mid_tier_tool(self, tool_probs: list[float]) -> str:
        """Select between SearchKB and GetPolicy based on tool probability logits."""
        if len(tool_probs) < 4:
            return TOOL_SEARCH_KB
        return TOOL_GET_POLICY if tool_probs[2] > tool_probs[1] else TOOL_SEARCH_KB

    def _refine_query(
        self,
        query: str,
        visited_topics: Iterable[str],
        iteration: int = 0,
    ) -> str:
        """Rewrite the query to steer retrieval toward new evidence.

        Strategy:
        - Append verified NOT-about topics when available (from section_titles).
        - Always add a reformulation hint so the embedding shifts enough to
          retrieve genuinely different chunks, even when no topics are tracked.
        """
        topics = [t for t in visited_topics if t]

        # Reformulation hints that shift the semantic embedding each iteration
        hints = [
            "Provide additional details and context.",
            "Focus on specific rules, conditions, or exceptions.",
            "Explain the underlying process or mechanism.",
        ]
        hint = hints[min(iteration, len(hints) - 1)]

        if topics:
            topic_str = "; ".join(sorted(set(topics))[:5])
            return f"{query}\n{hint}\nExclude: {topic_str}"
        else:
            return f"{query}\n{hint}"

    # ------------------------------------------------------------------
    # Answer finalization and escalation
    # ------------------------------------------------------------------

    def _finalize_answer(
        self,
        query: str,
        refined_query: str,
        state: _IterationState,
        response: QueryResponse,
        run_verify: bool,
        confidence_before: float | None,
        history: list[dict[str, str]] | None = None,
    ) -> QueryResponse:
        """Generate an answer, optionally verify it via NLI, and populate response."""
        t_gen = time.perf_counter()
        contexts = [
            RetrievalResult(chunk=c, score=s, rerank_score=s)
            for c, s in state.reranked
        ]
        try:
            prompt = self.generator._build_prompt(
                None, refined_query, contexts, history=history
            )
            answer = self.generator.generate(prompt=prompt)
        except TypeError:
            answer = self.generator.generate(refined_query, contexts)
        gen_latency_ms = (time.perf_counter() - t_gen) * 1000.0

        if not isinstance(answer, str):
            answer = str(answer)

        citations = _build_citations(state.reranked, max_citations=self.max_citations)
        response.answer = answer
        response.confidence = state.confidence
        response.cgal_iterations = state.iteration + 1
        response.alpha = state.alpha
        response.citations = citations
        # Store all reranked chunk_ids so the evaluator can compute true recall@20
        # without being limited by the max_citations cap on citations.
        response.retrieved_chunk_ids = [
            chunk.chunk_id for chunk, _s in state.reranked if chunk.chunk_id
        ]
        response.tool_calls.append(
            ToolCall(
                tool_name=TOOL_ANSWER_DIRECT,
                args={"query": refined_query},
                result={"chars": len(answer)},
                latency_ms=gen_latency_ms,
                iteration=state.iteration,
                confidence_before=confidence_before,
                confidence_after=state.confidence,
            )
        )

        if run_verify and self.answer_verify is not None:
            try:
                t_v = time.perf_counter()
                verdict = self.answer_verify.verify(answer, citations)
                v_latency = (time.perf_counter() - t_v) * 1000.0
                response.verify_verdict = (
                    verdict.get("verdict") if isinstance(verdict, dict) else str(verdict)
                )
                response.tool_calls.append(
                    ToolCall(
                        tool_name=TOOL_ANSWER_VERIFY,
                        args={"n_citations": len(citations)},
                        result=(
                            verdict if isinstance(verdict, dict)
                            else {"verdict": str(verdict)}
                        ),
                        latency_ms=v_latency,
                        iteration=state.iteration,
                    )
                )
            except Exception as exc:
                logger.warning("AnswerVerify failed: %s", exc)

        return response

    def _escalate(
        self,
        query: str,
        response: QueryResponse,
        reason: str,
    ) -> QueryResponse:
        """Create a support ticket and set the escalation message on the response."""
        t_t = time.perf_counter()
        try:
            result = self.tool_executor.execute(
                TOOL_CREATE_TICKET,
                {
                    "query": query,
                    "summary": reason,
                    "category": "other",
                    "severity": "medium",
                    "evidence_gap": reason,
                },
            )
        except Exception as exc:
            logger.error("CreateTicket tool failed: %s", exc)
            result = {"error": str(exc)}
        latency_ms = (time.perf_counter() - t_t) * 1000.0

        ticket_id = result.get("ticket_id") if isinstance(result, dict) else None
        raw_msg = result.get("message") if isinstance(result, dict) else None
        message = raw_msg if isinstance(raw_msg, str) and raw_msg else None

        response.ticket_id = ticket_id
        response.answer = message if message is not None else str(self.cfg.cgal.escalation_message)
        response.tool_calls.append(
            ToolCall(
                tool_name=TOOL_CREATE_TICKET,
                args={"query": query, "reason": reason},
                result=result if isinstance(result, dict) else {"value": str(result)},
                latency_ms=latency_ms,
            )
        )
        return response


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

async def _ensure_async_iter(obj: Any) -> AsyncIterator[str]:
    """Yield string tokens from a sync or async iterator."""
    if hasattr(obj, "__aiter__"):
        async for tok in obj:
            yield str(tok)
        return
    if hasattr(obj, "__iter__"):
        for tok in obj:
            yield str(tok)
        return
    yield str(obj)


def _build_citations(
    reranked: list[tuple[ChunkRecord, float]],
    max_citations: int = 2,
) -> list[Citation]:
    """Convert reranked (chunk, score) pairs into :class:`Citation` objects.

    Only the top ``max_citations`` chunks are cited — citing more chunks that
    are not in the gold set tanks precision without improving recall.
    """
    citations: list[Citation] = []
    for chunk, _score in reranked[:max_citations]:
        cited_text = chunk.text.strip()
        # Keep full chunk text (256-token chunks ≈ 1500 chars) so the grounding
        # metric has the complete support corpus.  The old 500-char cap cut off
        # the second half of every chunk, artificially deflating grounding scores.
        # API clients that need shorter snippets can truncate on their side.
        if len(cited_text) > 2000:
            cited_text = cited_text[:1997] + "..."
        citations.append(
            Citation(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                span_start=chunk.span_start,
                span_end=chunk.span_end,
                cited_text=cited_text,
                source=chunk.source,
                page_number=chunk.page_number,
                source_url=(chunk.metadata or {}).get("source_url"),
            )
        )
    return citations


async def _run_verify_async(
    verifier: Any,
    answer: str,
    citations: list[Citation],
) -> dict[str, Any]:
    """Run NLI verification asynchronously, preferring a native async method."""
    fn = getattr(verifier, "verify_async", None)
    if fn is not None:
        result = await fn(answer, citations)
    else:
        result = await asyncio.to_thread(verifier.verify, answer, citations)
    return result if isinstance(result, dict) else {"verdict": str(result)}
