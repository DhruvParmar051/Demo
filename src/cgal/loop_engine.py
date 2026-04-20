"""
AegisRAG - CGAL Loop Engine

Orchestrates the bounded Confidence-Gated Action Loop.

For every incoming query the engine:

1. Optionally decomposes multi-part queries into atomic sub-queries
   (``decomposer.is_multi_part`` + ``decomposer.split``), runs each
   sub-query independently through :meth:`_run_single`, and merges the
   results via ``decomposer.merger``.

2. For each single query, iterates up to ``cgal.max_iterations`` times:

   * encode query, predict adaptive ``alpha``,
   * hybrid retrieval (top-``retrieval.top_k``),
   * ColBERT reranking to ``retrieval.rerank_top_k``,
   * confidence + tool-policy scoring,
   * routing:

     - ``conf >= cgal.high_confidence`` -> generate directly (no verify);
     - ``cgal.medium_confidence <= conf < cgal.high_confidence`` -> generate
       and launch async NLI verification;
     - ``cgal.low_confidence <= conf < cgal.medium_confidence`` -> dispatch
       a tool (SearchKB / GetPolicy per tool-policy head), re-retrieve,
       and loop;
     - ``conf < cgal.low_confidence`` -> escalate via CreateTicket.

3. After ``cgal.max_iterations`` iterations without a confident answer,
   escalate via CreateTicket.

The engine is deterministic: a fixed seed is set on construction,
generation is forced to temperature 0, and retrieval results are cached
by ``chunk_id`` so repeat retrievals in a single ``run()`` are stable.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
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


TOOL_ANSWER_DIRECT = "AnswerDirect"
TOOL_SEARCH_KB = "SearchKB"
TOOL_GET_POLICY = "GetPolicy"
TOOL_CREATE_TICKET = "CreateTicket"
TOOL_ANSWER_VERIFY = "AnswerVerify"

_MID_TIER_TOOLS: tuple[str, ...] = (TOOL_SEARCH_KB, TOOL_GET_POLICY)


@dataclass
class _IterationState:
    """Per-iteration bookkeeping used inside :meth:`_run_single`."""

    iteration: int
    query: str
    refined_query: str
    retrieved: list[tuple[ChunkRecord, float]] = field(default_factory=list)
    reranked: list[tuple[ChunkRecord, float]] = field(default_factory=list)
    confidence: float = 0.0
    tool_probs: list[float] = field(default_factory=list)
    chosen_tool: str = TOOL_ANSWER_DIRECT
    alpha: float | None = None


class CGALLoopEngine:
    """The bounded confidence-gated action loop.

    Parameters
    ----------
    retriever : HybridRetriever
        Hybrid dense + sparse retrieval backend.
    reranker : ColBERTReranker
        Cross-encoder reranker.
    confidence_head : ConfidenceHead
        Joint confidence + tool-policy model.
    generator : object
        An object exposing ``generate(query, context) -> str`` and
        optionally ``stream(query, context) -> AsyncIterator[str]`` for
        token streaming.  The engine forces ``temperature=0`` when the
        generator exposes a ``set_generation_kwargs`` hook.
    tool_executor : object
        An object exposing ``execute(tool_name: str, args: dict) -> dict``.
        Must support ``SearchKB``, ``GetPolicy``, and ``CreateTicket``.
    answer_verify : object
        An NLI-based verifier exposing ``verify(answer, citations) -> dict``
        (sync) and ``verify_async(...)`` (awaitable).  May be None.
    decomposer : object or None
        Optional object exposing ``is_multi_part(query) -> bool``,
        ``split(query) -> list[str]``, and ``merger: ResultMerger``.
    alpha_network : AlphaNetwork or None
        Optional adaptive alpha predictor.  If provided, per-query alpha is
        used for hybrid retrieval.
    config : AegisConfig or None
        Optional config override.  Defaults to ``get_config()``.
    seed : int
        Seed used for determinism on construction.
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

        # Force deterministic generation where possible.
        if hasattr(self.generator, "set_generation_kwargs"):
            try:
                self.generator.set_generation_kwargs(
                    temperature=0.0, do_sample=False
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to force deterministic generation: %s", exc)

        # Put modules with parameters into eval() where applicable.
        for m in (self.confidence_head, self.alpha_network):
            if m is not None and hasattr(m, "eval"):
                try:
                    m.eval()
                except Exception:  # pragma: no cover - defensive
                    pass

        # Retrieval thresholds.
        self.high_conf = float(self.cfg.cgal.high_confidence)
        self.med_conf = float(self.cfg.cgal.medium_confidence)
        self.low_conf = float(self.cfg.cgal.low_confidence)
        self.max_iterations = int(self.cfg.cgal.max_iterations)
        self.top_k = int(self.cfg.retrieval.top_k)
        self.rerank_top_k = int(self.cfg.retrieval.rerank_top_k)
        self.enable_decomp = bool(self.cfg.cgal.enable_query_decomposition)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        stream: bool = False,
    ) -> QueryResponse | AsyncIterator[dict[str, Any]]:
        """Run the full CGAL pipeline for a single query.

        Parameters
        ----------
        query : str
            The raw user query.
        stream : bool
            When True, returns an async generator yielding stream events
            of the form ``{"type": "token" | "meta" | "citation" | "done",
            "data": ...}``.  When False, returns a fully-populated
            :class:`QueryResponse`.

        Returns
        -------
        QueryResponse or AsyncIterator[dict]
        """
        if stream:
            return self._run_streaming(query)
        return self._run_blocking(query)

    # ------------------------------------------------------------------
    # Blocking path
    # ------------------------------------------------------------------

    def _run_blocking(self, query: str) -> QueryResponse:
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
            logger.info(
                "Decomposed query into %d sub-queries.", len(sub_queries)
            )
            sub_responses: list[QueryResponse] = []
            for sq in sub_queries:
                sub_responses.append(self._run_single(sq, session_id))
            merger = getattr(self.decomposer, "merger", None)
            if merger is None:
                raise RuntimeError(
                    "Decomposer is missing a 'merger'; cannot combine "
                    "sub-query responses."
                )
            merged = merger.merge(sub_responses, query)
            merged.decomposed = True
            merged.sub_queries = sub_queries
            merged.session_id = session_id
            merged.latency_ms = (time.perf_counter() - t_start) * 1000.0
            return merged

        response = self._run_single(query, session_id)
        response.decomposed = decomposed
        response.sub_queries = sub_queries
        response.latency_ms = (time.perf_counter() - t_start) * 1000.0
        return response

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    async def _run_streaming(
        self, query: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Async generator form used by FastAPI SSE endpoints.

        Emits ``meta`` first (confidence + tool decisions), then ``token``
        events from the generator, optional ``citation`` events, a
        ``verify`` event once the async verify task resolves, and finally
        a ``done`` event carrying the fully-assembled :class:`QueryResponse`
        dict.
        """
        t_start = time.perf_counter()
        session_id = str(uuid.uuid4())

        # Run the non-streaming CGAL loop to resolve retrieval + routing.
        # Generation is re-played in streaming mode if the generator
        # exposes ``stream(...)``.
        response = await asyncio.to_thread(self._run_single, query, session_id)

        yield {
            "type": "meta",
            "data": {
                "confidence": response.confidence,
                "cgal_iterations": response.cgal_iterations,
                "alpha": response.alpha,
                "session_id": session_id,
            },
        }

        # Stream tokens if the generator supports it, otherwise yield the
        # resolved answer as a single token chunk.
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

        # Kick off async verify if we are in the medium-confidence band.
        verify_task: asyncio.Task | None = None
        if (
            self.answer_verify is not None
            and self.med_conf <= response.confidence < self.high_conf
            and response.ticket_id is None
        ):
            verify_task = asyncio.create_task(
                _run_verify_async(
                    self.answer_verify, response.answer, response.citations
                )
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

    def _run_single(self, query: str, session_id: str) -> QueryResponse:
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

        for it in range(self.max_iterations):
            state = _IterationState(
                iteration=it,
                query=query,
                refined_query=refined_query,
            )

            # ---- encode + alpha ------------------------------------------------
            query_emb = self._encode_query(refined_query)
            alpha = self._predict_alpha(refined_query, query_emb)
            state.alpha = alpha

            # ---- hybrid retrieval ---------------------------------------------
            t_ret = time.perf_counter()
            retrieved = self.retriever.retrieve(
                query=refined_query,
                top_k=self.top_k,
                alpha=alpha,
            )
            # Deduplicate against chunks seen in prior iterations.
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
                logger.info(
                    "Iteration %d: no novel retrieval candidates; escalating.", it
                )
                return self._escalate(query, response, reason="no_candidates")

            # ---- rerank --------------------------------------------------------
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

            # ---- score confidence + tool policy --------------------------------
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

            # ---- route ---------------------------------------------------------
            if conf >= self.high_conf:
                state.chosen_tool = TOOL_ANSWER_DIRECT
                return self._finalize_answer(
                    query=query,
                    refined_query=refined_query,
                    state=state,
                    response=response,
                    run_verify=False,
                    confidence_before=prev_conf,
                )

            if conf >= self.med_conf:
                state.chosen_tool = TOOL_ANSWER_DIRECT
                return self._finalize_answer(
                    query=query,
                    refined_query=refined_query,
                    state=state,
                    response=response,
                    run_verify=True,
                    confidence_before=prev_conf,
                )

            if conf >= self.low_conf:
                chosen = self._pick_mid_tier_tool(tool_probs)
                state.chosen_tool = chosen
                tool_call_start = time.perf_counter()
                try:
                    tool_result = self.tool_executor.execute(
                        chosen,
                        {"query": refined_query, "iteration": it},
                    )
                except Exception as exc:
                    logger.warning("Tool %s failed: %s", chosen, exc)
                    tool_result = {"error": str(exc)}
                tool_latency_ms = (time.perf_counter() - tool_call_start) * 1000.0
                response.tool_calls.append(
                    ToolCall(
                        tool_name=chosen,  # type: ignore[arg-type]
                        args={"query": refined_query, "iteration": it},
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

            # conf < low_conf
            state.chosen_tool = TOOL_CREATE_TICKET
            last_state = state
            return self._escalate(query, response, reason="low_confidence")

        # Exhausted iterations without high/medium confidence -> escalate.
        logger.info("Exhausted %d CGAL iterations; escalating.", self.max_iterations)
        return self._escalate(query, response, reason="max_iterations")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_decompose(self, query: str) -> bool:
        if self.decomposer is None:
            return False
        if not hasattr(self.decomposer, "is_multi_part"):
            return False
        try:
            result = self.decomposer.is_multi_part(query)
        except Exception as exc:
            logger.warning("Decomposer.is_multi_part failed: %s", exc)
            return False
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode the query via the retriever's dense model.

        Falls back to a zero vector if the retriever does not expose an
        encoder.  The confidence head will still function -- with a
        degraded signal -- in that case.
        """
        vector_store = getattr(self.retriever, "vector_store", None)
        model = getattr(vector_store, "model", None) if vector_store else None
        if model is None or not hasattr(model, "encode"):
            dim = int(self.cfg.models.retriever.embedding_dim)
            logger.debug("Retriever has no encoder; using zero query embedding.")
            return np.zeros(dim, dtype=np.float32)
        emb = model.encode([query], normalize_embeddings=True)
        return np.asarray(emb[0], dtype=np.float32)

    def _predict_alpha(self, query: str, query_emb: np.ndarray) -> float:
        if self.alpha_network is None:
            return float(self.cfg.retrieval.initial_alpha)
        try:
            return float(
                self.alpha_network.predict_alpha(query, query_emb, domain="")
            )
        except Exception as exc:
            logger.warning(
                "AlphaNetwork.predict_alpha failed, using default: %s", exc
            )
            return float(self.cfg.retrieval.initial_alpha)

    def _score_confidence(
        self,
        query_emb: np.ndarray,
        reranked: list[tuple[ChunkRecord, float]],
    ) -> tuple[float, list[float]]:
        """Produce (confidence, tool_probs) for the current top-k evidence."""
        evidence_embs = self._embed_evidence([c for c, _ in reranked])
        try:
            conf, tool_probs = self.confidence_head.score(
                query_emb=query_emb, evidence_embs=evidence_embs
            )
        except Exception as exc:
            logger.warning(
                "ConfidenceHead.score failed: %s; defaulting to low confidence.",
                exc,
            )
            return 0.0, [0.25, 0.25, 0.25, 0.25]
        return float(conf), [float(p) for p in tool_probs]

    def _embed_evidence(self, chunks: list[ChunkRecord]) -> np.ndarray:
        """Embed retrieved chunks for the confidence head.

        Re-uses the retriever's dense model if available.  Falls back to a
        zero matrix when no encoder is reachable.
        """
        if not chunks:
            dim = int(self.cfg.models.retriever.embedding_dim)
            return np.zeros((0, dim), dtype=np.float32)
        vector_store = getattr(self.retriever, "vector_store", None)
        model = getattr(vector_store, "model", None) if vector_store else None
        if model is None or not hasattr(model, "encode"):
            dim = int(self.cfg.models.retriever.embedding_dim)
            return np.zeros((len(chunks), dim), dtype=np.float32)
        texts = [c.text for c in chunks]
        embs = model.encode(texts, normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)

    def _pick_mid_tier_tool(self, tool_probs: list[float]) -> str:
        """Select SearchKB or GetPolicy based on highest tool-policy prob.

        The confidence head emits a 4-way distribution in the order
        ``[AnswerDirect, SearchKB, GetPolicy, CreateTicket]``.  In the mid-tier
        we restrict the choice to SearchKB vs GetPolicy.
        """
        if len(tool_probs) < 4:
            return TOOL_SEARCH_KB
        search_p = tool_probs[1]
        policy_p = tool_probs[2]
        return TOOL_GET_POLICY if policy_p > search_p else TOOL_SEARCH_KB

    def _refine_query(self, query: str, visited_topics: Iterable[str]) -> str:
        """Append a negative constraint to steer retrieval to new material."""
        topics = [t for t in visited_topics if t]
        if not topics:
            return query
        # Limit to 5 topics to keep the refined query short.
        topic_str = "; ".join(sorted(set(topics))[:5])
        return f"{query}\nNOT about: {topic_str}"

    # ------------------------------------------------------------------
    # Answer finalization / escalation
    # ------------------------------------------------------------------

    def _finalize_answer(
        self,
        query: str,
        refined_query: str,
        state: _IterationState,
        response: QueryResponse,
        run_verify: bool,
        confidence_before: float | None,
    ) -> QueryResponse:
        """Run the generator, attach citations, and (optionally) verify."""
        t_gen = time.perf_counter()
        contexts = [
            RetrievalResult(
                chunk=c,
                score=s,
                rerank_score=s,
            )
            for c, s in state.reranked
        ]
        try:
            answer = self.generator.generate(query=refined_query, context=contexts)
        except TypeError:
            # Generator signatures vary; try the simpler positional form.
            answer = self.generator.generate(refined_query, contexts)
        gen_latency_ms = (time.perf_counter() - t_gen) * 1000.0

        if not isinstance(answer, str):
            answer = str(answer)

        citations = _build_citations(state.reranked)

        response.answer = answer
        response.confidence = state.confidence
        response.cgal_iterations = state.iteration + 1
        response.alpha = state.alpha
        response.citations = citations
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
                        result=verdict if isinstance(verdict, dict) else {"verdict": str(verdict)},
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
        """Create a support ticket and return the escalation response."""
        t_t = time.perf_counter()
        try:
            result = self.tool_executor.execute(
                TOOL_CREATE_TICKET,
                {"query": query, "reason": reason, "session_id": response.session_id},
            )
        except Exception as exc:
            logger.error("CreateTicket tool failed: %s", exc)
            result = {"error": str(exc)}
        latency_ms = (time.perf_counter() - t_t) * 1000.0

        ticket_id = None
        if isinstance(result, dict):
            ticket_id = result.get("ticket_id")

        response.ticket_id = ticket_id
        message: str | None = None
        if isinstance(result, dict):
            raw_msg = result.get("message")
            if isinstance(raw_msg, str) and raw_msg:
                message = raw_msg
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


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _build_citations(
    reranked: list[tuple[ChunkRecord, float]],
) -> list[Citation]:
    """Convert reranked chunks into lightweight :class:`Citation` objects."""
    citations: list[Citation] = []
    for chunk, _score in reranked:
        cited_text = chunk.text.strip()
        if len(cited_text) > 500:
            cited_text = cited_text[:497] + "..."
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
    """Best-effort async invocation of the NLI verifier."""
    fn = getattr(verifier, "verify_async", None)
    if fn is not None:
        result = await fn(answer, citations)
    else:
        result = await asyncio.to_thread(verifier.verify, answer, citations)
    if not isinstance(result, dict):
        return {"verdict": str(result)}
    return result


async def _ensure_async_iter(obj: Any) -> AsyncIterator[str]:
    """Yield string tokens from either an async or sync iterable."""
    if hasattr(obj, "__aiter__"):
        async for tok in obj:
            yield str(tok)
        return
    if hasattr(obj, "__iter__"):
        for tok in obj:
            yield str(tok)
        return
    yield str(obj)
