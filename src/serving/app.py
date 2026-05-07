"""
FastAPI application factory for AegisRAG.

Endpoints:

    POST /query             -- synchronous, returns full QueryResponse JSON
    POST /query/stream      -- SSE streaming
    POST /query/baseline    -- single-pass baseline (b1/b2/b3)
    GET  /health            -- health + device info
    GET  /tickets           -- escalation tickets from the audit log
    GET  /metrics           -- Prometheus-format metrics

SSE streaming emits periodic heartbeat comments to prevent client-side
timeout. The interval is configurable via ``cfg.serving.sse_heartbeat_interval``
(default 15 seconds).
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from src.data.schema import QueryResponse
from src.serving.audit_logger import AuditLogger
from src.serving.sse import (
    EVENT_CITATION,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_TOKEN,
    EVENT_TOOL_CALL,
    EVENT_VERIFY_RESULT,
    EVENT_VERIFY_START,
    format_sse,
)
from src.utils.config import get_config
from src.utils.device import get_device_string
from src.serving.ingest_router import build_ingest_router
from loguru import logger

# Imported at module level so Pydantic v2 can resolve forward references in
# request models without needing the global namespace trick.
try:
    from fastapi import APIRouter, FastAPI, HTTPException, Query, Request  # type: ignore
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore
    from fastapi.responses import PlainTextResponse  # type: ignore
    from pydantic import BaseModel, ConfigDict, Field  # type: ignore

    class QueryRequest(BaseModel):
        model_config = ConfigDict(protected_namespaces=())

        query: str = Field(..., min_length=1, max_length=8192)
        model_tag: str = Field("m5")
        conversation_history: list[dict[str, str]] = Field(
            default_factory=list,
            description="Prior turns as [{'role': 'user'|'assistant', 'content': '...'}]. "
            "Injected into the generator prompt before the current query.",
        )

except ImportError:
    # fastapi not installed — create_app will raise a clear error at call time.
    QueryRequest = None  # type: ignore


# ----------------------------------------------------------------------
# Pipeline registry (lazy)
# ----------------------------------------------------------------------


class PipelineRegistry:
    """Lazily builds and caches pipelines by tag."""

    def __init__(self, config: Any) -> None:
        self.cfg = config
        self._cache: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, tag: str) -> Any:
        tag = tag.lower()
        if tag in self._cache:
            return self._cache[tag]
        async with self._lock:
            if tag in self._cache:
                return self._cache[tag]
            self._cache[tag] = await asyncio.to_thread(self._build, tag)
            return self._cache[tag]

    def _build(self, tag: str) -> Any:
        if tag == "b1":
            from src.models.baselines import BaselineB1
            return BaselineB1()
        if tag == "b2":
            from src.models.baselines import BaselineB2
            return BaselineB2()
        if tag == "b3":
            from src.models.baselines import BaselineB3
            return BaselineB3()
        from src.models.m5_pipeline import M5Pipeline
        return M5Pipeline.from_tag(tag, self.cfg)


# ----------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------


def create_app(config: Any = None, model_tag: str | None = None) -> Any:
    """Build and return the FastAPI application."""
    if QueryRequest is None:
        raise RuntimeError(
            "fastapi and pydantic are required to run the server."
        )

    try:
        from sse_starlette.sse import EventSourceResponse  # type: ignore
    except ImportError:
        EventSourceResponse = None  # type: ignore

    cfg = config if config is not None else get_config()
    api_prefix = getattr(cfg.serving, "api_prefix", "").rstrip("/")  # e.g. "/api/v1"
    app = FastAPI(title="AegisRAG", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://localhost:3000",
                       "http://localhost"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    audit_db = Path(getattr(cfg.paths, "audit_db", "data/audit.sqlite"))
    audit = AuditLogger(audit_db)
    registry = PipelineRegistry(cfg)

    # Bounded LRU cache for exact-query responses (cuts latency on repeated queries).
    _resp_cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
    _resp_cache_max: int = 512

    def _cache_get(query: str, tag: str) -> dict | None:
        key = (query.strip().lower(), tag.lower())
        return _resp_cache.get(key)

    def _cache_put(query: str, tag: str, resp: dict) -> None:
        key = (query.strip().lower(), tag.lower())
        _resp_cache[key] = resp
        if len(_resp_cache) > _resp_cache_max:
            _resp_cache.popitem(last=False)

    # Pre-build the default pipeline synchronously in the main thread so that
    # llama-cpp-python (GGUF backend) is never initialised from a thread-pool
    # executor. We also force _ensure_loaded() so the Llama object exists before
    # uvicorn begins accepting requests.
    if model_tag:
        _tag = model_tag.lower()
        logger.info("Pre-loading pipeline '%s' in main thread ...", _tag)
        try:
            _pipeline = registry._build(_tag)
            registry._cache[_tag] = _pipeline
            # Force the generator backend (GGUF/HF) to fully initialize now.
            _gen = getattr(_pipeline, "generator", None)
            if _gen is not None and hasattr(_gen, "_ensure_loaded"):
                logger.info("Loading generator backend in main thread ...")
                _gen._ensure_loaded()
                logger.info("Generator ready (backend=%s).", _gen.backend)
            logger.info("Pipeline '%s' ready.", _tag)
        except Exception as _exc:
            logger.warning("Pipeline pre-load failed (%s); will retry on first request.", _exc)

    sse_heartbeat_interval: float = float(
        getattr(cfg.serving, "sse_heartbeat_interval", 15)
    )

    # All routes are registered on this router so they share the /api/v1 prefix.
    router = APIRouter(prefix=api_prefix)

    # ---------------- health ----------------------------------------------

    @router.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "device": get_device_string(),
            "models_cached": list(registry._cache.keys()),
        }

    # ---------------- /query ----------------------------------------------

    @router.post("/query")
    async def query(req: QueryRequest) -> dict[str, Any]:
        cached = _cache_get(req.query, req.model_tag)
        if cached is not None:
            logger.debug("query cache hit for model=%s", req.model_tag)
            return cached

        pipeline = await registry.get(req.model_tag)
        t_start = time.perf_counter()
        history = req.conversation_history or None
        try:
            response: QueryResponse = await asyncio.to_thread(
                pipeline.run, req.query, False, history
            )
        except Exception as exc:
            logger.exception("query failed")
            raise HTTPException(status_code=500, detail=str(exc))
        if response.latency_ms == 0.0:
            response.latency_ms = (time.perf_counter() - t_start) * 1000.0
        await asyncio.to_thread(audit.log, response, req.model_tag, req.query)
        result = response.to_dict()
        _cache_put(req.query, req.model_tag, result)
        return result

    # ---------------- /query/baseline --------------------------------------

    @router.post("/query/baseline")
    async def query_baseline(
        req: QueryRequest,
        baseline: str = Query("b1", pattern="^b[1-3]$"),
    ) -> dict[str, Any]:
        pipeline = await registry.get(baseline)
        response: QueryResponse = await asyncio.to_thread(pipeline.run, req.query)
        await asyncio.to_thread(audit.log, response, baseline, req.query)
        return response.to_dict()

    # ---------------- /query/stream ---------------------------------------

    @router.post("/query/stream")
    async def query_stream(req: QueryRequest):
        if EventSourceResponse is None:
            raise HTTPException(
                status_code=500, detail="sse-starlette not installed"
            )

        pipeline = await registry.get(req.model_tag)
        engine = getattr(pipeline, "engine", None) or pipeline

        # A background task emits SSE comment lines while the generator runs
        # so the client connection stays alive during long inference passes.
        heartbeat_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _heartbeat_sender() -> None:
            """Emit SSE comment lines at a fixed interval to keep the connection alive."""
            while True:
                await asyncio.sleep(sse_heartbeat_interval)
                await heartbeat_queue.put(": heartbeat\n\n")

        async def _gen():
            t_start = time.perf_counter()
            final: QueryResponse | None = None
            verify_emitted = False

            heartbeat_task = asyncio.create_task(_heartbeat_sender())

            try:
                stream_fn = (
                    getattr(engine, "_run_streaming", None)
                    or getattr(engine, "run_streaming", None)
                )
                if stream_fn is not None:
                    agen = stream_fn(req.query, history=req.conversation_history or None)
                    if asyncio.iscoroutine(agen):
                        agen = await agen

                    async def _drain_agen():
                        async for ev in agen:
                            yield ev

                    # Flush any pending heartbeats before each real event.
                    async for ev in _drain_agen():
                        while not heartbeat_queue.empty():
                            hb = heartbeat_queue.get_nowait()
                            if hb:
                                yield hb

                        etype = ev.get("type", "token")
                        data = ev.get("data")
                        if etype == "token":
                            yield format_sse(EVENT_TOKEN, {"text": data})
                        elif etype == "citation":
                            yield format_sse(EVENT_CITATION, data)
                        elif etype == "tool_call":
                            yield format_sse(EVENT_TOOL_CALL, data)
                        elif etype == "verify":
                            if not verify_emitted:
                                yield format_sse(EVENT_VERIFY_START, {})
                                verify_emitted = True
                            yield format_sse(EVENT_VERIFY_RESULT, data)
                        elif etype == "done":
                            final_dict = data if isinstance(data, dict) else {}
                            yield format_sse(EVENT_DONE, final_dict)
                else:
                    # Non-streaming fallback: poll and emit heartbeats while waiting.
                    response_task = asyncio.create_task(
                        asyncio.to_thread(pipeline.run, req.query)
                    )
                    while not response_task.done():
                        await asyncio.sleep(min(1.0, sse_heartbeat_interval))
                        yield f": heartbeat\n\n"
                    response: QueryResponse = await response_task
                    final = response
                    yield format_sse(EVENT_TOKEN, {"text": response.answer})
                    for c in response.citations:
                        yield format_sse(EVENT_CITATION, c.to_dict())
                    yield format_sse(EVENT_DONE, response.to_dict())

            except Exception as exc:
                logger.exception("stream failed")
                yield format_sse(EVENT_ERROR, {"message": str(exc),
                                                "type": exc.__class__.__name__})
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

                if final is not None:
                    if final.latency_ms == 0.0:
                        final.latency_ms = (time.perf_counter() - t_start) * 1000.0
                    await asyncio.to_thread(audit.log, final,
                                             req.model_tag, req.query)

        return EventSourceResponse(_gen(), media_type="text/event-stream")

    # ---------------- /query/user_docs ------------------------------------

    @router.post("/query/user_docs")
    async def query_user_docs(
        req: QueryRequest,
        collection_id: str = Query(..., min_length=1, max_length=128),
    ) -> dict[str, Any]:
        """Answer a query using only a user-specific document collection.

        The collection is created by uploading documents with the
        ``X-Collection-ID`` header on the ``/ingest`` endpoint.  This
        endpoint does dense-only retrieval so no BM25 index is required.
        """
        from src.retrieval.vector_store import ChromaVectorStore
        from src.data.schema import Citation, RetrievalResult

        try:
            user_vs = await asyncio.to_thread(ChromaVectorStore, collection_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Collection error: {exc}")

        try:
            results = await asyncio.to_thread(user_vs.query, req.query, 5)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")

        # Filter low-similarity results — below 0.30 cosine means no real match
        MIN_SIMILARITY = 0.30
        if results:
            logger.info("user_docs scores: {}", [round(s, 3) for _, s in results])
        results = [(c, s) for c, s in results if s >= MIN_SIMILARITY]

        def _create_ticket(query: str) -> dict:
            from src.tools.executor import ToolExecutor
            from pathlib import Path as _Path
            _ticket_db = _Path(cfg.data.audit_db_path).parent / "tickets.db"
            _executor = ToolExecutor(retriever=None, ticket_store_path=_ticket_db)
            return _executor.create_ticket(
                query=query,
                summary=f"Out-of-scope query: {query[:200]}",
                category="other",
                severity="medium",
            )

        if not results:
            ticket_result = await asyncio.to_thread(_create_ticket, req.query)
            escalation = QueryResponse(
                answer=(
                    "This question is outside the scope of your uploaded documents. "
                    f"A support ticket has been created (ID: {ticket_result['ticket_id']}). "
                    f"A specialist will follow up within {ticket_result['estimated_response_time']}."
                ),
                citations=[],
                confidence=0.0,
                ticket_id=ticket_result["ticket_id"],
            )
            await asyncio.to_thread(audit.log, escalation, req.model_tag, req.query)
            return escalation.to_dict()

        # Reuse the already-loaded M5 generator — no extra model loading.
        pipeline = await registry.get("m5")
        engine = getattr(pipeline, "engine", pipeline)
        generator = getattr(engine, "generator", None)
        if generator is None:
            raise HTTPException(status_code=500, detail="Generator not available")

        contexts = [
            RetrievalResult(chunk=c, score=s, rerank_score=s) for c, s in results
        ]
        try:
            answer = await asyncio.to_thread(
                lambda: generator.generate(query=req.query, context=contexts)
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

        # Detect out-of-scope answers — model said it can't answer from context
        _OUT_OF_SCOPE_PHRASES = (
            "not directly applicable",
            "not directly related",
            "not applicable to the context",
            "not found in the",
            "no relevant information",
            "cannot answer",
            "outside the scope",
            "not covered in",
            "not mentioned in",
            "not present in",
            "does not contain",
            "no information",
        )
        answer_lower = str(answer).lower()
        is_out_of_scope = any(p in answer_lower for p in _OUT_OF_SCOPE_PHRASES)

        if is_out_of_scope:
            ticket_result = await asyncio.to_thread(_create_ticket, req.query)
            escalation = QueryResponse(
                answer=(
                    "This question is outside the scope of your uploaded documents. "
                    f"A support ticket has been created (ID: {ticket_result['ticket_id']}). "
                    f"A specialist will follow up within {ticket_result['estimated_response_time']}."
                ),
                citations=[],
                confidence=0.0,
                ticket_id=ticket_result["ticket_id"],
            )
            await asyncio.to_thread(audit.log, escalation, req.model_tag, req.query)
            return escalation.to_dict()

        citations = [
            Citation(
                doc_id=c.doc_id,
                chunk_id=c.chunk_id,
                span_start=c.span_start,
                span_end=c.span_end,
                cited_text=c.text[:500],
                source=c.source,
                page_number=c.page_number,
            )
            for c, _ in results
        ]
        response = QueryResponse(
            answer=str(answer),
            citations=citations,
            confidence=0.0,
        )
        await asyncio.to_thread(audit.log, response, req.model_tag, req.query)
        return response.to_dict()

    # ---------------- /tickets --------------------------------------------

    @router.get("/tickets")
    async def tickets() -> dict[str, Any]:
        return {"tickets": audit.list_tickets()}

    # ---------------- /metrics --------------------------------------------

    @router.get("/metrics")
    async def metrics() -> Any:
        s = audit.stats()
        lines = [
            "# HELP aegisrag_queries_total Total served queries.",
            "# TYPE aegisrag_queries_total counter",
            f"aegisrag_queries_total {s['total_queries']}",
            "# HELP aegisrag_escalations_total Escalation tickets created.",
            "# TYPE aegisrag_escalations_total counter",
            f"aegisrag_escalations_total {s['escalations']}",
            "# HELP aegisrag_avg_latency_ms Mean end-to-end latency (ms).",
            "# TYPE aegisrag_avg_latency_ms gauge",
            f"aegisrag_avg_latency_ms {s['avg_latency_ms']:.3f}",
            "# HELP aegisrag_avg_confidence Mean response confidence.",
            "# TYPE aegisrag_avg_confidence gauge",
            f"aegisrag_avg_confidence {s['avg_confidence']:.4f}",
            "# HELP aegisrag_queries_by_tag Count of queries per model tag.",
            "# TYPE aegisrag_queries_by_tag counter",
        ]
        for tag, n in s["queries_by_tag"].items():
            lines.append(f'aegisrag_queries_by_tag{{tag="{tag}"}} {n}')
        return PlainTextResponse("\n".join(lines) + "\n")

    # ---------------- session collection cleanup --------------------------
    # User-upload collections accumulate in ChromaDB indefinitely.
    # Purge collections prefixed "user_" that have not been accessed in
    # SESSION_TTL_HOURS hours to prevent unbounded growth.
    SESSION_TTL_HOURS: int = 2

    @router.delete("/sessions/{collection_id}")
    async def delete_session(collection_id: str) -> dict[str, Any]:
        """Explicitly delete a user session's ChromaDB collection."""
        if not collection_id.startswith("user_"):
            raise HTTPException(status_code=400, detail="Only user_ collections may be deleted.")
        try:
            from src.retrieval.vector_store import ChromaVectorStore
            ChromaVectorStore.delete_collection(collection_id)
            return {"status": "deleted", "collection_id": collection_id}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.on_event("startup")
    async def _start_session_cleanup() -> None:
        """Background task: purge stale user collections every hour."""
        async def _cleanup_loop() -> None:
            while True:
                await asyncio.sleep(SESSION_TTL_HOURS * 3600)
                try:
                    from src.retrieval.vector_store import ChromaVectorStore
                    ChromaVectorStore.purge_stale_user_collections(
                        max_age_hours=SESSION_TTL_HOURS
                    )
                    logger.info("Session collection cleanup completed.")
                except Exception as exc:
                    logger.warning("Session cleanup failed: %s", exc)
        asyncio.create_task(_cleanup_loop())

    # ---------------- shutdown hook ---------------------------------------

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        audit.close()

    app.include_router(router)
    app.include_router(build_ingest_router(config), prefix=api_prefix)

    return app


app = None  # populated by run.py