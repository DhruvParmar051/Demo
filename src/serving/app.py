"""
FastAPI application factory for AegisRAG.

Endpoints:

    POST /query             -- synchronous, returns full QueryResponse JSON
    POST /query/stream      -- SSE streaming
    POST /query/baseline    -- single-pass baseline (b1/b2/b3)
    GET  /health            -- health + device info
    GET  /tickets           -- escalation tickets from the audit log
    GET  /metrics           -- Prometheus-format metrics
"""

from __future__ import annotations

import asyncio
import logging
import time
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
    """Build and return the FastAPI application.

    Parameters
    ----------
    config : object, optional
        Parsed config (e.g. from :func:`get_config`). If ``None``, falls back
        to ``get_config()``.
    model_tag : str, optional
        Default model tag to warm on startup. If provided, the registry will
        eagerly build this pipeline on the first request path rather than
        lazy-building. Accepted values: ``b1``, ``b2``, ``b3``, ``m1``..``m5``.
    """
    try:
        from fastapi import FastAPI, HTTPException, Query, Request  # type: ignore
        from fastapi.middleware.cors import CORSMiddleware  # type: ignore
        from fastapi.responses import JSONResponse, PlainTextResponse  # type: ignore
        from pydantic import BaseModel, Field  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "fastapi and pydantic are required to run the server."
        ) from exc

    try:
        from sse_starlette.sse import EventSourceResponse  # type: ignore
    except ImportError:
        EventSourceResponse = None  # type: ignore

    cfg = config if config is not None else get_config()
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

    default_tag = (model_tag or "m5").lower()

    class QueryRequest(BaseModel):
        query: str = Field(..., min_length=1, max_length=8192)
        model_tag: str = Field(default_tag)

    # ---------------- health ----------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "device": get_device_string(),
            "models_cached": list(registry._cache.keys()),
        }

    # ---------------- /query ----------------------------------------------

    @app.post("/query")
    async def query(req: QueryRequest) -> dict[str, Any]:
        pipeline = await registry.get(req.model_tag)
        t_start = time.perf_counter()
        try:
            response: QueryResponse = await asyncio.to_thread(
                pipeline.run, req.query
            )
        except Exception as exc:
            logger.exception("query failed")
            raise HTTPException(status_code=500, detail=str(exc))
        if response.latency_ms == 0.0:
            response.latency_ms = (time.perf_counter() - t_start) * 1000.0
        await asyncio.to_thread(audit.log, response, req.model_tag, req.query)
        return response.to_dict()

    # ---------------- /query/baseline --------------------------------------

    @app.post("/query/baseline")
    async def query_baseline(
        req: QueryRequest,
        baseline: str = Query("b1", pattern="^b[1-3]$"),
    ) -> dict[str, Any]:
        pipeline = await registry.get(baseline)
        response: QueryResponse = await asyncio.to_thread(pipeline.run, req.query)
        await asyncio.to_thread(audit.log, response, baseline, req.query)
        return response.to_dict()

    # ---------------- /query/stream ---------------------------------------

    @app.post("/query/stream")
    async def query_stream(req: QueryRequest):
        if EventSourceResponse is None:
            raise HTTPException(
                status_code=500, detail="sse-starlette not installed"
            )

        pipeline = await registry.get(req.model_tag)
        engine = getattr(pipeline, "engine", None) or pipeline
        async def _gen():
            t_start = time.perf_counter()
            final: QueryResponse | None = None
            verify_emitted = False
            try:
                # Prefer engine's async streamer if present.
                stream_fn = getattr(engine, "_run_streaming", None) or getattr(
                    engine, "run_streaming", None
                )
                if stream_fn is not None:
                    agen = stream_fn(req.query)
                    # If stream_fn is an async function (not an async
                    # generator), calling it returns a coroutine we must
                    # await to get the actual async iterator.
                    if asyncio.iscoroutine(agen):
                        agen = await agen
                    async for ev in agen:
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
                    response: QueryResponse = await asyncio.to_thread(
                        pipeline.run, req.query
                    )
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
                if final is not None:
                    if final.latency_ms == 0.0:
                        final.latency_ms = (time.perf_counter() - t_start) * 1000.0
                    await asyncio.to_thread(audit.log, final,
                                             req.model_tag, req.query)

        return EventSourceResponse(_gen(), media_type="text/event-stream")

    # ---------------- /tickets --------------------------------------------

    @app.get("/tickets")
    async def tickets() -> dict[str, Any]:
        return {"tickets": audit.list_tickets()}

    # ---------------- /metrics --------------------------------------------

    @app.get("/metrics")
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

    # ---------------- shutdown hook ---------------------------------------

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        audit.close()

    app.include_router(build_ingest_router(config))


    return app


# Uvicorn entrypoint: ``uvicorn src.serving.app:app``.
app = None  # populated by run.py
