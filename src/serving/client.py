"""In-process query client for the CLI ``query`` subcommand.

Avoids the network trip: builds the requested pipeline directly and
calls it, returning either the full ``QueryResponse`` as a dict
(non-streaming) or a generator of text chunks (streaming). This keeps
``python run.py query`` usable without a running server.
"""

from __future__ import annotations

import logging
from typing import Any, Generator

logger = logging.getLogger(__name__)


def _build_pipeline(model_tag: str, cfg: Any) -> Any:
    tag = model_tag.lower()
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
    return M5Pipeline.from_tag(tag, cfg)


def _response_to_dict(resp: Any) -> dict[str, Any]:
    """Normalise any pipeline output into a plain dict."""
    if isinstance(resp, dict):
        return resp
    to_dict = getattr(resp, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
        if isinstance(out, dict):
            return out
    # Last-resort: pick a few well-known attributes.
    return {
        "answer": getattr(resp, "answer", str(resp)),
        "confidence": getattr(resp, "confidence", 0.0),
        "sources": [],
    }


def query_model(
    model_tag: str,
    query: str,
    stream: bool = False,
    config: Any | None = None,
) -> Any:
    """Answer ``query`` using the pipeline indicated by ``model_tag``.

    When ``stream=True``, returns a generator yielding answer chunks
    (or the full answer as a single chunk for pipelines without native
    streaming). Otherwise returns the response dict.
    """
    pipeline = _build_pipeline(model_tag, config)

    if stream:
        return _stream_pipeline(pipeline, query)

    run = getattr(pipeline, "run", None)
    if not callable(run):
        raise TypeError(f"Pipeline {model_tag} has no .run() method")
    response = run(query)
    return _response_to_dict(response)


def _stream_pipeline(pipeline: Any, query: str) -> Generator[str, None, None]:
    """Yield incremental text chunks for a pipeline.

    Falls back to a single-chunk yield for pipelines that do not
    expose a streaming interface.
    """
    # Prefer an explicit streaming method on the pipeline or its
    # inner CGAL engine.
    engine = getattr(pipeline, "engine", None) or pipeline
    stream_fn = getattr(engine, "run_streaming", None) or getattr(
        engine, "_run_streaming", None
    )

    if stream_fn is not None:
        try:
            for ev in stream_fn(query):
                if isinstance(ev, dict):
                    if ev.get("type") == "token":
                        data = ev.get("data")
                        if isinstance(data, str):
                            yield data
                    elif ev.get("type") == "done":
                        final = ev.get("data") or {}
                        if isinstance(final, dict) and final.get("answer"):
                            # Nothing to yield, already streamed tokens.
                            pass
                else:
                    yield str(ev)
            return
        except Exception as exc:
            logger.warning("Streaming failed, falling back to .run(): %s", exc)

    run = getattr(pipeline, "run", None)
    if not callable(run):
        raise TypeError("Pipeline does not support .run() or streaming")
    response = run(query)
    answer = getattr(response, "answer", None) or _response_to_dict(response).get(
        "answer", ""
    )
    yield str(answer)
