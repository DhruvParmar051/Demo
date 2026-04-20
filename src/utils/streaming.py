"""
Streaming utilities for AegisRAG serving layer.

Provides an async token generator that wraps model.generate() to yield
tokens one at a time, a StreamEvent dataclass for structured event types,
and SSE (Server-Sent Events) conversion helpers.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

import torch


class EventType(str, Enum):
    """Types of events emitted during streaming generation."""
    TOKEN = "token"
    CITATION = "citation"
    TOOL_CALL = "tool_call"
    VERIFY_START = "verify_start"
    VERIFY_RESULT = "verify_result"
    DONE = "done"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamEvent:
    """
    A single event in the streaming response.

    Attributes:
        event_type: The kind of event (token, citation, etc.).
        data: The payload. For TOKEN events this is the token string;
              for CITATION events it is a dict with citation info; etc.
        timestamp: Unix timestamp of event creation.
        sequence: Monotonically increasing sequence number within a stream.
    """
    event_type: EventType
    data: Any
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }

    def to_sse(self) -> str:
        """
        Format as a Server-Sent Events message.

        Returns a string like:
            event: token
            data: {"text": "Hello"}

        with a trailing blank line as per the SSE spec.
        """
        event_line = f"event: {self.event_type.value}"
        if isinstance(self.data, str):
            data_payload = json.dumps({"text": self.data})
        elif isinstance(self.data, dict):
            data_payload = json.dumps(self.data)
        else:
            data_payload = json.dumps(self.data)
        data_line = f"data: {data_payload}"
        return f"{event_line}\n{data_line}\n\n"


def make_token_event(token_text: str, seq: int) -> StreamEvent:
    """Create a TOKEN stream event."""
    return StreamEvent(event_type=EventType.TOKEN, data=token_text, sequence=seq)


def make_citation_event(
    doc_id: str,
    span_start: int,
    span_end: int,
    cited_text: str,
    verified: bool,
    seq: int,
) -> StreamEvent:
    """Create a CITATION stream event."""
    return StreamEvent(
        event_type=EventType.CITATION,
        data={
            "doc_id": doc_id,
            "span_start": span_start,
            "span_end": span_end,
            "cited_text": cited_text,
            "verified": verified,
        },
        sequence=seq,
    )


def make_tool_call_event(
    tool_name: str,
    args: Dict[str, Any],
    result: Any,
    latency_ms: float,
    seq: int,
) -> StreamEvent:
    """Create a TOOL_CALL stream event."""
    return StreamEvent(
        event_type=EventType.TOOL_CALL,
        data={
            "tool_name": tool_name,
            "args": args,
            "result": result,
            "latency_ms": latency_ms,
        },
        sequence=seq,
    )


def make_done_event(seq: int, metadata: Optional[Dict[str, Any]] = None) -> StreamEvent:
    """Create a DONE stream event."""
    return StreamEvent(
        event_type=EventType.DONE,
        data=metadata or {},
        sequence=seq,
    )


def make_heartbeat_event(seq: int) -> StreamEvent:
    """Create a HEARTBEAT keep-alive event."""
    return StreamEvent(
        event_type=EventType.HEARTBEAT,
        data="",
        sequence=seq,
    )


class AsyncTokenGenerator:
    """
    Wraps a HuggingFace model to yield tokens one at a time asynchronously.

    This uses a TextIteratorStreamer from transformers to bridge the
    synchronous model.generate() call into an async iterator.

    Usage::

        gen = AsyncTokenGenerator(model, tokenizer)
        async for event in gen.stream(prompt_ids, max_new_tokens=256):
            print(event.to_sse())
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    async def stream(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stop_strings: Optional[List[str]] = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Generate tokens from input_ids and yield StreamEvent objects.

        Args:
            input_ids: Tokenized prompt tensor of shape (1, seq_len).
            attention_mask: Optional attention mask.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling.
            repetition_penalty: Repetition penalty factor.
            do_sample: Whether to sample (vs greedy).
            stop_strings: Optional list of strings that stop generation.

        Yields:
            StreamEvent objects (TOKEN events, then a DONE event).
        """
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        input_ids = input_ids.to(self.device)
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "top_k": top_k if do_sample else 0,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "streamer": streamer,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask.to(self.device)

        # Run generation in a background thread so we can async-iterate
        loop = asyncio.get_running_loop()
        generate_task = loop.run_in_executor(
            None,
            lambda: self.model.generate(**generate_kwargs),
        )

        seq = 0
        accumulated = ""
        try:
            for text_chunk in streamer:
                # TextIteratorStreamer blocks until a new chunk is ready;
                # we yield control between chunks.
                accumulated += text_chunk

                # Check for stop strings
                if stop_strings:
                    should_stop = False
                    for stop_str in stop_strings:
                        if stop_str in accumulated:
                            # Yield only up to the stop string
                            idx = accumulated.index(stop_str)
                            final_text = accumulated[:idx]
                            if final_text:
                                yield make_token_event(final_text, seq)
                                seq += 1
                            should_stop = True
                            break
                    if should_stop:
                        break

                if text_chunk:
                    yield make_token_event(text_chunk, seq)
                    seq += 1

                # Yield control to the event loop
                await asyncio.sleep(0)
        except Exception as e:
            yield StreamEvent(
                event_type=EventType.ERROR,
                data={"error": str(e)},
                sequence=seq,
            )
            seq += 1

        # Ensure generation is complete
        await generate_task

        yield make_done_event(
            seq=seq,
            metadata={
                "total_tokens": seq,
                "accumulated_length": len(accumulated),
            },
        )


# ---------------------------------------------------------------------------
# SSE conversion utilities
# ---------------------------------------------------------------------------

def events_to_sse_bytes(events: List[StreamEvent]) -> bytes:
    """Convert a list of StreamEvents to SSE-formatted bytes."""
    return "".join(e.to_sse() for e in events).encode("utf-8")


def sse_comment(text: str) -> str:
    """Format an SSE comment line (used for heartbeats/debugging)."""
    return f": {text}\n\n"
