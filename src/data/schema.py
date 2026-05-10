"""
AegisRAG - Data Schema Definitions

Canonical data structures used throughout the pipeline. Dataclasses keep the
schemas lightweight (no Pydantic runtime overhead in the hot path) while still
providing to_dict/from_dict helpers for JSON serialization and audit logging.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


# ----------------------------------------------------------------------
# Document / retrieval layer
# ----------------------------------------------------------------------


@dataclass
class ChunkRecord:
    """A single chunk of text extracted from a document."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    page_number: int | None = None
    chunk_index: int = 0
    token_count: int = 0
    span_start: int = 0
    span_end: int = 0
    section_title: str = ""
    domain: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @staticmethod
    def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
        raw = f"{doc_id}::chunk::{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def generate_doc_id(source: str) -> str:
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChunkRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RetrievalResult:
    """A scored chunk returned by hybrid retrieval or reranking."""

    chunk: ChunkRecord
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "rerank_score": self.rerank_score,
        }


# ----------------------------------------------------------------------
# Response layer (served to clients and audit log)
# ----------------------------------------------------------------------


@dataclass
class Citation:
    """A pointer to a supporting evidence span referenced by the answer."""

    doc_id: str
    chunk_id: str
    span_start: int
    span_end: int
    cited_text: str
    source: str = ""
    page_number: int | None = None
    source_url: str | None = None
    verified: bool | None = None  # populated by AnswerVerify when it runs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    """Record of a single tool invocation during the CGAL loop."""

    tool_name: Literal[
        "SearchKB", "GetPolicy", "CreateTicket", "AnswerVerify", "AnswerDirect"
    ]
    args: dict[str, Any]
    result: dict[str, Any] | None = None
    latency_ms: float = 0.0
    iteration: int = 0
    confidence_before: float | None = None
    confidence_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryResponse:
    """Top-level response produced by the full pipeline."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    confidence: float = 0.0
    cgal_iterations: int = 0
    fcrs: float | None = None
    ticket_id: str | None = None
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decomposed: bool = False
    sub_queries: list[str] = field(default_factory=list)
    alpha: float | None = None
    verify_verdict: str | None = None
    model_tag: str = ""
    # chunk_ids of ALL retrieved candidates (before max_citations truncation).
    # Used by the evaluator for a true recall@20 measurement.
    retrieved_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "confidence": self.confidence,
            "cgal_iterations": self.cgal_iterations,
            "fcrs": self.fcrs,
            "ticket_id": self.ticket_id,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "session_id": self.session_id,
            "decomposed": self.decomposed,
            "sub_queries": self.sub_queries,
            "alpha": self.alpha,
            "verify_verdict": self.verify_verdict,
            "model_tag": self.model_tag,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
        }


# ----------------------------------------------------------------------
# Synthetic-data layer (training labels)
# ----------------------------------------------------------------------

@dataclass
class QAPair:
    """Synthetic QA training example for generator SFT."""

    query: str
    answer_with_citations: str
    gold_chunk_ids: list[str]
    question_type: Literal[
        "factoid", "policy", "procedural", "multi_part", "unanswerable"
    ] = "factoid"
    domain: str = ""
    citations: list[Citation] = field(default_factory=list)   # ✅ ADDED
    qa_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["citations"] = [c.to_dict() for c in self.citations]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QAPair":
        citations = [Citation(**c) for c in d.get("citations", [])]
        return cls(
            query=d["query"],
            answer_with_citations=d["answer_with_citations"] if "answer_with_citations" in d else d["answer"],
            gold_chunk_ids=d["gold_chunk_ids"] if "gold_chunk_ids" in d else d["chunk_id"],
            question_type=d.get("question_type", d.get("type", "factoid")),
            domain=d.get("domain", ""),
            citations=citations,
            qa_id=d.get("qa_id", str(uuid.uuid4())),
        )


@dataclass
class PreferenceTriplet:
    """DPO preference pair (chosen > rejected) with rejection reason tag."""

    query: str
    chosen: str
    rejected: str
    rejection_type: Literal[
        "hallucinated_citation",
        "no_citation",
        "verbose_unfaithful",
        "wrong_tool",
        "partial_truncation",
        "unsafe_tone",
    ]
    pref_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConfidenceLabel:
    """Soft (continuous) confidence target derived from BERTScore."""

    query: str
    top5_chunk_ids: list[str]
    soft_label: float  # in [0, 1]
    gold_answer: str = ""
    evidence_answer: str = ""
    label_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolRouteLabel:
    """Gold tool for joint confidence + tool-policy training."""

    query: str
    gold_tool: Literal["AnswerDirect", "SearchKB", "GetPolicy", "CreateTicket"]
    features: dict[str, float] = field(default_factory=dict)
    top5_chunk_ids: list[str] = field(default_factory=list)
    label_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlphaLabel:
    """Grid-searched optimal alpha for adaptive fusion network."""

    query: str
    optimal_alpha: float  # in [0, 1]
    grid_scores: dict[str, float] = field(default_factory=dict)  # alpha -> recall@20
    label_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecompLabel:
    """Training label for the multi-part query classifier/splitter."""

    query: str
    is_multi_part: bool
    sub_queries: list[str] = field(default_factory=list)
    label_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Operations layer (audit, escalation)
# ----------------------------------------------------------------------


@dataclass
class TicketRecord:
    """Support ticket created when CGAL escalates."""

    ticket_id: str
    session_id: str
    query: str
    summary: str
    category: Literal["billing", "technical", "account", "policy", "other"] = "other"
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    user_context: str = ""
    evidence_gap: str = ""
    estimated_response_time: str = "24h"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @staticmethod
    def generate_ticket_id() -> str:
        return "TKT-" + uuid.uuid4().hex[:10].upper()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
