"""
AegisRAG - Tool Executor

Dispatches tool calls (SearchKB, GetPolicy, CreateTicket) to the appropriate
backend and returns structured results. Persists escalation tickets via a
simple SQLite-backed store; falls back to an in-memory dict when no DB path
is configured.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.data.schema import TicketRecord
from src.tools.schemas import validate_tool_args

logger = logging.getLogger(__name__)


_SEVERITY_SLA = {
    "critical": "2h",
    "high": "8h",
    "medium": "24h",
    "low": "72h",
}


class _InMemoryTicketStore:
    """Fallback ticket store that keeps tickets in a dict."""

    def __init__(self) -> None:
        self._tickets: dict[str, TicketRecord] = {}

    def save(self, ticket: TicketRecord) -> None:
        self._tickets[ticket.ticket_id] = ticket

    def list_all(self) -> list[TicketRecord]:
        return list(self._tickets.values())

    def close(self) -> None:
        pass


class _SQLiteTicketStore:
    """SQLite-backed ticket store with WAL mode."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                session_id TEXT,
                query TEXT,
                summary TEXT,
                category TEXT,
                severity TEXT,
                user_context TEXT,
                evidence_gap TEXT,
                estimated_response_time TEXT,
                created_at TEXT
            )
            """
        )
        self.conn.commit()

    def save(self, ticket: TicketRecord) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO tickets VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ticket.ticket_id,
                ticket.session_id,
                ticket.query,
                ticket.summary,
                ticket.category,
                ticket.severity,
                ticket.user_context,
                ticket.evidence_gap,
                ticket.estimated_response_time,
                ticket.created_at,
            ),
        )
        self.conn.commit()

    def list_all(self) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT ticket_id, session_id, query, summary, category, severity, "
            "user_context, evidence_gap, estimated_response_time, created_at "
            "FROM tickets ORDER BY created_at DESC"
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()


class ToolExecutor:
    """Dispatches tool calls and executes them against the appropriate backend.

    Parameters
    ----------
    retriever : HybridRetriever
        Backing retriever for ``SearchKB``.
    reranker : ColBERTReranker or None
        Optional cross-encoder for post-retrieval reranking in ``SearchKB``.
    policy_index : dict[str, dict] or None
        Pre-built policy lookup (section_id -> {text, source, section_title}).
    ticket_store_path : Path or None
        SQLite database path for ticket persistence. When None, an in-memory
        dict is used.
    """

    def __init__(
        self,
        retriever: Any,
        reranker: Any | None = None,
        policy_index: dict[str, dict[str, Any]] | None = None,
        ticket_store_path: Path | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.policy_index: dict[str, dict[str, Any]] = policy_index or {}
        if ticket_store_path is not None:
            self.ticket_store: Any = _SQLiteTicketStore(Path(ticket_store_path))
        else:
            self.ticket_store = _InMemoryTicketStore()

    # ------------------------------------------------------------------

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call and return its result."""
        t_start = time.perf_counter()
        ok, errors = validate_tool_args(tool_name, args)
        if not ok:
            return {"error": "invalid_args", "details": errors}

        result: dict[str, Any]
        try:
            if tool_name == "SearchKB":
                result = self.search_kb(**{k: v for k, v in args.items() if k in
                                            ("query", "top_k", "filters")})
            elif tool_name == "GetPolicy":
                result = self.get_policy(
                    section_id=args.get("section_id", ""),
                    fuzzy=args.get("fuzzy", True),
                )
            elif tool_name == "CreateTicket":
                result = self.create_ticket(
                    query=args.get("query", ""),
                    session_id=args.get("session_id", ""),
                    summary=args.get("summary") or args.get("query", ""),
                    category=args.get("category", "other"),
                    severity=args.get("severity", "medium"),
                    user_context=args.get("user_context", ""),
                    evidence_gap=args.get("evidence_gap", args.get("reason", "")),
                )
            else:
                result = {"error": f"unknown_tool: {tool_name}"}
        except Exception as exc:
            logger.exception("Tool %s execution error", tool_name)
            result = {"error": str(exc)}

        result["_latency_ms"] = (time.perf_counter() - t_start) * 1000.0
        return result

    # ------------------------------------------------------------------

    def search_kb(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run hybrid retrieval + rerank and return top chunks."""
        if self.retriever is None:
            return {"error": "no_retriever", "results": []}
        retrieved = self.retriever.retrieve(
            query=query, top_k=max(top_k * 4, 20), filters=filters
        )
        if self.reranker is not None and retrieved:
            reranked = self.reranker.rerank(query, retrieved, top_k=top_k)
        else:
            reranked = retrieved[:top_k]

        results = []
        for chunk, score in reranked:
            results.append(
                {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "span_start": chunk.span_start,
                    "span_end": chunk.span_end,
                    "text": chunk.text,
                    "source": chunk.source,
                    "score": float(score),
                }
            )
        return {"results": results, "n_results": len(results)}

    # ------------------------------------------------------------------

    def get_policy(self, section_id: str, fuzzy: bool = True) -> dict[str, Any]:
        """Look up a policy section by ID (exact then fuzzy substring)."""
        if not section_id:
            return {"error": "missing_section_id"}
        if section_id in self.policy_index:
            record = self.policy_index[section_id]
            return {"section_id": section_id, "found": True, **record}
        if fuzzy:
            needle = section_id.lower()
            for sid, record in self.policy_index.items():
                if needle in sid.lower():
                    return {"section_id": sid, "found": True, "fuzzy": True, **record}
        return {"section_id": section_id, "found": False}

    # ------------------------------------------------------------------

    def create_ticket(
        self,
        query: str,
        session_id: str = "",
        summary: str = "",
        category: str = "other",
        severity: str = "medium",
        user_context: str = "",
        evidence_gap: str = "",
    ) -> dict[str, Any]:
        """Create an escalation ticket and persist it."""
        from typing import Literal, cast

        severity_key = severity if severity in _SEVERITY_SLA else "medium"
        sla = _SEVERITY_SLA[severity_key]
        category_key = category if category in (
            "billing", "technical", "account", "policy", "other"
        ) else "other"
        ticket = TicketRecord(
            ticket_id=TicketRecord.generate_ticket_id(),
            session_id=session_id,
            query=query,
            summary=summary or query[:200],
            category=cast(
                Literal["billing", "technical", "account", "policy", "other"],
                category_key,
            ),
            severity=cast(
                Literal["low", "medium", "high", "critical"],
                severity_key,
            ),
            user_context=user_context,
            evidence_gap=evidence_gap,
            estimated_response_time=sla,
        )
        severity = severity_key
        self.ticket_store.save(ticket)
        logger.info("Created ticket %s (severity=%s, SLA=%s)",
                    ticket.ticket_id, severity, sla)
        return {
            "ticket_id": ticket.ticket_id,
            "estimated_response_time": sla,
            "severity": severity,
            "category": ticket.category,
            "message": (
                f"I could not confidently answer this from available sources. "
                f"A support specialist will follow up (ticket {ticket.ticket_id}, "
                f"estimated response {sla})."
            ),
        }

    # ------------------------------------------------------------------

    def list_tickets(self) -> list[Any]:
        """Return all tickets (as dicts when SQLite-backed, records otherwise)."""
        return self.ticket_store.list_all()

    def close(self) -> None:
        self.ticket_store.close()
