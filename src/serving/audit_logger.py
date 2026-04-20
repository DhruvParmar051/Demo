"""
AuditLogger -- SQLite-backed append-only log for every served query.

Writes one row per ``QueryResponse`` so we can (a) replay the evaluation
set offline and (b) expose escalation tickets via ``GET /tickets``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.schema import QueryResponse

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT,
    model_tag TEXT,
    query TEXT,
    answer TEXT,
    citations TEXT,
    tool_calls TEXT,
    confidence REAL,
    cgal_iterations INTEGER,
    fcrs REAL,
    latency_ms REAL,
    ttft_ms REAL,
    ticket_id TEXT,
    decomposed INTEGER,
    verify_verdict TEXT
);
CREATE INDEX IF NOT EXISTS audit_session_idx ON audit(session_id);
CREATE INDEX IF NOT EXISTS audit_ticket_idx ON audit(ticket_id);
"""


class AuditLogger:
    """Thread-safe sqlite audit log."""

    def __init__(self, db_path: str | Path = "data/audit.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        for stmt in _SCHEMA.strip().split(";"):
            if stmt.strip():
                self._conn.execute(stmt)
        self._conn.commit()

    # ------------------------------------------------------------------

    def log(self, response: QueryResponse, model_tag: str, query: str) -> None:
        """Append an audit row for the given response."""
        ts = datetime.now(timezone.utc).isoformat()
        citations = json.dumps(
            [c.to_dict() for c in response.citations], ensure_ascii=False
        )
        tool_calls = json.dumps(
            [t.to_dict() for t in response.tool_calls], ensure_ascii=False
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO audit (
                    timestamp, session_id, model_tag, query, answer,
                    citations, tool_calls, confidence, cgal_iterations,
                    fcrs, latency_ms, ttft_ms, ticket_id, decomposed, verify_verdict
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ts,
                    response.session_id,
                    model_tag,
                    query,
                    response.answer,
                    citations,
                    tool_calls,
                    response.confidence,
                    response.cgal_iterations,
                    response.fcrs,
                    response.latency_ms,
                    response.ttft_ms,
                    response.ticket_id,
                    1 if response.decomposed else 0,
                    response.verify_verdict,
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------------

    def list_tickets(self) -> list[dict[str, Any]]:
        """Return distinct escalation tickets."""
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT ticket_id, session_id, query, answer, confidence,
                       model_tag, timestamp
                FROM audit
                WHERE ticket_id IS NOT NULL AND ticket_id != ''
                ORDER BY timestamp DESC
                """
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def stats(self) -> dict[str, Any]:
        """Aggregate counters for the /metrics endpoint."""
        with self._lock:
            c = self._conn.execute("SELECT COUNT(*) FROM audit")
            total = int(c.fetchone()[0])
            c = self._conn.execute(
                "SELECT model_tag, COUNT(*) FROM audit GROUP BY model_tag"
            )
            by_tag = {row[0] or "unknown": int(row[1]) for row in c.fetchall()}
            c = self._conn.execute(
                "SELECT AVG(latency_ms), AVG(confidence), "
                "SUM(CASE WHEN ticket_id IS NOT NULL AND ticket_id != '' THEN 1 ELSE 0 END) "
                "FROM audit"
            )
            avg_lat, avg_conf, n_esc = c.fetchone()
            return {
                "total_queries": total,
                "queries_by_tag": by_tag,
                "avg_latency_ms": float(avg_lat or 0.0),
                "avg_confidence": float(avg_conf or 0.0),
                "escalations": int(n_esc or 0),
            }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
