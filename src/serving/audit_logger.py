"""
AuditLogger -- SQLite-backed append-only log for every served query.

Writes one row per ``QueryResponse`` so we can (a) replay the evaluation
set offline and (b) expose escalation tickets via ``GET /tickets``.

Old entries are automatically pruned after a configurable number of days
(default 30). Pruning is throttled — it runs every ``_PRUNE_EVERY_N_WRITES``
log calls — to avoid excessive I/O per request.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from src.data.schema import QueryResponse

logger = logging.getLogger(__name__)

_DEFAULT_RETENTION_DAYS = 30    # days before non-ticket rows are deleted
_PRUNE_EVERY_N_WRITES = 100     # run pruning every N log() calls


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
CREATE INDEX IF NOT EXISTS audit_timestamp_idx ON audit(timestamp);
"""


class AuditLogger:
    """Thread-safe SQLite audit log with automatic retention pruning.

    Rows older than ``retention_days`` are deleted automatically.
    Pruning is throttled (runs every ``_PRUNE_EVERY_N_WRITES`` writes) to
    minimise per-request overhead.
    """

    def __init__(
        self,
        db_path: str | Path = "data/audit.sqlite",
        retention_days: int = _DEFAULT_RETENTION_DAYS,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._retention_days = int(retention_days)
        self._write_count = 0
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

            self._write_count += 1
            if self._write_count % _PRUNE_EVERY_N_WRITES == 0:
                self._prune_old_entries()

    # ------------------------------------------------------------------
    # Retention policy
    # ------------------------------------------------------------------

    def _prune_old_entries(self) -> None:
        """Delete audit rows older than ``_retention_days`` days.

        Called inside the lock, so no additional locking is needed.
        Tickets (rows with a non-null ticket_id) are preserved regardless
        of age to maintain the escalation audit trail.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        ).isoformat()
        try:
            cur = self._conn.execute(
                "DELETE FROM audit WHERE timestamp < ? AND (ticket_id IS NULL OR ticket_id = '')",
                (cutoff,),
            )
            deleted = cur.rowcount
            self._conn.commit()
            if deleted > 0:
                logger.info(
                    "AuditLogger: pruned %d rows older than %d days",
                    deleted,
                    self._retention_days,
                )
        except Exception as exc:
            logger.warning("AuditLogger: pruning failed: %s", exc)

    def prune_now(self, retention_days: int | None = None) -> int:
        """Manually trigger a prune with an optional custom retention window.

        Returns the number of rows deleted.
        """
        days = retention_days if retention_days is not None else self._retention_days
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM audit WHERE timestamp < ? AND (ticket_id IS NULL OR ticket_id = '')",
                (cutoff,),
            )
            deleted = cur.rowcount
            self._conn.commit()
        logger.info(
            "AuditLogger.prune_now: deleted %d rows (retention=%d days)",
            deleted,
            days,
        )
        return deleted

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