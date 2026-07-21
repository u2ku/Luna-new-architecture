"""SQLite store for the email channel.

Persists every inbound and outbound email Luna has seen, plus a
thread index that makes "give me the last message in this
conversation" a single indexed query. The store does not own
ledger state — the world ledger (JSONL) is the source of truth for
Luna events; the SQLite file is a channel-local index for fast
email lookups (threading, dedup, outbound queue, delivery state).

File path is the canonical ``$LUNA_DATA_ROOT/email/email.db``;
``init_db`` creates the directory and the schema on first use.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    message_id        TEXT PRIMARY KEY,            -- RFC 822 Message-ID
    thread_id         TEXT NOT NULL,               -- email threading anchor
    stream_id         TEXT NOT NULL,               -- the Luna stream_id
    direction         TEXT NOT NULL,               -- 'inbound' or 'outbound'
    from_addr         TEXT NOT NULL,
    to_addr           TEXT NOT NULL,
    subject           TEXT,
    body              TEXT NOT NULL,
    in_reply_to       TEXT,
    received_at       TEXT NOT NULL,               -- ISO 8601 UTC
    ledger_event_id   TEXT,                        -- user_message or assistant_message event id
    turn_id           TEXT,                        -- the Luna turn
    status            TEXT NOT NULL DEFAULT 'received'
);

CREATE INDEX IF NOT EXISTS idx_messages_stream    ON messages(stream_id, received_at);
CREATE INDEX IF NOT EXISTS idx_messages_thread   ON messages(thread_id, received_at);
CREATE INDEX IF NOT EXISTS idx_messages_status   ON messages(status, received_at);

CREATE TABLE IF NOT EXISTS threads (
    thread_id         TEXT PRIMARY KEY,
    from_addr         TEXT NOT NULL,
    subject           TEXT,
    first_message_at  TEXT NOT NULL,
    last_message_at   TEXT NOT NULL,
    message_count     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_threads_last ON threads(last_message_at DESC);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class EmailStore:
    """SQLite-backed email index.

    Thin wrapper around stdlib ``sqlite3``. All write paths
    upsert on ``message_id`` so a redelivered SMTP message is
    idempotent.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ── writes ────────────────────────────────────────────────────────

    def record_message(
        self,
        *,
        message_id: str,
        thread_id: str,
        stream_id: str,
        direction: str,
        from_addr: str,
        to_addr: str,
        subject: str | None,
        body: str,
        in_reply_to: str | None,
        ledger_event_id: str | None = None,
        turn_id: str | None = None,
        status: str = "received",
    ) -> None:
        """Insert or update a message. Idempotent on message_id."""
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    message_id, thread_id, stream_id, direction,
                    from_addr, to_addr, subject, body, in_reply_to,
                    received_at, ledger_event_id, turn_id, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    ledger_event_id = COALESCE(excluded.ledger_event_id, messages.ledger_event_id),
                    turn_id         = COALESCE(excluded.turn_id,         messages.turn_id),
                    status          = excluded.status
                """,
                (
                    message_id, thread_id, stream_id, direction,
                    from_addr, to_addr, subject, body, in_reply_to,
                    now, ledger_event_id, turn_id, status,
                ),
            )
            conn.execute(
                """
                INSERT INTO threads (thread_id, from_addr, subject, first_message_at, last_message_at, message_count)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(thread_id) DO UPDATE SET
                    last_message_at = excluded.last_message_at,
                    message_count   = message_count + 1
                """,
                (thread_id, from_addr, subject, now, now),
            )
            conn.commit()

    def mark_status(self, message_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE messages SET status = ? WHERE message_id = ?",
                (status, message_id),
            )
            conn.commit()

    # ── reads ─────────────────────────────────────────────────────────

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM messages WHERE message_id = ?", (message_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_thread(self, thread_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE thread_id = ? ORDER BY received_at ASC",
                (thread_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def last_inbound_in_thread(self, thread_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM messages WHERE thread_id = ? AND direction = 'inbound' "
                "ORDER BY received_at DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
        return dict(row) if row else None

    def recent_threads(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM threads ORDER BY last_message_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
