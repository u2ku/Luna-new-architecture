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

-- Gmail API sync state (per account_email so multiple inboxes work)
CREATE TABLE IF NOT EXISTS gmail_state (
    account_email     TEXT PRIMARY KEY,
    last_history_id   TEXT,
    last_sync_at      TEXT NOT NULL
);

-- Transport IDs — maps Gmail-side IDs to our ledger/stream IDs.
-- Lets the sync skip messages we've already processed without
-- round-tripping to the ledger.
CREATE TABLE IF NOT EXISTS gmail_messages (
    gmail_message_id  TEXT PRIMARY KEY,            -- Gmail API message id
    rfc822_message_id TEXT NOT NULL,               -- RFC 822 Message-ID header
    thread_id         TEXT NOT NULL,               -- Gmail thread id
    account_email     TEXT NOT NULL,
    from_addr         TEXT NOT NULL,
    subject           TEXT,
    snippet           TEXT,
    received_at       TEXT NOT NULL,               -- when Gmail saw it
    synced_at         TEXT NOT NULL,               -- when we ingested
    stream_id         TEXT NOT NULL,
    ledger_event_id   TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'synced'  -- 'synced' | 'failed'
);

CREATE INDEX IF NOT EXISTS idx_gmail_messages_thread ON gmail_messages(thread_id);

-- Outbox: replies Luna generated but hasn't sent yet.
-- The read-side pipeline writes here after generating the reply;
-- a separate "send" step consumes it and calls gmail.send().
-- Credentials live in secrets/gmail/, NOT in this DB.
CREATE TABLE IF NOT EXISTS pending_replies (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id            TEXT NOT NULL,
    turn_id              TEXT NOT NULL,
    source_user_event_id TEXT NOT NULL,            -- the user_message we replied to
    source_message_id    TEXT NOT NULL,            -- the RFC 822 Message-ID
    to_addr              TEXT NOT NULL,
    from_addr            TEXT NOT NULL,
    subject              TEXT,
    in_reply_to          TEXT,
    references_json      TEXT,                     -- JSON list of message-ids
    body                 TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'pending',  -- 'pending'|'sending'|'sent'|'failed'
    created_at           TEXT NOT NULL,
    sent_at              TEXT,
    sent_gmail_message_id TEXT,
    error                TEXT
);

CREATE INDEX IF NOT EXISTS idx_pending_replies_status ON pending_replies(status, created_at);
CREATE INDEX IF NOT EXISTS idx_pending_replies_stream ON pending_replies(stream_id);
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

    # ── Gmail transport data ───────────────────────────────────────────

    def record_gmail_message(
        self,
        *,
        gmail_message_id: str,
        rfc822_message_id: str,
        thread_id: str,
        account_email: str,
        from_addr: str,
        subject: str | None,
        snippet: str | None,
        received_at: str,
        stream_id: str,
        ledger_event_id: str,
        status: str = "synced",
    ) -> None:
        """Record a Gmail message we've ingested. Idempotent on
        ``gmail_message_id`` so a redelivered history tick is safe.
        """
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO gmail_messages (
                    gmail_message_id, rfc822_message_id, thread_id, account_email,
                    from_addr, subject, snippet, received_at, synced_at,
                    stream_id, ledger_event_id, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(gmail_message_id) DO UPDATE SET
                    ledger_event_id = excluded.ledger_event_id,
                    status          = excluded.status
                """,
                (
                    gmail_message_id, rfc822_message_id, thread_id, account_email,
                    from_addr, subject, snippet, received_at, now,
                    stream_id, ledger_event_id, status,
                ),
            )
            conn.commit()

    def get_gmail_message(self, gmail_message_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM gmail_messages WHERE gmail_message_id = ?",
                (gmail_message_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_gmail_state(self, account_email: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM gmail_state WHERE account_email = ?",
                (account_email,),
            ).fetchone()
        return dict(row) if row else None

    def set_gmail_state(
        self, account_email: str, last_history_id: str | None
    ) -> None:
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO gmail_state (account_email, last_history_id, last_sync_at)
                VALUES (?, ?, ?)
                ON CONFLICT(account_email) DO UPDATE SET
                    last_history_id = excluded.last_history_id,
                    last_sync_at    = excluded.last_sync_at
                """,
                (account_email, last_history_id, now),
            )
            conn.commit()

    # ── Outbox (pending replies) ───────────────────────────────────────

    def enqueue_reply(
        self,
        *,
        stream_id: str,
        turn_id: str,
        source_user_event_id: str,
        source_message_id: str,
        to_addr: str,
        from_addr: str,
        subject: str | None,
        in_reply_to: str | None,
        references: list[str] | None,
        body: str,
    ) -> int:
        """Insert a pending reply. Returns the row id."""
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO pending_replies (
                    stream_id, turn_id, source_user_event_id, source_message_id,
                    to_addr, from_addr, subject, in_reply_to, references_json,
                    body, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stream_id, turn_id, source_user_event_id, source_message_id,
                    to_addr, from_addr, subject, in_reply_to,
                    json.dumps(references or []),
                    body, "pending", now,
                ),
            )
            conn.commit()
            return cur.lastrowid or 0

    def list_pending_replies(
        self, *, status: str = "pending", limit: int = 50
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pending_replies WHERE status = ? "
                "ORDER BY created_at ASC LIMIT ?",
                (status, limit),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if d.get("references_json"):
                try:
                    d["references"] = json.loads(d["references_json"])
                except json.JSONDecodeError:
                    d["references"] = []
            else:
                d["references"] = []
            out.append(d)
        return out

    def mark_reply_status(
        self,
        reply_id: int,
        status: str,
        *,
        sent_gmail_message_id: str | None = None,
        error: str | None = None,
    ) -> None:
        now = _now_iso()
        with self._connect() as conn:
            if status == "sent":
                conn.execute(
                    "UPDATE pending_replies SET status = ?, sent_at = ?, "
                    "sent_gmail_message_id = ?, error = NULL WHERE id = ?",
                    (status, now, sent_gmail_message_id, reply_id),
                )
            else:
                conn.execute(
                    "UPDATE pending_replies SET status = ?, error = ? WHERE id = ?",
                    (status, error, reply_id),
                )
            conn.commit()
