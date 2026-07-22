"""End-to-end test of the email inbox sync pipeline.

The "safest first milestone":

    Receive email → write SQLite → write user_message →
    generate reply → store reply as unsent in SQLite

Mocks the Gmail API HTTP calls and the model provider. Verifies
the canonical event shape (actor, source, destination, stream_id,
turn_id, payload.text) on both user_message and assistant_message,
and that transport data + outbox rows land in the right tables.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from luna.api.routes import ChatService
from luna.channels.email.inbox import InboxSync
from luna.channels.email.store import EmailStore
from luna.ledger import WorldLedger
from luna.models.base import (
    FinishReason,
    Message,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    Usage,
)


# ── test doubles ───────────────────────────────────────────────────────


class FakeProvider(ModelProvider):
    """Stand-in model that echoes the system prompt + last user text."""

    name = "fake"

    def complete(self, request: ModelRequest) -> ModelResponse:
        last = next(
            (m for m in reversed(request.messages) if m.role == "user"),
            None,
        )
        text = f"(luna) you said: {last.content if last else ''}"
        return ModelResponse(
            content=text,
            finish_reason=FinishReason.STOP,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="fake-1",
        )


@dataclass
class FakeGmail:
    """Stand-in for GmailClient with canned responses."""

    messages: list[dict[str, Any]]  # each: {id, raw_bytes}
    listed_ids: list[str] = None

    def list_messages(self, *, query=None, label_ids=None, max_results=20, page_token=None):
        if self.listed_ids is None:
            self.listed_ids = [m["id"] for m in self.messages]
        return self.listed_ids, None

    def get_message_raw(self, message_id: str) -> bytes:
        for m in self.messages:
            if m["id"] == message_id:
                return m["raw_bytes"]
        raise KeyError(message_id)

    def send_raw(self, raw_message: bytes) -> str:
        raise AssertionError("read-side pipeline should not send")


def _build_email(
    from_addr: str = "zac@example.com",
    to_addr: str = "luna@example.com",
    subject: str = "My favorite color is blue",
    body: str = "My favorite color is blue. Remember that.",
    message_id: str = "<test-in-1@example.com>",
    in_reply_to: str | None = None,
) -> bytes:
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["Message-ID"] = message_id
    msg["Date"] = "Wed, 22 Jul 2026 10:00:00 +0000"
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    msg.set_content(body)
    return msg.as_bytes()


# ── fixture ─────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_workspace(tmp_path: Path, monkeypatch):
    """A fresh ledger + email.db for each test, and patch the
    chat service's recent-message reader so it sees the same
    ledger the inbox writes to."""
    ledger_path = tmp_path / "world.jsonl"
    db_path = tmp_path / "email.db"
    from luna.context import recent_events as _re
    monkeypatch.setattr(_re, "ledger_path", lambda: ledger_path)
    return ledger_path, db_path


# ── the milestone test ────────────────────────────────────────────────


def test_receive_to_outbox_end_to_end(tmp_workspace):
    ledger_path, db_path = tmp_workspace
    ledger = WorldLedger(ledger_path)
    store = EmailStore(db_path)
    chat = ChatService(
        provider=FakeProvider(),
        ledger=ledger,
        system_prompt="You are Luna.",
        model_name="fake-1",
        temperature=0.3,
        max_tokens=200,
    )
    gmail = FakeGmail(
        messages=[{"id": "gmail-msg-1", "raw_bytes": _build_email()}]
    )
    sync = InboxSync(
        gmail=gmail, ledger=ledger, store=store, chat=chat,
        agent_email="luna@example.com",
    )

    stats = sync.sync()
    assert stats == {"fetched": 1, "processed": 1, "skipped": 0, "failed": 0}, stats

    # ── 1. world ledger: user_message + assistant_message ──────────
    with ledger_path.open() as f:
        events = [json.loads(line) for line in f if line.strip()]
    assert len(events) == 2, events
    user, assistant = events

    # user_message shape
    assert user["type"] == "user_message"
    assert user["actor"] == {
        "id": "identity:zac@example.com",
        "type": "human",
        "display_name": "zac@example.com",
    }
    assert user["source"] == {
        "platform": "email",
        "adapter": "gmail-api",
        "account_id": "example.com",
        "conversation_id": "zac@example.com",
        "message_id": "<test-in-1@example.com>",
        "external_actor_id": "zac@example.com",
        "thread_id": "<test-in-1@example.com>",
    }
    assert user["destination"] == {"platform": "luna-runtime"}
    assert user["stream_id"] == "email:zac@example.com:<test-in-1@example.com>"
    assert "turn_id" in user
    assert user["payload"]["text"].rstrip("\n") == "My favorite color is blue. Remember that."
    assert user["payload"]["subject"] == "My favorite color is blue"
    assert user["payload"]["from"] == "zac@example.com"
    assert user["payload"]["to"] == "luna@example.com"
    assert user["payload"]["in_reply_to"] is None
    assert user["payload"]["message_id_header"] == "<test-in-1@example.com>"

    # assistant_message shape — same stream_id, same turn_id
    assert assistant["type"] == "assistant_message"
    assert assistant["actor"] == {
        "id": "agent:luna",
        "type": "agent",
        "display_name": "Luna",
    }
    assert assistant["source"] == {"platform": "luna-runtime"}
    assert assistant["destination"]["platform"] == "email"
    assert assistant["stream_id"] == user["stream_id"]
    assert assistant["turn_id"] == user["turn_id"]
    assert "(luna) you said:" in assistant["payload"]["text"]
    assert assistant["destination"]["conversation_id"] == "zac@example.com"
    assert "reply_to_event_id" in assistant["payload"]

    # ── 2. SQLite: gmail_messages, messages, threads, pending_replies
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        gmail_rows = conn.execute("SELECT * FROM gmail_messages").fetchall()
        msg_rows = conn.execute("SELECT * FROM messages").fetchall()
        thread_rows = conn.execute("SELECT * FROM threads").fetchall()
        pending_rows = conn.execute("SELECT * FROM pending_replies").fetchall()

    assert len(gmail_rows) == 1
    gr = dict(gmail_rows[0])
    assert gr["gmail_message_id"] == "gmail-msg-1"
    assert gr["rfc822_message_id"] == "<test-in-1@example.com>"
    assert gr["from_addr"] == "zac@example.com"
    assert gr["stream_id"] == user["stream_id"]
    assert gr["ledger_event_id"] == user["event_id"]
    assert gr["status"] == "synced"

    assert len(msg_rows) == 1  # inbound only — no outbound yet
    mr = dict(msg_rows[0])
    assert mr["direction"] == "inbound"
    assert mr["from_addr"] == "zac@example.com"
    assert mr["status"] == "processed"
    assert mr["ledger_event_id"] == user["event_id"]

    assert len(thread_rows) == 1
    tr = dict(thread_rows[0])
    assert tr["from_addr"] == "zac@example.com"
    assert tr["message_count"] == 1

    # The outbox — the reply is queued, NOT sent.
    assert len(pending_rows) == 1
    pr = dict(pending_rows[0])
    assert pr["status"] == "pending"
    assert pr["stream_id"] == user["stream_id"]
    assert pr["turn_id"] == user["turn_id"]
    assert pr["to_addr"] == "zac@example.com"
    assert pr["from_addr"] == "luna@example.com"
    assert pr["source_user_event_id"] == user["event_id"]
    assert pr["source_message_id"] == "<test-in-1@example.com>"
    assert pr["in_reply_to"] == "<test-in-1@example.com>"
    assert "(luna) you said:" in pr["body"]
    assert pr["sent_gmail_message_id"] is None
    assert pr["sent_at"] is None


def test_sync_is_idempotent(tmp_workspace):
    """Re-running sync with the same Gmail message IDs is a no-op."""
    ledger_path, db_path = tmp_workspace
    ledger = WorldLedger(ledger_path)
    store = EmailStore(db_path)
    chat = ChatService(
        provider=FakeProvider(), ledger=ledger, system_prompt="You are Luna.",
    )
    gmail = FakeGmail(
        messages=[{"id": "gmail-msg-1", "raw_bytes": _build_email()}]
    )
    sync = InboxSync(
        gmail=gmail, ledger=ledger, store=store, chat=chat,
        agent_email="luna@example.com",
    )

    s1 = sync.sync()
    s2 = sync.sync()
    assert s1 == {"fetched": 1, "processed": 1, "skipped": 0, "failed": 0}
    assert s2 == {"fetched": 1, "processed": 0, "skipped": 1, "failed": 0}

    # Only 2 events in the ledger (one user + one assistant), not 4
    with ledger_path.open() as f:
        events = [json.loads(line) for line in f if line.strip()]
    assert len(events) == 2
    # Only 1 pending reply
    pending = store.list_pending_replies()
    assert len(pending) == 1


def test_threaded_email_reuses_thread_anchor(tmp_workspace):
    """A reply (In-Reply-To) shares the thread anchor; the second
    message's stream_id collapses to the first message's."""
    ledger_path, db_path = tmp_workspace
    ledger = WorldLedger(ledger_path)
    store = EmailStore(db_path)
    chat = ChatService(
        provider=FakeProvider(), ledger=ledger, system_prompt="You are Luna.",
    )
    m1_id = "<thread-anchor-1@example.com>"
    m2_id = "<thread-reply-1@example.com>"
    gmail = FakeGmail(
        messages=[
            {"id": "g-1", "raw_bytes": _build_email(message_id=m1_id, body="first")},
            {"id": "g-2", "raw_bytes": _build_email(
                message_id=m2_id, in_reply_to=m1_id, body="second"
            )},
        ]
    )
    sync = InboxSync(
        gmail=gmail, ledger=ledger, store=store, chat=chat,
        agent_email="luna@example.com",
    )
    stats = sync.sync()
    assert stats["processed"] == 2

    # Both messages share the thread anchor and therefore the same
    # stream_id, even though they have different RFC 822 message ids.
    with ledger_path.open() as f:
        events = [json.loads(line) for line in f if line.strip()]
    stream_ids = {e["stream_id"] for e in events}
    assert len(stream_ids) == 1, f"expected one stream, got {stream_ids}"
