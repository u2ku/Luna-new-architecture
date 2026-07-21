"""Read the ledger and return the last N message events for a stream.

The ledger is an append-only JSONL file of world events. This module walks
it in order, filters to message events for a specific ``stream_id``,
and returns the most recent ``limit`` of them. The context builder
imports :func:`recent_message_events` directly.

Multi-channel isolation
-----------------------

Luna talks to many channels — the web UI, Slack, Google Chat, Discord,
Telegram, ad-hoc APIs — and a single ledger holds events from all of
them. ``stream_id`` is the per-channel/per-thread handle; it MUST be
supplied so a private web conversation can never bleed into a Slack
or Google Space context. There is intentionally no way to read
"recent messages globally".

A typical ``stream_id`` shape:

    <platform>:<account_id>:<conversation_id>:<thread_id>

For example ``web::session-uuid`` for a private web session,
``slack:T123:C456:1720000000.000100`` for a Slack thread.

What counts as a "message event"
--------------------------------

An event is a message event when its ``type`` is one of:

* ``user_message``      — an incoming user message
* ``assistant_message`` — an outgoing assistant message

Other event types (tool calls, system events, receipts) are filtered
out. The ledger stays the single source of truth for all event types;
this module is just a conversation-shaped view of one stream.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_LIMIT = 25
MESSAGE_TYPES: frozenset[str] = frozenset({"user_message", "assistant_message"})


@dataclass(frozen=True)
class MessageEvent:
    """A user or assistant message event from one stream of the ledger.

    Mirrors the ``world_event`` schema (see
    ``schemas/world_event.schema.json``) with one small addition:

    * ``turn_id`` — the writer-attached identifier that groups events
      belonging to the same turn. ``None`` when the writer did not
      attach one.
    """

    event_id: str
    seq: int
    timestamp: str
    type: str
    actor: dict
    source: dict
    destination: dict
    stream_id: str
    payload: dict
    turn_id: str | None = None

    @property
    def text(self) -> str:
        """The plain text of the message, or ``""`` if not present."""
        value = self.payload.get("text")
        return value if isinstance(value, str) else ""

    @property
    def is_user(self) -> bool:
        return self.type == "user_message"

    @property
    def is_assistant(self) -> bool:
        return self.type == "assistant_message"


class LedgerNotFoundError(FileNotFoundError):
    """Raised when the ledger file cannot be located at the resolved path."""


class StreamMismatchError(ValueError):
    """Raised when an event's stream_id does not match the requested stream.

    Defensive: an event that reaches the consumer without a stream_id,
    or with a stream_id that doesn't match the caller's, is a bug.
    The runtime should never pass mixed-stream events to the model.
    """


def ledger_path() -> Path:
    """Return the path to the canonical event ledger.

    Resolution order:

    1. ``$LUNA_DATA_ROOT/ledger/world.jsonl`` — the documented location
       (matches ``LunaData/ledger/world.jsonl`` in a live deployment).
    2. ``./ledger/world.jsonl`` — fallback for tests and dev runs that
       haven't set ``LUNA_DATA_ROOT``.
    """
    root = os.environ.get("LUNA_DATA_ROOT")
    if root:
        return Path(root) / "ledger" / "world.jsonl"
    return Path("ledger") / "world.jsonl"


def _is_message_event(event: dict, message_types: frozenset[str]) -> bool:
    return event.get("type") in message_types


def _coerce(event: dict, stream_id: str) -> MessageEvent:
    """Coerce a ledger dict to a :class:`MessageEvent`.

    Enforces that the event's stream_id matches the requested stream.
    An event from a different stream is a programming error, not a
    soft warning — the runtime would otherwise pass cross-channel
    context to the model.
    """
    payload = event.get("payload") or {}
    turn_id = payload.get("turn_id")
    event_stream = event.get("stream_id")
    if event_stream is None:
        raise StreamMismatchError(
            f"event {event.get('event_id')} has no stream_id; cannot "
            f"include in stream {stream_id!r}"
        )
    if event_stream != stream_id:
        raise StreamMismatchError(
            f"event {event.get('event_id')} belongs to stream "
            f"{event_stream!r}, not {stream_id!r}"
        )
    return MessageEvent(
        event_id=str(event.get("event_id", "")),
        seq=int(event.get("seq", 0)),
        timestamp=str(event.get("timestamp", "")),
        type=str(event.get("type", "")),
        actor=event.get("actor") or {},
        source=event.get("source") or {},
        destination=event.get("destination") or {},
        stream_id=str(event_stream),
        payload=payload,
        turn_id=str(turn_id) if turn_id is not None else None,
    )


def _iter_events(path: Path) -> Iterable[dict]:
    """Yield parsed events from a JSONL ledger.

    Blank lines and lines that fail JSON parsing are silently skipped —
    the ledger is meant to be append-only and validated at write time,
    so a single bad line should not break a read.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def recent_message_events(
    stream_id: str,
    limit: int = DEFAULT_LIMIT,
    *,
    path: Path | None = None,
    message_types: frozenset[str] = MESSAGE_TYPES,
) -> list[MessageEvent]:
    """Read the ledger and return the most recent ``limit`` message events for one stream.

    Parameters
    ----------
    stream_id:
        Required. The conversation stream to filter to. Events with a
        different ``stream_id`` are NOT returned — this is the
        security boundary that keeps a private web conversation out
        of a Slack or Google Space context. Events with no
        ``stream_id`` raise :class:`StreamMismatchError`.
    limit:
        Maximum number of message events to return. ``0`` or negative
        returns an empty list.
    path:
        Override the ledger path. Defaults to :func:`ledger_path`.
    message_types:
        Set of event type names to treat as message events. Defaults
        to ``MESSAGE_TYPES`` (``user_message`` and ``assistant_message``).

    Returns
    -------
    list[MessageEvent]
        The matching events in ledger order (oldest first), capped at
        the last ``limit`` of them. Streaming pass over the file; the
        whole ledger is never held in memory.

    Raises
    ------
    LedgerNotFoundError
        If the resolved ledger path is not a regular file.
    StreamMismatchError
        If the ledger contains an event with a missing or
        mismatched ``stream_id``. This is a defensive guard against
        the runtime accidentally passing cross-stream context to
        the model.
    """
    if not stream_id:
        raise ValueError("stream_id is required")
    if limit <= 0 or not message_types:
        return []
    resolved = path if path is not None else ledger_path()
    if not resolved.is_file():
        raise LedgerNotFoundError(f"ledger not found at {resolved}")

    window: deque[MessageEvent] = deque(maxlen=limit)
    for event in _iter_events(resolved):
        if not _is_message_event(event, message_types):
            continue
        if event.get("stream_id") != stream_id:
            continue
        window.append(_coerce(event, stream_id))
    return list(window)
