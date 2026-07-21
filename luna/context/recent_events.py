"""Read the ledger and return the last N message events to the context builder.

The ledger is an append-only JSONL file of world events. This module walks
it in order, filters to user/assistant message events, and returns the
most recent ``limit`` of them. The context builder imports
:func:`recent_message_events` directly.

What counts as a "message event"
--------------------------------

An event is a message event when its ``type`` is one of:

* ``user_message``      — an incoming user message
* ``assistant_message`` — an outgoing assistant message

These are the names the live ``luna-server`` actually writes; the
filter is parameterised so a future deployment can override the set
without rewriting the reader.

Tool calls, tool results, system events, receipts, and any other event
type are filtered out. The ledger stays the single source of truth for
all event types; this module is just a conversation-shaped view of it.
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
    """A user or assistant message event from the ledger.

    Mirrors the ``world_event`` schema (see
    ``schemas/world_event.schema.json``) with two small additions:

    * ``turn_id`` — optional identifier the writer attached to group
      events that belong to the same turn. ``None`` when the writer
      did not attach one.
    * ``text`` — convenience accessor that pulls the message text out
      of ``payload`` when present.
    """

    event_id: str
    seq: int
    timestamp: str
    type: str
    actor: str
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


def ledger_path() -> Path:
    """Return the path to the canonical event ledger.

    Resolution order:

    1. ``$LUNA_DATA_ROOT/ledger/world.jsonl`` — the documented location
       (matches ``LunaData/ledger/world.jsonl`` in a live deployment).
    2. ``./ledger/world.jsonl`` — fallback for tests and dev runs that
       haven't set ``LUNA_DATA_ROOT``.

    The function does not check existence; callers that need to surface
    a missing file should use :class:`LedgerNotFoundError`.
    """
    root = os.environ.get("LUNA_DATA_ROOT")
    if root:
        return Path(root) / "ledger" / "world.jsonl"
    return Path("ledger") / "world.jsonl"


def _is_message_event(event: dict, message_types: frozenset[str]) -> bool:
    return event.get("type") in message_types


def _coerce(event: dict) -> MessageEvent:
    payload = event.get("payload") or {}
    turn_id = payload.get("turn_id")
    return MessageEvent(
        event_id=str(event.get("event_id", "")),
        seq=int(event.get("seq", 0)),
        timestamp=str(event.get("timestamp", "")),
        type=str(event.get("type", "")),
        actor=str(event.get("actor", "")),
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
    limit: int = DEFAULT_LIMIT,
    *,
    path: Path | None = None,
    message_types: frozenset[str] = MESSAGE_TYPES,
) -> list[MessageEvent]:
    """Read the ledger and return the most recent ``limit`` message events.

    Parameters
    ----------
    limit:
        Maximum number of message events to return. ``0`` or negative
        returns an empty list.
    path:
        Override the ledger path. Defaults to :func:`ledger_path`.
    message_types:
        Set of event type names to treat as message events. Defaults
        to ``MESSAGE_TYPES`` (``user_message`` and ``assistant_message``).
        Override when a deployment uses different names; an empty
        frozenset returns an empty list.

    Returns
    -------
    list[MessageEvent]
        The matching events in ledger order (oldest first), capped at
        the last ``limit`` of them. The function is a streaming pass
        over the file: it does not load the whole ledger into memory.

    Raises
    ------
    LedgerNotFoundError
        If the resolved ledger path is not a regular file.
    """
    if limit <= 0 or not message_types:
        return []
    resolved = path if path is not None else ledger_path()
    if not resolved.is_file():
        raise LedgerNotFoundError(f"ledger not found at {resolved}")

    window: deque[MessageEvent] = deque(maxlen=limit)
    for event in _iter_events(resolved):
        if not _is_message_event(event, message_types):
            continue
        window.append(_coerce(event))
    return list(window)

