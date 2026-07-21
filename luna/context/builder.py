"""Context builder for the model.

Composes the inputs the model needs for a turn: the recent conversation
for the active stream plus whatever else the runtime decides to add
(system prompt, retrieved documents, tool instructions, …).

This module is the consumer of :mod:`luna.context.recent_events`. The
caller of the builder supplies the surrounding structure; this module
is responsible for pulling the recent-message slice from one stream of
the ledger and shaping it for inclusion in the model context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .recent_events import MessageEvent, recent_message_events


@dataclass(frozen=True)
class ConversationTurn:
    """One user/assistant exchange extracted from the ledger.

    A turn is at most one user message followed by at most one assistant
    message. Tool calls and tool results are not represented here — they
    live in other context layers.
    """

    user: MessageEvent | None
    assistant: MessageEvent | None


def build_recent_messages(
    stream_id: str,
    limit: int = 25,
) -> list[MessageEvent]:
    """Return the last ``limit`` message events for ``stream_id``.

    ``stream_id`` is required. The builder never returns "recent
    messages globally" — that would mix private and public-channel
    context. See :mod:`luna.context.recent_events` for the security
    boundary this enforces.
    """
    return recent_message_events(stream_id=stream_id, limit=limit)


def group_into_turns(
    events: Sequence[MessageEvent],
) -> list[ConversationTurn]:
    """Group a flat list of message events into user/assistant turns.

    Pairs each user message with the next assistant message. A trailing
    user message with no assistant reply yields a turn with
    ``assistant=None``. An orphan assistant message (no preceding user)
    is dropped — the ledger's own ``turn_id`` metadata is the
    authoritative grouping when available; this is a best-effort view
    for ledgers that don't attach ``turn_id``.
    """
    turns: list[ConversationTurn] = []
    pending_user: MessageEvent | None = None
    for event in events:
        if event.is_user:
            if pending_user is not None:
                # Back-to-back user messages: close the previous turn
                # without an assistant reply and start a new one.
                turns.append(ConversationTurn(user=pending_user, assistant=None))
            pending_user = event
        elif event.is_assistant and pending_user is not None:
            turns.append(ConversationTurn(user=pending_user, assistant=event))
            pending_user = None
    if pending_user is not None:
        turns.append(ConversationTurn(user=pending_user, assistant=None))
    return turns
