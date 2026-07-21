"""Current-state reducer: track the most recent user and assistant turn.

Handles the two message event types actually present in the live
ledger (``user_message`` and ``assistant_message``). Each handler
update mutates the :class:`CurrentState` slot in the reducer's
state dict.

The handler declares its domain as ``"current_state"`` so the
reducer knows where to put its state object and what to call
``initial()`` to get a fresh one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurrentState:
    """Latest-message snapshot the rest of the runtime can read.

    ``last_user_event`` and ``last_assistant_event`` are the full
    event dicts (id, seq, timestamp, type, actor, payload) so callers
    can read whatever they need without re-walking the ledger.

    ``turn_count`` is the number of complete user → assistant pairs
    seen so far. It only increments on ``assistant_message``; a
    trailing user message with no reply does not bump it.
    """

    last_user_event: dict | None = None
    last_assistant_event: dict | None = None
    turn_count: int = 0


def initial() -> CurrentState:
    return CurrentState()


def handle(event: dict, state: CurrentState) -> None:
    """Reducer handler for both ``user_message`` and ``assistant_message``.

    The reducer routes both event types through this single handler
    because the two share state — the assistant side is the close of
    a turn that the user side opened.
    """
    event_type = event.get("type")
    if event_type == "user_message":
        state.last_user_event = event
    elif event_type == "assistant_message":
        state.last_assistant_event = event
        state.turn_count += 1


# Handler metadata read by the reducer.
handle.domain = "current_state"
handle.initial = initial
