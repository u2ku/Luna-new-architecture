"""Reduce a ledger of world events into a derived state.

The reducer reads the canonical event ledger, discovers the event
types that are actually present, and dispatches each event to a
registered handler. Events whose type has no handler are filtered
out — the reducer only processes what it knows how to process, and
the set of handlers is the contract.

This is the property the rest of the runtime depends on: a reducer
run over a ledger of unknown event types will silently drop the
unfamiliar ones rather than crash or coerce them. New event types
show up in the stats so the operator can decide whether to add a
handler.

The reducer is intentionally a single linear pass. It is not a
streaming operator; the whole ledger is read once, handlers are
called in ledger order, and a derived state object is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from ..context.recent_events import _iter_events


# A handler mutates its state object in place. The reducer calls
# ``handler(event, state_slot)`` where ``state_slot`` is the
# per-domain state object (NOT a dict).
Handler = Callable[[dict, object], None]


def _handler_domain(handler: Handler) -> str:
    """Return the domain key for a handler.

    Handlers declare their domain via a ``.domain`` attribute. If
    unset, the function's ``__name__`` is used.
    """
    domain = getattr(handler, "domain", None)
    if isinstance(domain, str) and domain:
        return domain
    return handler.__name__


def _initial_state(handler: Handler) -> object:
    """Return the initial value for the handler's state slot.

    Handlers may attach a no-arg ``initial`` attribute (function or
    classmethod) to declare their starting value. If they don't, the
    slot starts as ``None`` and the handler is expected to populate
    it on first use.
    """
    initial = getattr(handler, "initial", None)
    if callable(initial):
        return initial()
    return None


@dataclass(frozen=True)
class ReductionResult:
    """The output of a reducer run.

    Attributes
    ----------
    state:
        Per-domain state. Keys are handler-declared domain names;
        values are the per-domain state objects each handler mutates.
    stats:
        Counters describing what the run did:

        * ``events_in_ledger`` — total parsed events seen
        * ``events_processed``  — events that had a matching handler
        * ``events_filtered``   — events with no handler (unknown type)
        * ``event_types_in_ledger`` — distinct types seen
        * ``event_types_handled``   — distinct types that had a handler
        * ``event_types_ignored``   — distinct types seen but not handled
    """

    state: Mapping[str, object]
    stats: Mapping[str, int] = field(default_factory=dict)


class Reducer:
    """Reducer driven by a handler registry keyed by event type.

    Construct with the event types the reducer knows how to handle::

        reducer = Reducer(handlers={
            "user_message": current_state.handle,
            "assistant_message": current_state.handle,
        })

    Then call :meth:`reduce` against a ledger path. Unknown event
    types in the ledger are filtered (counted, not processed).
    """

    def __init__(self, handlers: Mapping[str, Handler]) -> None:
        if not handlers:
            raise ValueError("Reducer requires at least one handler")
        self._handlers: dict[str, Handler] = dict(handlers)

        # Validate that no two distinct handlers share a domain. Two
        # event types routing to the same handler is fine (same
        # domain, same handler object).
        domains: dict[str, Handler] = {}
        for handler in self._handlers.values():
            domain = _handler_domain(handler)
            existing = domains.get(domain)
            if existing is not None and existing is not handler:
                raise ValueError(
                    f"two distinct handlers declared domain {domain!r}: "
                    f"{existing!r} and {handler!r}"
                )
            domains[domain] = handler
        self._domain_handlers: dict[str, Handler] = domains

    @property
    def handled_types(self) -> frozenset[str]:
        """The event types this reducer has handlers for."""
        return frozenset(self._handlers)

    def discover_event_types(self, path: Path) -> set[str]:
        """Scan a ledger and return the set of distinct event types.

        Used to figure out which event types the runtime needs to add
        handlers for, and which currently show up in the live data.
        """
        types: set[str] = set()
        for event in _iter_events(path):
            t = event.get("type")
            if isinstance(t, str) and t:
                types.add(t)
        return types

    def reduce(self, path: Path) -> ReductionResult:
        """Read ``path`` and reduce it to a derived state.

        Linear pass over the JSONL ledger. Each event is dispatched to
        the registered handler for its type, or counted as filtered
        if no handler exists. The per-domain state objects are
        initialised fresh on each call from each handler's
        ``initial``.
        """
        state: dict[str, object] = {
            domain: _initial_state(handler)
            for domain, handler in self._domain_handlers.items()
        }
        present_types: set[str] = set()
        processed = 0
        filtered = 0
        for event in _iter_events(path):
            event_type = event.get("type")
            if not isinstance(event_type, str) or not event_type:
                # Malformed event (no type field) — filter it.
                filtered += 1
                continue
            present_types.add(event_type)
            handler = self._handlers.get(event_type)
            if handler is None:
                filtered += 1
                continue
            domain = _handler_domain(handler)
            handler(event, state[domain])
            processed += 1

        handled_types = present_types & set(self._handlers)
        ignored_types = present_types - set(self._handlers)
        stats: dict[str, int] = {
            "events_in_ledger": processed + filtered,
            "events_processed": processed,
            "events_filtered": filtered,
            "event_types_in_ledger": len(present_types),
            "event_types_handled": len(handled_types),
            "event_types_ignored": len(ignored_types),
        }
        return ReductionResult(state=state, stats=stats)
