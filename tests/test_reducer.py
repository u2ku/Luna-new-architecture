"""Tests for luna.reducer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luna.reducer.current_state import CurrentState, handle, initial
from luna.reducer.reducer import Reducer, ReductionResult


# ── helpers ────────────────────────────────────────────────────────────────


def _write_ledger(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def _msg(seq: int, type_: str, text: str = "") -> dict:
    return {
        "event_id": f"e-{seq:04d}",
        "seq": seq,
        "timestamp": "2026-07-22T00:00:00Z",
        "type": type_,
        "actor": "user" if type_ == "user_message" else "luna",
        "payload": {"text": text},
    }


# ── Reducer construction ───────────────────────────────────────────────────


def test_empty_handlers_raises() -> None:
    with pytest.raises(ValueError):
        Reducer(handlers={})


def test_handled_types_exposes_registry() -> None:
    reducer = Reducer(handlers={"user_message": handle, "assistant_message": handle})
    assert reducer.handled_types == frozenset({"user_message", "assistant_message"})


# ── discover_event_types ───────────────────────────────────────────────────


def test_discover_returns_set_of_types(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            _msg(1, "user_message"),
            _msg(2, "assistant_message"),
            _msg(3, "tool.call"),
            _msg(4, "system.boot"),
        ],
    )
    reducer = Reducer(handlers={"user_message": handle, "assistant_message": handle})
    assert reducer.discover_event_types(p) == {
        "user_message",
        "assistant_message",
        "tool.call",
        "system.boot",
    }


def test_discover_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(_msg(1, "user_message")),
                "{bad json",
                json.dumps(_msg(2, "assistant_message")),
            ]
        )
    )
    reducer = Reducer(handlers={"user_message": handle})
    assert reducer.discover_event_types(p) == {"user_message", "assistant_message"}


def test_discover_empty_ledger(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    p.write_text("")
    reducer = Reducer(handlers={"user_message": handle})
    assert reducer.discover_event_types(p) == set()


# ── reduce: dispatch & filtering ───────────────────────────────────────────


def test_reduce_processes_known_events(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            _msg(1, "user_message", "hi"),
            _msg(2, "assistant_message", "hello back"),
        ],
    )
    reducer = Reducer(handlers={"user_message": handle, "assistant_message": handle})
    result = reducer.reduce(p)
    cs: CurrentState = result.state["current_state"]
    assert cs.last_user_event["seq"] == 1
    assert cs.last_assistant_event["seq"] == 2
    assert cs.turn_count == 1


def test_reduce_filters_unknown_types(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            _msg(1, "user_message", "hi"),
            _msg(2, "tool.call", ""),
            _msg(3, "system.boot", ""),
            _msg(4, "assistant_message", "hey"),
        ],
    )
    reducer = Reducer(handlers={"user_message": handle, "assistant_message": handle})
    result = reducer.reduce(p)
    assert result.stats["events_in_ledger"] == 4
    assert result.stats["events_processed"] == 2
    assert result.stats["events_filtered"] == 2
    assert result.stats["event_types_in_ledger"] == 4
    assert result.stats["event_types_handled"] == 2
    assert result.stats["event_types_ignored"] == 2


def test_reduce_empty_ledger(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    p.write_text("")
    reducer = Reducer(handlers={"user_message": handle, "assistant_message": handle})
    result = reducer.reduce(p)
    assert result.stats["events_in_ledger"] == 0
    assert result.stats["events_processed"] == 0
    assert result.stats["events_filtered"] == 0
    cs: CurrentState = result.state["current_state"]
    assert cs.turn_count == 0
    assert cs.last_user_event is None


def test_reduce_missing_ledger_does_not_raise(tmp_path: Path) -> None:
    p = tmp_path / "nope.jsonl"
    reducer = Reducer(handlers={"user_message": handle})
    # The reducer's reduce() doesn't check existence; it relies on the
    # iterator yielding nothing from a non-existent file (which raises).
    # Confirm the behavior contract for the caller.
    with pytest.raises(FileNotFoundError):
        reducer.reduce(p)


def test_reduce_calls_handlers_in_ledger_order(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            _msg(1, "user_message", "u1"),
            _msg(2, "user_message", "u2"),
            _msg(3, "user_message", "u3"),
        ],
    )
    reducer = Reducer(handlers={"user_message": handle})
    cs: CurrentState = reducer.reduce(p).state["current_state"]
    assert cs.last_user_event["seq"] == 3


def test_reduce_filters_events_with_no_type(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    p.write_text(
        json.dumps({"event_id": "x", "seq": 1, "payload": {}}) + "\n"
    )  # no "type" field
    reducer = Reducer(handlers={"user_message": handle})
    result = reducer.reduce(p)
    assert result.stats["events_filtered"] == 1
    assert result.stats["events_processed"] == 0


# ── current_state handler ──────────────────────────────────────────────────


def test_current_state_initial_is_empty() -> None:
    cs = initial()
    assert cs.last_user_event is None
    assert cs.last_assistant_event is None
    assert cs.turn_count == 0


def test_current_state_user_then_assistant() -> None:
    cs: CurrentState = initial()
    handle(_msg(1, "user_message", "hi"), cs)
    assert cs.last_user_event is not None
    assert cs.last_assistant_event is None
    assert cs.turn_count == 0

    handle(_msg(2, "assistant_message", "hello"), cs)
    assert cs.last_assistant_event is not None
    assert cs.turn_count == 1


def test_current_state_trailing_user_does_not_bump_turns() -> None:
    cs: CurrentState = initial()
    handle(_msg(1, "user_message"), cs)
    handle(_msg(2, "assistant_message"), cs)
    handle(_msg(3, "user_message"), cs)  # trailing
    assert cs.turn_count == 1  # still 1, not 2
    assert cs.last_user_event["seq"] == 3
    assert cs.last_assistant_event["seq"] == 2


# ── integration with the live ledger shape ─────────────────────────────────


def test_reduce_against_real_event_names(tmp_path: Path) -> None:
    """The names the live luna-server actually emits."""
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            {
                "event_id": "u1",
                "seq": 1,
                "timestamp": "2026-07-21T12:27:01.710Z",
                "type": "user_message",
                "actor": "user",
                "payload": {"text": "hello", "session_id": "abc"},
            },
            {
                "event_id": "a1",
                "seq": 2,
                "timestamp": "2026-07-21T12:27:04.011Z",
                "type": "assistant_message",
                "actor": "luna",
                "payload": {
                    "text": "Hello. How can I help you today?",
                    "session_id": "abc",
                    "provider": "whooshd",
                    "model": "mlx-community/gemma-4-26B-A4B-it-4bit",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 36, "completion_tokens": 10, "total_tokens": 46},
                    "reply_to_event_id": "u1",
                },
            },
        ],
    )
    reducer = Reducer(
        handlers={"user_message": handle, "assistant_message": handle}
    )
    result = reducer.reduce(p)
    assert result.stats["events_processed"] == 2
    cs: CurrentState = result.state["current_state"]
    assert cs.last_user_event["payload"]["text"] == "hello"
    assert cs.last_assistant_event["payload"]["provider"] == "whooshd"
    assert cs.turn_count == 1
