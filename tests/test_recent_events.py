"""Tests for luna.context.recent_events."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luna.context import builder
from luna.context.recent_events import (
    LedgerNotFoundError,
    MessageEvent,
    _coerce,
    ledger_path,
    recent_message_events,
)


def _write_ledger(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def _msg(seq: int, type_: str, text: str, *, turn_id: str | None = None) -> dict:
    payload: dict = {"text": text}
    if turn_id is not None:
        payload["turn_id"] = turn_id
    return {
        "event_id": f"e-{seq:04d}",
        "seq": seq,
        "timestamp": "2026-07-22T00:00:00Z",
        "type": type_,
        "actor": "user" if type_ == "message.in" else "luna",
        "payload": payload,
    }


def test_empty_ledger_returns_empty_list(tmp_path: Path) -> None:
    _write_ledger(tmp_path / "world.jsonl", [])
    assert recent_message_events(limit=25, path=tmp_path / "world.jsonl") == []


def test_missing_ledger_raises(tmp_path: Path) -> None:
    with pytest.raises(LedgerNotFoundError):
        recent_message_events(limit=25, path=tmp_path / "nope.jsonl")


def test_limit_zero_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(p, [_msg(1, "message.in", "hi")])
    assert recent_message_events(limit=0, path=p) == []


def test_fewer_than_limit_returns_all(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    events = [
        _msg(1, "message.in", "first"),
        _msg(2, "message.out", "first-reply"),
    ]
    _write_ledger(p, events)
    out = recent_message_events(limit=25, path=p)
    assert len(out) == 2
    assert [e.text for e in out] == ["first", "first-reply"]


def test_more_than_limit_keeps_tail_in_order(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    events = [_msg(i, "message.in", f"msg-{i}") for i in range(1, 31)]
    _write_ledger(p, events)
    out = recent_message_events(limit=25, path=p)
    assert len(out) == 25
    assert out[0].text == "msg-6"  # first kept = 30 - 25 + 1
    assert out[-1].text == "msg-30"
    assert [e.seq for e in out] == list(range(6, 31))


def test_only_message_events_are_returned(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    events = [
        _msg(1, "message.in", "hi"),
        {"event_id": "x", "seq": 2, "timestamp": "t", "type": "tool.call",
         "actor": "luna", "payload": {"name": "search"}},
        {"event_id": "y", "seq": 3, "timestamp": "t", "type": "tool.result",
         "actor": "luna", "payload": {"output": "..."}},
        {"event_id": "z", "seq": 4, "timestamp": "t", "type": "system",
         "actor": "runtime", "payload": {"note": "boot"}},
        _msg(5, "message.out", "hello back"),
    ]
    _write_ledger(p, events)
    out = recent_message_events(limit=25, path=p)
    assert [e.type for e in out] == ["message.in", "message.out"]
    assert [e.text for e in out] == ["hi", "hello back"]


def test_malformed_lines_are_skipped(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(_msg(1, "message.in", "first")),
                "{not valid json",
                "",
                json.dumps(_msg(2, "message.out", "second")),
            ]
        )
        + "\n"
    )
    out = recent_message_events(limit=25, path=p)
    assert [e.text for e in out] == ["first", "second"]


def test_turn_id_is_extracted_from_payload(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            _msg(1, "message.in", "hi", turn_id="turn-1"),
            _msg(2, "message.out", "hello", turn_id="turn-1"),
            _msg(3, "message.in", "next", turn_id="turn-2"),
        ],
    )
    out = recent_message_events(limit=25, path=p)
    assert [e.turn_id for e in out] == ["turn-1", "turn-1", "turn-2"]


def test_user_and_assistant_helpers(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            _msg(1, "message.in", "u"),
            _msg(2, "message.out", "a"),
        ],
    )
    [u, a] = recent_message_events(limit=25, path=p)
    assert u.is_user and not u.is_assistant
    assert a.is_assistant and not a.is_user


def test_ledger_path_uses_luna_data_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    p = ledger_path()
    assert p == tmp_path / "ledger" / "world.jsonl"


def test_ledger_path_falls_back_when_env_unset(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("LUNA_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    p = ledger_path()
    assert p == Path("ledger") / "world.jsonl"


# ── builder integration ─────────────────────────────────────────────────────


def test_builder_returns_recent_events(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [_msg(i, "message.in" if i % 2 else "message.out", f"m{i}") for i in range(1, 6)],
    )
    monkeypatch.setattr("luna.context.recent_events.ledger_path", lambda: p)
    out = builder.build_recent_messages(limit=3)
    assert len(out) == 3
    assert [e.seq for e in out] == [3, 4, 5]


def test_builder_groups_into_turns() -> None:
    events = [
        _coerce(_msg(1, "message.in", "u1")),
        _coerce(_msg(2, "message.out", "a1")),
        _coerce(_msg(3, "message.in", "u2")),
        _coerce(_msg(4, "message.out", "a2")),
    ]
    turns = builder.group_into_turns(events)
    assert len(turns) == 2
    assert turns[0].user.text == "u1" and turns[0].assistant.text == "a1"
    assert turns[1].user.text == "u2" and turns[1].assistant.text == "a2"


def test_builder_trailing_user_with_no_reply() -> None:
    events = [
        _coerce(_msg(1, "message.in", "u1")),
        _coerce(_msg(2, "message.out", "a1")),
        _coerce(_msg(3, "message.in", "u2")),  # no reply yet
    ]
    turns = builder.group_into_turns(events)
    assert len(turns) == 2
    assert turns[1].user.text == "u2"
    assert turns[1].assistant is None


def test_builder_drops_orphan_assistant() -> None:
    events = [
        _coerce(_msg(1, "message.out", "orphan")),  # no preceding user
        _coerce(_msg(2, "message.in", "u1")),
    ]
    turns = builder.group_into_turns(events)
    assert len(turns) == 1
    assert turns[0].user.text == "u1"
    assert turns[0].assistant is None
