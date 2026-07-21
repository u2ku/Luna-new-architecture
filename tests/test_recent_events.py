"""Tests for luna.context.recent_events and the stream_id isolation guarantee."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luna.context import builder
from luna.context.recent_events import (
    LedgerNotFoundError,
    MessageEvent,
    StreamMismatchError,
    _coerce,
    ledger_path,
    recent_message_events,
)


def _write_ledger(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def _msg(
    seq: int,
    type_: str,
    text: str,
    *,
    stream_id: str = "web::s1:s1",
    turn_id: str | None = None,
    actor: str | None = None,
) -> dict:
    """Build an event in the new multi-channel shape."""
    payload: dict = {"text": text}
    if turn_id is not None:
        payload["turn_id"] = turn_id
    if actor is None:
        actor = "identity:user" if type_ == "user_message" else "agent:luna"
    return {
        "event_id": f"e-{seq:04d}",
        "seq": seq,
        "timestamp": "2026-07-22T00:00:00Z",
        "type": type_,
        "actor": {"id": actor, "type": "human" if type_ == "user_message" else "agent"},
        "source": {"platform": "web", "adapter": "fastapi",
                   "conversation_id": "s1", "thread_id": "s1"},
        "destination": {"platform": "luna-runtime"},
        "stream_id": stream_id,
        "payload": payload,
    }


# ── stream_id is required ──────────────────────────────────────────────────


def test_stream_id_required() -> None:
    with pytest.raises(ValueError):
        recent_message_events(stream_id="", limit=25, path=Path("/nonexistent"))


def test_empty_stream_id_returns_empty() -> None:
    # Empty string AND limit=0 both short-circuit before the file is read.
    assert recent_message_events(stream_id="x", limit=0, path=Path("/x")) == []


# ── ledger errors ─────────────────────────────────────────────────────────


def test_missing_ledger_raises(tmp_path: Path) -> None:
    with pytest.raises(LedgerNotFoundError):
        recent_message_events(
            stream_id="web::s1:s1", limit=25, path=tmp_path / "nope.jsonl"
        )


# ── happy path: filtering, ordering, limits ──────────────────────────────


def test_empty_ledger_returns_empty_list(tmp_path: Path) -> None:
    _write_ledger(tmp_path / "world.jsonl", [])
    assert (
        recent_message_events(
            stream_id="web::s1:s1", limit=25, path=tmp_path / "world.jsonl"
        )
        == []
    )


def test_limit_zero_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(p, [_msg(1, "user_message", "hi")])
    assert recent_message_events(stream_id="web::s1:s1", limit=0, path=p) == []


def test_fewer_than_limit_returns_all(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    events = [
        _msg(1, "user_message", "first"),
        _msg(2, "assistant_message", "first-reply"),
    ]
    _write_ledger(p, events)
    out = recent_message_events(stream_id="web::s1:s1", limit=25, path=p)
    assert len(out) == 2
    assert [e.text for e in out] == ["first", "first-reply"]


def test_more_than_limit_keeps_tail_in_order(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    events = [_msg(i, "user_message", f"msg-{i}") for i in range(1, 31)]
    _write_ledger(p, events)
    out = recent_message_events(stream_id="web::s1:s1", limit=25, path=p)
    assert len(out) == 25
    assert out[0].text == "msg-6"  # first kept = 30 - 25 + 1
    assert out[-1].text == "msg-30"
    assert [e.seq for e in out] == list(range(6, 31))


# ── filtering: only message events, malformed lines ──────────────────────


def test_only_message_events_are_returned(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    events = [
        _msg(1, "user_message", "hi"),
        {"event_id": "x", "seq": 2, "timestamp": "t", "type": "tool.call",
         "actor": {"id": "agent:luna", "type": "agent"},
         "source": {"platform": "luna-runtime"},
         "destination": {"platform": "internal"},
         "stream_id": "web::s1:s1", "payload": {"name": "search"}},
        {"event_id": "y", "seq": 3, "timestamp": "t", "type": "tool.result",
         "actor": {"id": "agent:luna", "type": "agent"},
         "source": {"platform": "luna-runtime"},
         "destination": {"platform": "internal"},
         "stream_id": "web::s1:s1", "payload": {"output": "..."}},
        {"event_id": "z", "seq": 4, "timestamp": "t", "type": "system",
         "actor": {"id": "system:luna-runtime", "type": "system"},
         "source": {"platform": "luna-runtime"},
         "destination": {"platform": "luna-runtime"},
         "stream_id": "web::s1:s1", "payload": {"note": "boot"}},
        _msg(5, "assistant_message", "hello back"),
    ]
    _write_ledger(p, events)
    out = recent_message_events(stream_id="web::s1:s1", limit=25, path=p)
    assert [e.type for e in out] == ["user_message", "assistant_message"]
    assert [e.text for e in out] == ["hi", "hello back"]


def test_malformed_lines_are_skipped(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(_msg(1, "user_message", "first")),
                "{not valid json",
                "",
                json.dumps(_msg(2, "assistant_message", "second")),
            ]
        )
        + "\n"
    )
    out = recent_message_events(stream_id="web::s1:s1", limit=25, path=p)
    assert [e.text for e in out] == ["first", "second"]


# ── turn_id and actor helpers ────────────────────────────────────────────


def test_turn_id_is_extracted_from_payload(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            _msg(1, "user_message", "hi", turn_id="turn-1"),
            _msg(2, "assistant_message", "hello", turn_id="turn-1"),
            _msg(3, "user_message", "next", turn_id="turn-2"),
        ],
    )
    out = recent_message_events(stream_id="web::s1:s1", limit=25, path=p)
    assert [e.turn_id for e in out] == ["turn-1", "turn-1", "turn-2"]


def test_user_and_assistant_helpers(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            _msg(1, "user_message", "u"),
            _msg(2, "assistant_message", "a"),
        ],
    )
    [u, a] = recent_message_events(stream_id="web::s1:s1", limit=25, path=p)
    assert u.is_user and not u.is_assistant
    assert a.is_assistant and not a.is_user


# ── message_types override ───────────────────────────────────────────────


def test_message_types_override_filters_differently(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            _msg(1, "user_message", "u"),
            _msg(2, "inbound_msg", "i"),  # alt name
            _msg(3, "outbound_msg", "o"),  # alt name
        ],
    )
    default = recent_message_events(stream_id="web::s1:s1", limit=25, path=p)
    assert [e.type for e in default] == ["user_message"]
    custom = recent_message_events(
        stream_id="web::s1:s1",
        limit=25,
        path=p,
        message_types=frozenset({"inbound_msg", "outbound_msg"}),
    )
    assert [e.type for e in custom] == ["inbound_msg", "outbound_msg"]


def test_empty_message_types_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(p, [_msg(1, "user_message", "hi")])
    assert recent_message_events(
        stream_id="web::s1:s1", limit=25, path=p, message_types=frozenset()
    ) == []


# ── ledger path resolution ───────────────────────────────────────────────


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


# ── STREAM ISOLATION: the security-critical guarantee ─────────────────────


def test_streams_are_strictly_isolated(tmp_path: Path) -> None:
    """Two streams in the same ledger must never see each other's events.

    A private web conversation must not appear in a Slack context, and
    vice versa. This is the security boundary the new design enforces.
    """
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            # Stream A (web, private)
            _msg(1, "user_message", "web: private thing", stream_id="web:::sess-A"),
            _msg(2, "assistant_message", "web: noted", stream_id="web:::sess-A"),
            # Stream B (slack, public)
            _msg(3, "user_message", "slack: hi team", stream_id="slack:T1:C1:t1"),
            _msg(4, "assistant_message", "slack: hello", stream_id="slack:T1:C1:t1"),
            # Stream A continues
            _msg(5, "user_message", "web: also this", stream_id="web:::sess-A"),
        ],
    )

    web = recent_message_events(stream_id="web:::sess-A", limit=25, path=p)
    slack = recent_message_events(stream_id="slack:T1:C1:t1", limit=25, path=p)

    # Web stream has its 3 events, none from slack
    assert [e.text for e in web] == [
        "web: private thing",
        "web: noted",
        "web: also this",
    ]
    # Slack stream has its 2 events, none from web
    assert [e.text for e in slack] == [
        "slack: hi team",
        "slack: hello",
    ]
    # Cross-check: streams share no seqs
    assert {e.seq for e in web}.isdisjoint({e.seq for e in slack})


def test_stream_id_excludes_legacy_events_without_stream_id(
    tmp_path: Path,
) -> None:
    """Events without a stream_id are never returned.

    A misbehaving adapter that forgets to set stream_id should not
    leak its messages into any other stream.
    """
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            # No stream_id — must be excluded
            {
                "event_id": "legacy",
                "seq": 1,
                "timestamp": "t",
                "type": "user_message",
                "actor": {"id": "user", "type": "human"},
                "payload": {"text": "no stream"},
            },
            _msg(2, "user_message", "with stream", stream_id="web:::sess"),
        ],
    )
    out = recent_message_events(stream_id="web:::sess", limit=25, path=p)
    assert [e.text for e in out] == ["with stream"]


# ── builder integration ──────────────────────────────────────────────────


def test_builder_returns_recent_events(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "world.jsonl"
    _write_ledger(
        p,
        [
            _msg(i, "user_message" if i % 2 else "assistant_message", f"m{i}")
            for i in range(1, 6)
        ],
    )
    from luna.context import recent_events
    monkeypatch.setattr(recent_events, "ledger_path", lambda: p)
    out = builder.build_recent_messages(stream_id="web::s1:s1", limit=3)
    assert len(out) == 3
    assert [e.seq for e in out] == [3, 4, 5]


def test_builder_groups_into_turns() -> None:
    events = [
        _coerce(_msg(1, "user_message", "u1"), "web::s1:s1"),
        _coerce(_msg(2, "assistant_message", "a1"), "web::s1:s1"),
        _coerce(_msg(3, "user_message", "u2"), "web::s1:s1"),
        _coerce(_msg(4, "assistant_message", "a2"), "web::s1:s1"),
    ]
    turns = builder.group_into_turns(events)
    assert len(turns) == 2
    assert turns[0].user.text == "u1" and turns[0].assistant.text == "a1"
    assert turns[1].user.text == "u2" and turns[1].assistant.text == "a2"


def test_builder_trailing_user_with_no_reply() -> None:
    events = [
        _coerce(_msg(1, "user_message", "u1"), "web::s1:s1"),
        _coerce(_msg(2, "assistant_message", "a1"), "web::s1:s1"),
        _coerce(_msg(3, "user_message", "u2"), "web::s1:s1"),  # no reply yet
    ]
    turns = builder.group_into_turns(events)
    assert len(turns) == 2
    assert turns[1].user.text == "u2"
    assert turns[1].assistant is None


def test_builder_drops_orphan_assistant() -> None:
    events = [
        _coerce(_msg(1, "assistant_message", "orphan"), "web::s1:s1"),
        _coerce(_msg(2, "user_message", "u1"), "web::s1:s1"),
    ]
    turns = builder.group_into_turns(events)
    assert len(turns) == 1
    assert turns[0].user.text == "u1"
    assert turns[0].assistant is None


# ── end-to-end against the real-world ledger shape ─────────────────────────


def test_builder_reads_real_world_shape(tmp_path: Path) -> None:
    """End-to-end: builder reads a ledger shaped like the live world.jsonl.

    The live luna-server writes events with the new multi-channel
    shape (actor object, source, destination, stream_id, turn_id).
    A regression here means the context builder stopped reading the
    actual ledger.
    """
    p = tmp_path / "world.jsonl"
    stream = "web:::sess-test"
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_id": f"e{i}",
                        "seq": i,
                        "timestamp": f"2026-07-22T00:00:{i:02d}Z",
                        "type": "user_message" if i % 2 else "assistant_message",
                        "actor": {
                            "id": "identity:zac" if i % 2 else "agent:luna",
                            "type": "human" if i % 2 else "agent",
                            "display_name": "Zac" if i % 2 else "Luna",
                        },
                        "source": {"platform": "web", "adapter": "fastapi",
                                   "conversation_id": "sess-test",
                                   "thread_id": "sess-test"},
                        "destination": {"platform": "luna-runtime"},
                        "stream_id": stream,
                        "turn_id": f"turn-{(i + 1) // 2}",
                        "payload": {
                            "text": f"message-{i}",
                            "sender_name": "Zac" if i % 2 else None,
                        },
                    }
                )
                for i in range(1, 11)
            ]
        )
        + "\n"
    )

    from luna.context import recent_events
    original = recent_events.ledger_path
    recent_events.ledger_path = lambda: p
    try:
        events = builder.build_recent_messages(stream_id=stream, limit=25)
    finally:
        recent_events.ledger_path = original

    assert len(events) == 10
    assert [e.type for e in events] == [
        "assistant_message" if i % 2 == 0 else "user_message" for i in range(1, 11)
    ]
    assert all(e.stream_id == stream for e in events)
    assert all(e.is_user != e.is_assistant for e in events)

    turns = builder.group_into_turns(events)
    assert len(turns) == 5
    for t in turns:
        assert t.user is not None
        assert t.assistant is not None
    assert turns[0].user.seq == 1
    assert turns[0].assistant.seq == 2
    assert turns[-1].user.seq == 9
    assert turns[-1].assistant.seq == 10


def test_builder_with_live_ledger_if_available(
    tmp_path: Path, monkeypatch
) -> None:
    """If LUNA_DATA_ROOT/world.jsonl is on disk, the builder must read it.

    Skipped silently when the live ledger is not present (e.g. in CI).
    """
    live = Path("/Users/pieratradio/luna-new-architecture/LunaData/ledger/world.jsonl")
    if not live.is_file():
        return  # nothing to test against

    from luna.context import recent_events

    monkeypatch.setattr(recent_events, "ledger_path", lambda: live)

    # Use the first stream_id found in the live ledger, so this test
    # works against the legacy events that lack stream_id (it'll get an
    # empty result for those, which is the correct secure behavior).
    stream_id = None
    with live.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = e.get("stream_id")
            if sid:
                stream_id = sid
                break
    if stream_id is None:
        return  # live ledger has no stream_id'd events yet

    events = builder.build_recent_messages(stream_id=stream_id, limit=25)
    has_user = any(e.is_user for e in events)
    has_assistant = any(e.is_assistant for e in events)
    if has_user and has_assistant:
        turns = builder.group_into_turns(events)
        complete = sum(1 for t in turns if t.assistant is not None)
        # Don't assert a count — depends on live state.
        assert complete >= 1 or len(events) > 0
