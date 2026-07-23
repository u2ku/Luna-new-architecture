"""Tests for receipts: pairing, sanitisation, bounded arguments."""

from __future__ import annotations

from pathlib import Path

import pytest

from luna.archive.search import SEARCH_ARCHIVE_SPEC, handle_search_archive
from luna.archive.artifact_writer import CREATE_ARTIFACT_SPEC
from luna.tools.executor import execute_with_receipts, build_archive_registry
from luna.tools.protocol import ToolContext, ToolRequest
from luna.tools.receipts import (
    bounded_arguments,
    build_tool_call_payload,
    build_tool_result_payload,
    redact_secrets,
)
from luna.ledger import WorldLedger


def _ctx(root: Path | None, output_root: Path) -> ToolContext:
    return ToolContext(
        archive_root=root,
        artifact_output_root=output_root,
        search_default_limit=8,
        search_max_limit=20,
        read_default_lines=200,
        read_max_lines=500,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
        stream_id="web::test:test",
        turn_id="turn-1",
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_bounded_arguments_drops_undeclared_and_truncates():
    spec = CREATE_ARTIFACT_SPEC
    args = {
        "title": "T",
        "content": "x" * 5000,
        "category": "luna-system",
        "source_event_ids": ["e1", "e2"],
        "smuggled_path": "/etc/passwd",  # undeclared, must be dropped
    }
    bounded = bounded_arguments(args, spec)
    assert "smuggled_path" not in bounded
    assert len(bounded["content"]) <= 201  # truncated + ellipsis
    assert bounded["content_length"] == 5000


def test_redact_secrets():
    assert "[REDACTED]" in redact_secrets("password=hunter2secret")
    assert "[REDACTED]" in redact_secrets("Authorization: Bearer " + "a" * 40)
    assert redact_secrets("plain text") == "plain text"


def test_call_payload_records_intent_only():
    req = ToolRequest(
        name="create_artifact",
        arguments={"title": "T", "content": "body"},
        call_id="c1",
    )
    payload = build_tool_call_payload(req, CREATE_ARTIFACT_SPEC)
    assert payload["tool"] == "create_artifact"
    assert payload["call_id"] == "c1"
    assert "arguments" in payload


def test_result_payload_omits_content(tmp_path):
    from luna.tools.protocol import ToolResult

    result = ToolResult(
        call_id="c1", name="search_archive", ok=True,
        content={"big": "x" * 1000}, artifact_ids=("archive:abc",),
    )
    payload = build_tool_result_payload(
        result, "call-event-1", started_at="2026-07-23T00:00:00.000Z",
        finished_at="2026-07-23T00:00:01.000Z",
    )
    assert payload["call_event_id"] == "call-event-1"
    assert payload["status"] == "ok"
    assert payload["artifact_ids"] == ["archive:abc"]
    assert payload["started_at"] == "2026-07-23T00:00:00.000Z"
    assert payload["finished_at"] == "2026-07-23T00:00:01.000Z"
    # common envelope fields are always present
    assert payload["error_code"] is None
    assert payload["result_summary"]
    assert payload["affected_resources"] == ["archive:abc"]
    assert "content" not in payload  # never persisted


def test_tool_call_result_pairing(tmp_path):
    root = tmp_path / "archive"
    _write(root / "doc.md", "# Hull\nproject hull sensor certification")
    ledger = WorldLedger(path=tmp_path / "world.jsonl", lock_path=tmp_path / "w.lock")
    ctx = _ctx(root, tmp_path / "artifacts")
    reg = build_archive_registry()

    result = execute_with_receipts(
        reg,
        ToolRequest(
            name="search_archive",
            arguments={"query": "hull sensor", "limit": 3},
            call_id="c1",
        ),
        ctx,
        ledger,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
    )
    assert result.ok is True

    events = ledger.tail(10)
    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    call = next(e for e in events if e["type"] == "tool_call")
    res = next(e for e in events if e["type"] == "tool_result")
    # the result references its call event
    assert res["payload"]["call_event_id"] == call["event_id"]
    assert res["payload"]["tool"] == "search_archive"
    assert res["payload"]["status"] == "ok"
    assert res["stream_id"] == "web::test:test"
    assert res["turn_id"] == "turn-1"


def test_receipt_sanitises_full_content(tmp_path):
    root = tmp_path / "archive"
    _write(root / "doc.md", "# Hull\nproject hull sensor")
    ledger = WorldLedger(path=tmp_path / "world.jsonl", lock_path=tmp_path / "w.lock")
    ctx = _ctx(root, tmp_path / "artifacts")
    reg = build_archive_registry()

    execute_with_receipts(
        reg,
        ToolRequest(
            name="create_artifact",
            arguments={
                "title": "Decision",
                "content": "A" * 4000,  # large body
                "category": "luna-system",
            },
            call_id="c1",
        ),
        ctx,
        ledger,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
    )
    events = ledger.tail(10)
    call = next(e for e in events if e["type"] == "tool_call")
    # The full 4000-char content must not appear in the receipt.
    assert "A" * 4000 not in __import__("json").dumps(call["payload"])
    # Only a bounded form is kept.
    assert len(call["payload"]["arguments"]["content"]) <= 201


def test_error_receipt_paired(tmp_path):
    ledger = WorldLedger(path=tmp_path / "world.jsonl", lock_path=tmp_path / "w.lock")
    ctx = _ctx(None, tmp_path / "artifacts")  # no archive root
    reg = build_archive_registry()
    result = execute_with_receipts(
        reg,
        ToolRequest(
            name="read_artifact",
            arguments={"artifact_id": "archive:deadbeef"},
            call_id="c1",
        ),
        ctx,
        ledger,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
    )
    assert result.ok is False
    events = ledger.tail(10)
    res = next(e for e in events if e["type"] == "tool_result")
    assert res["payload"]["status"] == "error"
    assert res["payload"]["error_code"] == "archive_unavailable"
    # common envelope fields present even on error
    assert res["payload"]["started_at"]
    assert res["payload"]["finished_at"]
    assert res["payload"]["result_summary"]


# ---------------------------------------------------------------------------
# Per-tool receipt digests (enriched tool_result fields)
# ---------------------------------------------------------------------------


def _result_event(ledger, tool_name):
    events = ledger.tail(50)
    return next(
        e for e in events
        if e["type"] == "tool_result" and e["payload"]["tool"] == tool_name
    )


def test_search_result_receipt_fields(tmp_path):
    root = tmp_path / "archive"
    _write(root / "project-hull" / "sensors.md",
           "# Hull Sensor Stack\nproject hull sensor stack certification")
    ledger = WorldLedger(path=tmp_path / "world.jsonl", lock_path=tmp_path / "w.lock")
    ctx = _ctx(root, tmp_path / "artifacts")
    reg = build_archive_registry()

    result = execute_with_receipts(
        reg,
        ToolRequest(name="search_archive",
                    arguments={"query": "hull sensor", "limit": 3}, call_id="c1"),
        ctx, ledger,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
    )
    payload = _result_event(ledger, "search_archive")["payload"]
    # common envelope
    assert payload["status"] == "ok"
    assert payload["error_code"] is None
    assert payload["started_at"] and payload["finished_at"]
    assert "hull sensor" in payload["result_summary"]
    assert payload["affected_resources"] == list(result.artifact_ids)
    # per-tool digest
    assert payload["receipt"]["query"] == "hull sensor"
    assert payload["receipt"]["result_count"] == 1
    top = payload["receipt"]["top_results"]
    assert len(top) == 1
    assert {"artifact_id", "title", "relative_path", "score"} == set(top[0])
    # no excerpt / matched_terms / full content leaks into the receipt
    blob = __import__("json").dumps(payload)
    assert "excerpt" not in blob
    assert "matched_terms" not in blob
    assert "sensor stack certification" not in blob


def test_read_result_receipt_fields(tmp_path):
    root = tmp_path / "archive"
    body = "\n".join(f"line {i}" for i in range(1, 21))
    _write(root / "doc.md", "# Title\n" + body)
    ledger = WorldLedger(path=tmp_path / "world.jsonl", lock_path=tmp_path / "w.lock")
    ctx = _ctx(root, tmp_path / "artifacts")
    reg = build_archive_registry()

    # find the artifact_id via a quick search
    from luna.archive.search import search_archive
    res = search_archive("line", root, limit=1)
    aid = res["results"][0]["artifact_id"]

    result = execute_with_receipts(
        reg,
        ToolRequest(name="read_artifact",
                    arguments={"artifact_id": aid, "line_count": 30}, call_id="c1"),
        ctx, ledger,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
    )
    payload = _result_event(ledger, "read_artifact")["payload"]
    assert payload["status"] == "ok"
    assert payload["receipt"]["artifact_id"] == aid
    assert payload["receipt"]["relative_path"] == "doc.md"
    assert payload["receipt"]["start_line"] == 1
    assert payload["receipt"]["end_line"] == 21  # whole 21-line file
    assert payload["receipt"]["truncated"] is True  # asked for 30, only 21 remained
    # content_hash matches a sha256 of the returned text
    import hashlib
    expected = hashlib.sha256(
        "\n".join(e["text"] for e in result.content["content"]).encode("utf-8")
    ).hexdigest()
    assert payload["receipt"]["content_hash"] == expected
    assert payload["receipt"]["characters_returned"] > 0
    # the actual line text must not appear in the receipt
    blob = __import__("json").dumps(payload)
    assert "line 5" not in blob


def test_create_result_receipt_fields(tmp_path):
    ledger = WorldLedger(path=tmp_path / "world.jsonl", lock_path=tmp_path / "w.lock")
    ctx = _ctx(tmp_path / "archive", tmp_path / "artifacts")
    reg = build_archive_registry()

    result = execute_with_receipts(
        reg,
        ToolRequest(name="create_artifact",
                    arguments={"title": "Decision",
                               "content": "We chose retrieval-based archive.",
                               "category": "luna-system"},
                    call_id="c1"),
        ctx, ledger,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
    )
    payload = _result_event(ledger, "create_artifact")["payload"]
    assert payload["status"] == "ok"
    assert payload["receipt"]["title"] == "Decision"
    assert payload["receipt"]["category"] == "luna-system"
    assert payload["receipt"]["relative_path"].endswith(".md")
    assert payload["receipt"]["bytes_written"] > 0
    # content_hash matches the sha256 of the file actually written
    import hashlib
    written = (tmp_path / "artifacts" / payload["receipt"]["relative_path"]).read_bytes()
    assert payload["receipt"]["content_hash"] == hashlib.sha256(written).hexdigest()
    # the body text must not appear in the receipt
    blob = __import__("json").dumps(payload)
    assert "retrieval-based archive" not in blob


def test_receipt_never_carries_full_content(tmp_path):
    """Cross-check: no tool_result persists full search/read/create content."""
    root = tmp_path / "archive"
    _write(root / "doc.md", "# Hull\nthe hull sensor stack secret body text")
    ledger = WorldLedger(path=tmp_path / "world.jsonl", lock_path=tmp_path / "w.lock")
    ctx = _ctx(root, tmp_path / "artifacts")
    reg = build_archive_registry()

    execute_with_receipts(
        reg, ToolRequest(name="search_archive",
                         arguments={"query": "hull"}, call_id="c1"),
        ctx, ledger, actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
    )
    blob = __import__("json").dumps(ledger.tail(50))
    assert "secret body text" not in blob  # excerpt/body never persisted
