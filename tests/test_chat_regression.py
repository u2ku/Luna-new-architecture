"""Regression test: the non-tool chat path still works after tool wiring.

Ensures a ChatService with no registry behaves exactly as before — one
user_message, one assistant_message, a plain reply, no tool events.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from luna.api.routes import ChatRequest, ChatService
from luna.ledger import WorldLedger
from luna.models.base import (
    FinishReason,
    ModelProvider,
    ModelResponse,
    Usage,
)


class _PlainProvider(ModelProvider):
    name = "stub"

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    def complete(self, request) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            content=self._reply,
            finish_reason=FinishReason.STOP,
            usage=Usage(),
            model="stub",
        )


def test_chat_without_tools_writes_user_and_assistant(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    ledger = WorldLedger(
        path=tmp_path / "ledger" / "world.jsonl",
        lock_path=tmp_path / "locks" / "world.lock",
    )
    svc = ChatService(
        provider=_PlainProvider("Hello there."),
        ledger=ledger,
        system_prompt="You are Luna.",
        # registry intentionally None — no tools, legacy path.
        registry=None,
        archive_config=None,
        tools_config=None,
    )
    resp = svc.complete(ChatRequest(text="Hi"))
    assert resp.response == "Hello there."
    assert resp.tool_calls == 0
    assert resp.provider == "stub"

    events = ledger.tail(20)
    assert [e["type"] for e in events] == ["user_message", "assistant_message"]
    user = events[0]
    asst = events[1]
    assert user["payload"]["text"] == "Hi"
    assert asst["payload"]["text"] == "Hello there."
    assert asst["payload"]["reply_to_event_id"] == user["event_id"]
    assert user["stream_id"] == asst["stream_id"]
    assert user["turn_id"] == asst["turn_id"]


def test_chat_isolates_streams(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    ledger = WorldLedger(
        path=tmp_path / "ledger" / "world.jsonl",
        lock_path=tmp_path / "locks" / "world.lock",
    )
    svc = ChatService(
        provider=_PlainProvider("ok"),
        ledger=ledger,
        system_prompt="You are Luna.",
        registry=None,
    )
    # Two different sessions -> two different streams, no cross-context.
    r1 = svc.complete(ChatRequest(text="one", session_id="s1"))
    r2 = svc.complete(ChatRequest(text="two", session_id="s2"))
    assert r1.stream_id != r2.stream_id
    assert r1.stream_id.startswith("web::s1:")
    assert r2.stream_id.startswith("web::s2:")
