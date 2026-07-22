"""Tests for the structured model tool loop in ChatService."""

from __future__ import annotations

from pathlib import Path

import pytest

from luna.api.routes import ChatRequest, ChatService
from luna.ledger import WorldLedger
from luna.models.base import (
    FinishReason,
    Message,
    ModelProvider,
    ModelResponse,
    ToolCall,
    Usage,
)
from luna.tools.config import ArchiveConfig, ToolsConfig
from luna.tools.executor import build_archive_registry


def _archive_cfg(root: Path, output: Path) -> ArchiveConfig:
    return ArchiveConfig(
        root=root,
        artifact_output_root=output,
        search_default_limit=8,
        search_max_limit=20,
        read_default_lines=200,
        read_max_lines=500,
    )


def _tools_cfg() -> ToolsConfig:
    return ToolsConfig(
        enabled=["search_archive", "read_artifact", "create_artifact"],
        max_tool_calls_per_turn=6,
        max_result_chars_per_turn=20000,
    )


class _ScriptedProvider(ModelProvider):
    """Returns canned responses in order; asserts tools were forwarded."""

    name = "stub"

    def __init__(self, script: list[ModelResponse]) -> None:
        self._script = list(script)
        self.calls = 0
        self.last_tools: tuple = ()

    def complete(self, request) -> ModelResponse:
        self.calls += 1
        self.last_tools = request.tools
        return self._script.pop(0)


def _make_archive(tmp: Path) -> Path:
    root = tmp / "archive"
    (root / "project-hull").mkdir(parents=True, exist_ok=True)
    (root / "project-hull" / "hull-sensors.md").write_text(
        "# Hull Sensor Stack\nThe hull sensor stack for project hull.",
        encoding="utf-8",
    )
    return root


def test_structured_tool_loop_search_then_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    root = _make_archive(tmp_path)
    ledger = WorldLedger(
        path=tmp_path / "ledger" / "world.jsonl",
        lock_path=tmp_path / "locks" / "world.lock",
    )
    provider = _ScriptedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="search_archive",
                        arguments={"query": "hull sensor"},
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=Usage(),
                model="stub",
            ),
            ModelResponse(
                content="Project Hull sensor stack is documented in the archive.",
                finish_reason=FinishReason.STOP,
                usage=Usage(),
                model="stub",
            ),
        ]
    )
    svc = ChatService(
        provider=provider,
        ledger=ledger,
        system_prompt="You are Luna.",
        registry=build_archive_registry(),
        archive_config=_archive_cfg(root, tmp_path / "artifacts"),
        tools_config=_tools_cfg(),
    )
    resp = svc.complete(ChatRequest(text="What is the hull sensor stack?"))
    assert "Project Hull" in resp.response
    assert resp.tool_calls == 1
    # tools were forwarded on the first call
    assert provider.last_tools and provider.last_tools[0].name == "search_archive"

    events = ledger.tail(20)
    types = [e["type"] for e in events]
    assert types == ["user_message", "tool_call", "tool_result", "assistant_message"]
    call = next(e for e in events if e["type"] == "tool_call")
    res = next(e for e in events if e["type"] == "tool_result")
    assert res["payload"]["call_event_id"] == call["event_id"]
    assert res["payload"]["status"] == "ok"
    asst = next(e for e in events if e["type"] == "assistant_message")
    assert asst["payload"]["tool_calls"] == 1


def test_prose_tool_call_not_executed(tmp_path, monkeypatch):
    """Text mentioning a tool name is content, not a tool invocation."""
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    ledger = WorldLedger(
        path=tmp_path / "ledger" / "world.jsonl",
        lock_path=tmp_path / "locks" / "world.lock",
    )
    provider = _ScriptedProvider(
        [
            ModelResponse(
                content="You could use search_archive to find that. Reply.",
                finish_reason=FinishReason.STOP,
                usage=Usage(),
                model="stub",
            )
        ]
    )
    svc = ChatService(
        provider=provider,
        ledger=ledger,
        system_prompt="You are Luna.",
        registry=build_archive_registry(),
        archive_config=_archive_cfg(tmp_path / "archive", tmp_path / "artifacts"),
        tools_config=_tools_cfg(),
    )
    resp = svc.complete(ChatRequest(text="help"))
    assert resp.tool_calls == 0
    events = ledger.tail(20)
    assert [e["type"] for e in events] == ["user_message", "assistant_message"]


def test_max_tool_calls_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    root = _make_archive(tmp_path)
    ledger = WorldLedger(
        path=tmp_path / "ledger" / "world.jsonl",
        lock_path=tmp_path / "locks" / "world.lock",
    )
    # Provider returns tool calls for 7 turns (one beyond the cap of 6),
    # then a final reply. The 7th call must NOT be executed — the cap
    # forces tools off and the model answers on the next turn.
    script = [
        ModelResponse(
            content="",
            tool_calls=(ToolCall(id=f"c{i}", name="search_archive",
                                 arguments={"query": "hull"}),),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(),
            model="stub",
        )
        for i in range(7)
    ]
    script.append(
        ModelResponse(content="Final answer after cap.", finish_reason=FinishReason.STOP,
                      usage=Usage(), model="stub")
    )
    provider = _ScriptedProvider(script)
    svc = ChatService(
        provider=provider,
        ledger=ledger,
        system_prompt="You are Luna.",
        registry=build_archive_registry(),
        archive_config=_archive_cfg(root, tmp_path / "artifacts"),
        tools_config=_tools_cfg(),
    )
    resp = svc.complete(ChatRequest(text="again"))
    assert resp.response == "Final answer after cap."
    # Exactly 6 tool calls executed and receipted; the 7th was refused.
    tool_calls = [e for e in ledger.tail(50) if e["type"] == "tool_call"]
    assert len(tool_calls) == 6
    tool_results = [e for e in ledger.tail(50) if e["type"] == "tool_result"]
    assert len(tool_results) == 6


# ---------------------------------------------------------------------------
# Prompt-JSON transport (whooshd-style: tool calls in a sentinel block)
# ---------------------------------------------------------------------------


class _PromptScriptedProvider(ModelProvider):
    """A provider that does NOT support native tools; emits content."""

    name = "prompt-stub"
    supports_native_tools = False

    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self.calls = 0
        self.saw_tools_on_wire: tuple = ()
        self.saw_system_prompt = ""

    def complete(self, request) -> ModelResponse:
        self.calls += 1
        self.saw_tools_on_wire = request.tools
        # capture the system prompt (augmented with the protocol)
        if request.messages:
            self.saw_system_prompt = request.messages[0].content
        content = self._script.pop(0)
        return ModelResponse(content=content, usage=Usage(), model="prompt-stub")


def test_prompt_json_loop_search_then_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    root = _make_archive(tmp_path)
    ledger = WorldLedger(
        path=tmp_path / "ledger" / "world.jsonl",
        lock_path=tmp_path / "locks" / "world.lock",
    )
    provider = _PromptScriptedProvider(
        [
            # turn 1: a tool call in a sentinel block
            '```tool_call\n{"tool": "search_archive", "arguments": {"query": "hull sensor"}}\n```',
            # turn 2: final prose answer
            "The hull sensor stack is the biometric input layer for Project Hull.",
        ]
    )
    svc = ChatService(
        provider=provider,
        ledger=ledger,
        system_prompt="You are Luna.",
        registry=build_archive_registry(),
        archive_config=_archive_cfg(root, tmp_path / "artifacts"),
        tools_config=_tools_cfg(),
    )
    resp = svc.complete(ChatRequest(text="What is the hull sensor stack?"))
    assert "hull sensor stack" in resp.response
    assert resp.tool_calls == 1
    # tools were NOT sent on the wire (prompt transport)
    assert provider.saw_tools_on_wire == ()
    # the system prompt was augmented with the protocol
    assert "tool_call" in provider.saw_system_prompt
    assert "search_archive" in provider.saw_system_prompt

    events = ledger.tail(20)
    types = [e["type"] for e in events]
    assert types == ["user_message", "tool_call", "tool_result", "assistant_message"]
    call = next(e for e in events if e["type"] == "tool_call")
    res = next(e for e in events if e["type"] == "tool_result")
    assert call["payload"]["tool"] == "search_archive"
    assert res["payload"]["call_event_id"] == call["event_id"]
    assert res["payload"]["status"] == "ok"


def test_prompt_json_prose_not_executed(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    ledger = WorldLedger(
        path=tmp_path / "ledger" / "world.jsonl",
        lock_path=tmp_path / "locks" / "world.lock",
    )
    provider = _PromptScriptedProvider(
        [
            # prose mentioning a tool name — NOT a sentinel block
            "You could call search_archive, but I will just answer directly.",
        ]
    )
    svc = ChatService(
        provider=provider,
        ledger=ledger,
        system_prompt="You are Luna.",
        registry=build_archive_registry(),
        archive_config=_archive_cfg(tmp_path / "archive", tmp_path / "artifacts"),
        tools_config=_tools_cfg(),
    )
    resp = svc.complete(ChatRequest(text="hi"))
    assert resp.tool_calls == 0
    assert [e["type"] for e in ledger.tail(20)] == ["user_message", "assistant_message"]


def test_prompt_json_malformed_then_repair(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    root = _make_archive(tmp_path)
    ledger = WorldLedger(
        path=tmp_path / "ledger" / "world.jsonl",
        lock_path=tmp_path / "locks" / "world.lock",
    )
    provider = _PromptScriptedProvider(
        [
            # turn 1: malformed JSON in the sentinel block
            '```tool_call\n{"tool": "search_archive", "arguments": {bad}}\n```',
            # turn 2: a valid tool call (after the repair message)
            '```tool_call\n{"tool": "search_archive", "arguments": {"query": "hull"}}\n```',
            # turn 3: final answer
            "Done — I used the archive.",
        ]
    )
    svc = ChatService(
        provider=provider,
        ledger=ledger,
        system_prompt="You are Luna.",
        registry=build_archive_registry(),
        archive_config=_archive_cfg(root, tmp_path / "artifacts"),
        tools_config=_tools_cfg(),
    )
    resp = svc.complete(ChatRequest(text="look it up"))
    assert resp.response == "Done — I used the archive."
    assert resp.tool_calls == 1  # the valid call executed; the malformed one did not
    # exactly one tool_call/tool_result pair (the malformed attempt was a
    # repair message, not an executed/receipted call)
    calls = [e for e in ledger.tail(50) if e["type"] == "tool_call"]
    results = [e for e in ledger.tail(50) if e["type"] == "tool_result"]
    assert len(calls) == 1
    assert len(results) == 1
    assert calls[0]["payload"]["tool"] == "search_archive"
    assert provider.calls == 3  # malformed, valid, final
