"""Tests for the minimal tool registry: dispatch, validation, state."""

from __future__ import annotations

from pathlib import Path

import pytest

from luna.tools.protocol import (
    ToolAccess,
    ToolContext,
    ToolError,
    ToolRequest,
    ToolResult,
    ToolSpec,
)
from luna.tools.registry import ToolRegistry


def _ctx() -> ToolContext:
    return ToolContext(
        archive_root=Path("/nonexistent"),
        artifact_output_root=Path("/tmp/luna-test"),
        search_default_limit=8,
        search_max_limit=20,
        read_default_lines=200,
        read_max_lines=500,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
        stream_id="web::t:t",
        turn_id="turn-1",
    )


ECHO_SPEC = ToolSpec(
    name="echo",
    description="echo the argument",
    input_schema={
        "type": "object",
        "required": ["msg"],
        "additionalProperties": False,
        "properties": {"msg": {"type": "string"}},
    },
    access=ToolAccess.READ,
    enabled=True,
)


def _echo_handler(request: ToolRequest, context: ToolContext) -> ToolResult:
    return ToolResult(
        call_id=request.call_id,
        name="echo",
        ok=True,
        content={"echoed": request.arguments.get("msg")},
    )


def test_register_get_list():
    reg = ToolRegistry()
    reg.register(ECHO_SPEC, _echo_handler)
    assert reg.get("echo") is ECHO_SPEC
    assert [s.name for s in reg.list()] == ["echo"]
    assert reg.function_schemas()[0]["name"] == "echo"


def test_execute_dispatches():
    reg = ToolRegistry()
    reg.register(ECHO_SPEC, _echo_handler)
    result = reg.execute(
        ToolRequest(name="echo", arguments={"msg": "hi"}, call_id="c1"),
        _ctx(),
    )
    assert result.ok is True
    assert result.content == {"echoed": "hi"}
    assert result.call_id == "c1"
    assert result.duration_ms >= 0


def test_unknown_tool_rejected():
    reg = ToolRegistry()
    result = reg.execute(
        ToolRequest(name="nope", arguments={}, call_id="c1"), _ctx()
    )
    assert result.ok is False
    assert result.error.code == "tool_not_found"


def test_disabled_tool_rejected():
    disabled = ToolSpec(
        name="off",
        description="x",
        input_schema={"type": "object", "properties": {}},
        enabled=False,
    )
    reg = ToolRegistry()
    reg.register(disabled, lambda r, c: ToolResult(call_id=r.call_id, name="off", ok=True))
    result = reg.execute(ToolRequest(name="off", arguments={}, call_id="c1"), _ctx())
    assert result.error.code == "tool_disabled"
    # but list only returns enabled
    assert reg.list() == []
    assert reg.list(only_enabled=False) == [disabled]


def test_invalid_arguments_rejected():
    reg = ToolRegistry()
    reg.register(ECHO_SPEC, _echo_handler)
    # missing required
    r1 = reg.execute(ToolRequest(name="echo", arguments={}, call_id="c1"), _ctx())
    assert r1.error.code == "invalid_arguments"
    assert "msg" in r1.error.message
    # unexpected field
    r2 = reg.execute(
        ToolRequest(name="echo", arguments={"msg": "hi", "extra": "x"}, call_id="c1"),
        _ctx(),
    )
    assert r2.error.code == "invalid_arguments"
    assert "extra" in r2.error.message


def test_duplicate_register_raises():
    reg = ToolRegistry()
    reg.register(ECHO_SPEC, _echo_handler)
    with pytest.raises(ValueError):
        reg.register(ECHO_SPEC, _echo_handler)


def test_spec_requires_name():
    with pytest.raises(ValueError):
        ToolSpec(name="", description="x", input_schema={})
