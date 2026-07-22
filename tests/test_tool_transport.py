"""Tests for the tool transports and the sentinel parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from luna.models.base import ModelResponse, ToolCall, ToolSpec as ModelToolSpec, Usage
from luna.tools.transport import (
    Extraction,
    ExtractedCall,
    NativeTransport,
    PromptJsonTransport,
    parse_content,
    select_transport,
    tool_protocol_prompt,
)


# ---------------------------------------------------------------------------
# parse_content
# ---------------------------------------------------------------------------


def test_parse_valid_block():
    content = 'Here is my plan.\n```tool_call\n{"tool": "search_archive", "arguments": {"query": "hull"}}\n```\n'
    ext = parse_content(content, index=2)
    assert ext.malformed is None
    assert ext.has_block is True
    assert len(ext.calls) == 1
    c = ext.calls[0]
    assert c.name == "search_archive"
    assert c.arguments == {"query": "hull"}
    assert c.call_id == "prompt-2"


def test_parse_no_block_is_final():
    ext = parse_content("Just a normal answer in prose.", index=1)
    assert ext.calls == []
    assert ext.malformed is None
    assert ext.has_block is False


def test_parse_empty_content_is_final():
    ext = parse_content("", index=1)
    assert ext.calls == []
    assert ext.has_block is False


def test_parse_malformed_json():
    content = '```tool_call\n{"tool": "search_archive", "arguments": {bad}}\n```'
    ext = parse_content(content)
    assert ext.has_block is True
    assert ext.malformed is not None
    assert "invalid JSON" in ext.malformed
    assert ext.calls == []


def test_parse_missing_fields():
    content = '```tool_call\n{"name": "search_archive"}\n```'
    ext = parse_content(content)
    assert ext.malformed is not None
    assert "tool" in ext.malformed and "arguments" in ext.malformed


def test_parse_wrong_types():
    content = '```tool_call\n{"tool": 5, "arguments": []}\n```'
    ext = parse_content(content)
    assert ext.malformed is not None
    assert "must be" in ext.malformed


def test_parse_unclosed_block():
    content = '```tool_call\n{"tool": "search_archive", "arguments": {'
    ext = parse_content(content)
    assert ext.has_block is True
    assert ext.malformed is not None
    assert "not closed" in ext.malformed


def test_parse_nested_json_arguments():
    content = '```tool_call\n{"tool": "search_archive", "arguments": {"query": "hull", "filters": {"path": "a/b"}}}\n```'
    ext = parse_content(content)
    assert ext.malformed is None
    assert ext.calls[0].arguments["filters"] == {"path": "a/b"}


def test_parse_multiple_blocks_first_only():
    content = (
        '```tool_call\n{"tool": "search_archive", "arguments": {"query": "a"}}\n```\n'
        '```tool_call\n{"tool": "read_artifact", "arguments": {"artifact_id": "x"}}\n```'
    )
    ext = parse_content(content)
    assert len(ext.calls) == 1
    assert ext.calls[0].name == "search_archive"


def test_generic_json_block_not_a_tool_call():
    # A generic ```json block must NOT be treated as a tool call.
    content = '```json\n{"note": "this is just data"}\n```'
    ext = parse_content(content)
    assert ext.has_block is False
    assert ext.calls == []
    assert ext.malformed is None


def test_sentinel_inside_prose_still_parsed():
    content = (
        "Let me look that up for you.\n\n"
        '```tool_call\n{"tool": "search_archive", "arguments": {"query": "hull"}}\n```\n'
        "I will get back to you."
    )
    ext = parse_content(content)
    assert ext.calls[0].name == "search_archive"


# ---------------------------------------------------------------------------
# tool_protocol_prompt
# ---------------------------------------------------------------------------


def test_protocol_prompt_lists_tools_and_rules():
    specs = [
        ModelToolSpec(
            name="search_archive",
            description="Search the archive.",
            parameters={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string", "description": "the query"}},
            },
        )
    ]
    prompt = tool_protocol_prompt(specs)
    assert "search_archive" in prompt
    assert "```tool_call" in prompt
    assert "ONE" in prompt  # one tool per turn rule
    assert "query" in prompt
    assert "required" in prompt


def test_protocol_prompt_empty_specs():
    assert tool_protocol_prompt([])  # still produces something, no crash


# ---------------------------------------------------------------------------
# NativeTransport
# ---------------------------------------------------------------------------


def test_native_extract_maps_tool_calls():
    resp = ModelResponse(
        content="",
        tool_calls=(ToolCall(id="c1", name="search_archive", arguments={"query": "h"}),),
        usage=Usage(),
        model="m",
    )
    ext = NativeTransport().extract(resp)
    assert ext.calls[0].name == "search_archive"
    assert ext.calls[0].call_id == "c1"
    assert ext.has_block is True

    msg = NativeTransport().assistant_message(resp)
    assert msg.role == "assistant"
    assert msg.tool_calls == resp.tool_calls

    tr = NativeTransport().tool_result_message(ext.calls[0], '{"ok": true}')
    assert tr.role == "tool"
    assert tr.tool_call_id == "c1"


def test_native_no_calls_is_final():
    resp = ModelResponse(content="answer", usage=Usage(), model="m")
    ext = NativeTransport().extract(resp)
    assert ext.calls == []
    assert ext.malformed is None


def test_native_force_final_is_none():
    assert NativeTransport().force_final_message() is None


def test_native_wants_tools_on_wire():
    assert NativeTransport().wants_tools_on_wire is True


# ---------------------------------------------------------------------------
# PromptJsonTransport
# ---------------------------------------------------------------------------


def test_prompt_extract_parses_content():
    resp = ModelResponse(
        content='```tool_call\n{"tool":"search_archive","arguments":{"query":"h"}}\n```',
        usage=Usage(), model="m",
    )
    ext = PromptJsonTransport().extract(resp, call_index=3)
    assert ext.calls[0].name == "search_archive"
    assert ext.calls[0].call_id == "prompt-3"


def test_prompt_does_not_want_tools_on_wire():
    assert PromptJsonTransport().wants_tools_on_wire is False


def test_prompt_augment_system_prompt():
    specs = [ModelToolSpec(name="search_archive", description="Search.", parameters={})]
    sp = PromptJsonTransport().augment_system_prompt("You are Luna.", specs)
    assert "tool_call" in sp
    assert "You are Luna." in sp


def test_prompt_augment_no_specs_unchanged():
    assert PromptJsonTransport().augment_system_prompt("You are Luna.", []) == "You are Luna."


def test_prompt_tool_result_message_role_user():
    call = ExtractedCall(call_id="prompt-1", name="search_archive", arguments={})
    msg = PromptJsonTransport().tool_result_message(call, '{"ok": true, "content": {}}')
    assert msg.role == "user"
    assert "```tool_result" in msg.content
    assert "```tool_call" in msg.content  # instruction to continue


def test_prompt_repair_message():
    msg = PromptJsonTransport().repair_message("invalid JSON: ...")
    assert msg.role == "user"
    assert "tool_call" in msg.content
    assert "invalid JSON" in msg.content


def test_prompt_force_final_message():
    msg = PromptJsonTransport().force_final_message()
    assert msg is not None
    assert msg.role == "user"


# ---------------------------------------------------------------------------
# select_transport
# ---------------------------------------------------------------------------


class _NativeProvider:
    supports_native_tools = True


class _PromptProvider:
    supports_native_tools = False


def test_select_transport_by_flag():
    assert isinstance(select_transport(_NativeProvider()), NativeTransport)
    assert isinstance(select_transport(_PromptProvider()), PromptJsonTransport)


def test_select_transport_default_native():
    class Bare:
        pass
    assert isinstance(select_transport(Bare()), NativeTransport)
