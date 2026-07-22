"""Receipts for tool execution.

Every tool execution appends a pair of events to ``world.jsonl``:

* ``tool_call``  — written BEFORE dispatch, recording intent;
* ``tool_result`` — written AFTER dispatch, recording proof, and
  referencing its ``tool_call`` by id.

The architecture rule is that a tool call is intent and a result is
proof. To honour that, receipts never carry complete artifact contents:
arguments are bounded to declared fields with long strings truncated
and token-shaped values redacted, and the result payload records only
status, artifact ids, error code, and duration — never the content the
tool returned.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .protocol import ToolError, ToolRequest, ToolResult, ToolSpec

#: Max characters of any single string value kept in a receipt argument.
MAX_ARG_STR = 200

#: Max items kept from any list value in a receipt argument.
MAX_LIST = 10

# Token-shaped patterns redacted out of receipt arguments (and out of
# any artifact content summary). Mirrors the create_artifact rejecter
# but redacts instead of refusing, since a receipt must still be
# written even when content was borderline.
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|bearer)\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._\-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
)


def redact_secrets(text: str) -> str:
    """Replace token-shaped substrings with ``[REDACTED]``."""
    out = text
    for pattern in _TOKEN_PATTERNS:
        out = pattern.sub("[REDACTED]", out)
    return out


def _sanitize_value(value: Any) -> Any:
    """Truncate / redact a value for safe inclusion in a receipt."""
    if isinstance(value, str):
        redacted = redact_secrets(value)
        if len(redacted) > MAX_ARG_STR:
            return redacted[:MAX_ARG_STR] + "…"
        return redacted
    if isinstance(value, list):
        items = [_sanitize_value(v) for v in value[:MAX_LIST]]
        if len(value) > MAX_LIST:
            items.append(f"…(+{len(value) - MAX_LIST} more)")
        return items
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    return value


def bounded_arguments(
    arguments: Mapping[str, Any], spec: ToolSpec | None
) -> dict[str, Any]:
    """Return a receipt-safe copy of ``arguments``.

    Only fields the tool declared are kept (an undeclared field — e.g. a
    caller-supplied path — is dropped). Each value is truncated and
    redacted. The create_artifact ``content`` field, which is the full
    artifact body, is reduced to a short summary so the receipt never
    persists the document.
    """
    properties: Mapping[str, Any] = (
        (spec.input_schema.get("properties") if spec else None) or {}
    )
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        if properties and key not in properties:
            continue
        if key == "content":
            # Never persist a full artifact body in a receipt.
            text = value if isinstance(value, str) else str(value)
            out[key] = redact_secrets(text)[:MAX_ARG_STR] + (
                "…" if len(text) > MAX_ARG_STR else ""
            )
            out[f"{key}_length"] = len(text)
            continue
        out[key] = _sanitize_value(value)
    return out


def build_tool_call_payload(
    request: ToolRequest, spec: ToolSpec | None
) -> dict[str, Any]:
    """Payload for the ``tool_call`` event (intent)."""
    return {
        "tool": request.name,
        "call_id": request.call_id,
        "arguments": bounded_arguments(request.arguments, spec),
    }


def build_tool_result_payload(
    result: ToolResult, call_event_id: str
) -> dict[str, Any]:
    """Payload for the ``tool_result`` event (proof).

    Records the matching call id, status, artifact ids touched, error
    code on failure, and duration. Deliberately omits ``content``.
    """
    payload: dict[str, Any] = {
        "tool": result.name,
        "call_event_id": call_event_id,
        "call_id": result.call_id,
        "status": "ok" if result.ok else "error",
        "artifact_ids": list(result.artifact_ids),
        "duration_ms": result.duration_ms,
    }
    if result.error is not None:
        payload["error_code"] = result.error.code
        payload["error_message"] = result.error.message
    return payload
