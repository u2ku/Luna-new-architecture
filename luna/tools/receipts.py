"""Receipts for tool execution.

Every tool execution appends a pair of events to ``world.jsonl``:

* ``tool_call``  — written BEFORE dispatch, recording intent;
* ``tool_result`` — written AFTER dispatch, recording proof, and
  referencing its ``tool_call`` by id.

The architecture rule is that a tool call is intent and a result is
proof. To honour that, receipts never carry complete artifact contents:
arguments are bounded to declared fields with long strings truncated
and token-shaped values redacted. The result payload carries only the
envelope every tool_result needs — ``started_at`` / ``finished_at``,
``status``, ``duration_ms``, ``error_code``, ``result_summary``,
``affected_resources`` — plus a bounded per-tool digest (``query``,
``result_count``, ``top_results``, ``content_hash``, ``bytes_written``,
…) nested under ``receipt``. Never the content the tool returned.
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


def _default_result_summary(result: ToolResult) -> str:
    """A fallback one-line summary when a tool did not provide one."""
    if result.error is not None:
        return result.error.message
    if result.artifact_ids:
        return f"{result.name}: ok ({len(result.artifact_ids)} resource(s))"
    return f"{result.name}: {result.ok}"


def build_tool_result_payload(
    result: ToolResult,
    call_event_id: str,
    *,
    started_at: str = "",
    finished_at: str = "",
) -> dict[str, Any]:
    """Payload for the ``tool_result`` event (proof).

    The envelope carries the fields every tool_result needs regardless of
    tool: ``started_at`` / ``finished_at`` (when the dispatch ran),
    ``status``, ``duration_ms``, ``error_code`` (always present, ``None``
    on success), ``result_summary`` (a one-line human description), and
    ``affected_resources`` (the stable ids/paths the call touched — enough
    to locate the durable result).

    A tool may attach a bounded per-tool digest via :attr:`ToolResult.receipt`.
    ``result_summary`` and ``affected_resources`` are *promoted* from that
    digest to the envelope (and dropped from the nested block to avoid
    duplication); the rest is kept under ``receipt`` and sanitised. The
    full content the tool returned is never persisted — only enough to
    prove what happened and locate the durable result.
    """
    digest: dict[str, Any] = dict(result.receipt) if getattr(result, "receipt", None) else {}

    # Promote the two common fields the tool knows best; default otherwise.
    result_summary = digest.pop("result_summary", None)
    if not result_summary:
        result_summary = _default_result_summary(result)
    result_summary = redact_secrets(str(result_summary))[:MAX_ARG_STR]

    affected = digest.pop("affected_resources", None)
    if affected is None:
        affected = list(result.artifact_ids)

    payload: dict[str, Any] = {
        "tool": result.name,
        "call_event_id": call_event_id,
        "call_id": result.call_id,
        "status": "ok" if result.ok else "error",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": result.duration_ms,
        "error_code": result.error.code if result.error is not None else None,
        "error_message": result.error.message if result.error is not None else None,
        "result_summary": result_summary,
        "affected_resources": list(affected),
        "artifact_ids": list(result.artifact_ids),
    }
    # Any remaining per-tool digest (query, result_count, top_results,
    # content_hash, bytes_written, …) is sanitised and nested under
    # ``receipt``. Only ``result_summary`` and ``affected_resources`` are
    # promoted to the envelope (above); everything else the tool attaches
    # stays here. Tools must not put full content in the digest.
    if digest:
        payload["receipt"] = _sanitize_value(digest)
    return payload
