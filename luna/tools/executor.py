"""Tool dispatch with paired ledger receipts.

Wires the three archive tools into a :class:`ToolRegistry` and wraps a
dispatch in the ``tool_call`` / ``tool_result`` receipt pair the
architecture requires. The turn loop (:mod:`luna.api.routes`) calls
:func:`execute_with_receipts`; it never calls the registry directly so a
tool can never run unreceipted.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..archive import (
    CREATE_ARTIFACT_SPEC,
    READ_ARTIFACT_SPEC,
    SEARCH_ARCHIVE_SPEC,
    handle_create_artifact,
    handle_read_artifact,
    handle_search_archive,
)
from ..ledger import WorldLedger
from .protocol import ToolContext, ToolRequest, ToolResult
from .receipts import build_tool_call_payload, build_tool_result_payload
from .registry import ToolRegistry
from .web_tools import (
    FETCH_WEBPAGE_SPEC,
    SEARCH_WEB_SPEC,
    handle_fetch_webpage,
    handle_search_web,
)

#: Per-turn budget. Enforced by the turn loop, declared here so the
#: limit lives next to the tools it bounds.
MAX_TOOL_CALLS_PER_TURN = 6
MAX_RESULT_CHARS_PER_TURN = 20_000

#: Tool names that exercise the network and are subject to the separate
#: web per-turn ceilings (see :class:`~luna.tools.config.WebTurnLimits`).
WEB_TOOL_NAMES: frozenset[str] = frozenset({"search_web", "fetch_webpage"})


def build_archive_registry() -> ToolRegistry:
    """Register the three archive tools and return the registry."""
    registry = ToolRegistry()
    registry.register(SEARCH_ARCHIVE_SPEC, handle_search_archive)
    registry.register(READ_ARTIFACT_SPEC, handle_read_artifact)
    registry.register(CREATE_ARTIFACT_SPEC, handle_create_artifact)
    return registry


def register_web_tools(registry: ToolRegistry) -> None:
    """Register the two web research tools on an existing registry."""
    registry.register(SEARCH_WEB_SPEC, handle_search_web)
    registry.register(FETCH_WEBPAGE_SPEC, handle_fetch_webpage)


def build_registry() -> ToolRegistry:
    """Register the archive + web tools and return the registry.

    Web tools are always registered; their handlers surface
    ``available: False`` when no provider is configured, so registering
    them never blocks startup. The ``tools.enabled`` config list gates
    which are exposed to the model via ``list(only_enabled=True)``.
    """
    registry = build_archive_registry()
    register_web_tools(registry)
    return registry


def execute_with_receipts(
    registry: ToolRegistry,
    request: ToolRequest,
    context: ToolContext,
    ledger: WorldLedger,
    *,
    actor: Mapping[str, Any],
    source: Mapping[str, Any] | None = None,
) -> ToolResult:
    """Dispatch one request and write its paired receipts.

    Writes ``tool_call`` before dispatch and ``tool_result`` after,
    the latter referencing the call's ``event_id``. Returns the
    :class:`ToolResult` so the turn loop can feed a bounded form of it
    back to the model.
    """
    spec = registry.get(request.name)
    call_payload = build_tool_call_payload(request, spec)

    call_event = ledger.append(
        event_type="tool_call",
        actor=actor,
        payload=call_payload,
        source=source or {"platform": "luna-runtime"},
        destination={"platform": "luna-runtime"},
        stream_id=context.stream_id,
        turn_id=context.turn_id,
    )

    result = registry.execute(request, context)

    result_payload = build_tool_result_payload(result, call_event["event_id"])
    ledger.append(
        event_type="tool_result",
        actor=actor,
        payload=result_payload,
        source=source or {"platform": "luna-runtime"},
        destination={"platform": "luna-runtime"},
        stream_id=context.stream_id,
        turn_id=context.turn_id,
    )
    return result
