"""Shared tool protocol for Luna's model-callable tools.

This module defines the minimal types every Luna tool — and the runtime
that dispatches it — agrees on. It is deliberately small and dependency-free:

* :class:`ToolSpec`     — a tool's declaration (name, description, input
  schema, read/write classification, enabled state).
* :class:`ToolRequest`  — one invocation: which tool, with what arguments,
  plus the provenance the receipts need.
* :class:`ToolResult`    — the bounded outcome: ok/failed, structured
  content, artifact identifiers, and an optional error.
* :class:`ToolError`    — a structured failure (code + message) so callers
  never have to parse prose to know what went wrong.
* :class:`ToolContext`  — runtime state a tool needs to execute safely
  (archive root, output root, limits, actor/source/stream/turn).

The registry (:mod:`luna.tools.registry`) dispatches
:class:`ToolRequest` → :class:`ToolResult` through registered handlers.
The executor (:mod:`luna.tools.executor`) wraps a dispatch in paired
``tool_call`` / ``tool_result`` ledger receipts.

Nothing here knows about a specific tool (search, read, write) or a
specific model provider. That separation is what lets the same tools be
exercised by the live turn loop, by tests, and by the smoke script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


# ---------------------------------------------------------------------------
# Access classification
# ---------------------------------------------------------------------------


class ToolAccess(str, Enum):
    """Read vs write classification.

    Read tools mutate nothing outside the ledger receipts; write tools
    persist artifacts. The turn loop uses this for budgeting and the
    receipts use it for audit.
    """

    READ = "read"
    WRITE = "write"


# ---------------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """Declaration of one model-callable tool.

    Attributes
    ----------
    name:
        Stable tool name the model calls (e.g. ``"search_archive"``).
    description:
        Human-readable summary shown to the model provider.
    input_schema:
        A JSON Schema object describing the tool's arguments. The
        registry validates requests against ``required`` and the
        ``type`` of each declared property before dispatch.
    access:
        Read or write. Defaults to read.
    enabled:
        Whether the tool is exposed to the model. A disabled tool is
        listed by the registry but rejected on execution.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    access: ToolAccess = ToolAccess.READ
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ToolSpec.name must not be empty")

    def to_function_schema(self) -> dict[str, Any]:
        """Render this spec as an OpenAI-style function-calling schema.

        The model layer (:mod:`luna.models.base`) ships its own
        ``ToolSpec`` that is exactly this wire shape; the turn loop
        converts through this method so the tools layer never imports
        provider-specific types.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.input_schema) if self.input_schema else {},
        }


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolError:
    """A structured tool failure.

    ``code`` is a stable machine identifier (e.g.
    ``"invalid_arguments"``, ``"path_traversal"``, ``"tool_disabled"``);
    ``message`` is a short human-readable explanation; ``details`` carries
    optional structured context (never full artifact contents).
    """

    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


# ---------------------------------------------------------------------------
# Request / result
# ---------------------------------------------------------------------------


#: Signature every tool handler implements.
ToolHandler = Callable[["ToolRequest", "ToolContext"], "ToolResult"]


@dataclass(frozen=True)
class ToolRequest:
    """One tool invocation requested by the model.

    ``call_id`` is the model's tool-call identifier (carried back on the
    matching :class:`ToolResult` and into the ledger receipts so a
    ``tool_result`` event can reference its ``tool_call``).
    """

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(frozen=True)
class ToolResult:
    """The bounded outcome of one tool execution.

    Either ``ok`` is True and ``content`` carries structured data (an
    object, list, or short string — never a full document), or ``ok`` is
    False and ``error`` explains the failure. ``artifact_ids`` lists the
    stable identifiers touched (search hits, the artifact read, the
    artifact created) so receipts can record them without recording
    content.
    """

    call_id: str
    name: str
    ok: bool
    content: Any = None
    artifact_ids: tuple[str, ...] = ()
    error: ToolError | None = None
    duration_ms: int = 0
    #: Optional bounded receipt summary (web tools set this so the
    #: ``tool_result`` event records generic execution facts — query,
    #: provider, result count, URLs, hashes — without persisting full
    #: snippets or page text. The receipts writer sanitises it.
    receipt: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "call_id": self.call_id,
            "name": self.name,
            "ok": self.ok,
            "artifact_ids": list(self.artifact_ids),
            "duration_ms": self.duration_ms,
        }
        if self.error is not None:
            out["error"] = self.error.to_dict()
        else:
            out["error"] = None
        out["content"] = self.content
        return out


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolContext:
    """Runtime state a tool needs to execute safely.

    Built by the turn loop per call. Tools must not reach beyond what
    this context grants them — e.g. ``search_archive`` only reads
    ``archive_root`` and ``read_artifact`` only resolves ids against it.
    """

    archive_root: Path | None
    artifact_output_root: Path
    search_default_limit: int
    search_max_limit: int
    read_default_lines: int
    read_max_lines: int
    actor: Mapping[str, Any]
    source: Mapping[str, Any]
    stream_id: str
    turn_id: str
    #: Web research tool configuration (None when web tools are not
    #: wired — handlers then surface ``available: False``).
    web_search: Any = None
    web_fetch: Any = None
