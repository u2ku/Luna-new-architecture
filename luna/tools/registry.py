"""Minimal tool registry.

Holds the set of declared tools (their :class:`ToolSpec` and handler)
and dispatches a :class:`ToolRequest` to the right handler, validating
the request against the tool's input schema first.

The registry is intentionally minimal: ``register``, ``get``, ``list``,
``execute``. No plugins, no sub-agents, no dynamic discovery — tools are
registered explicitly at startup (see :mod:`luna.tools.executor` and
:func:`luna.api.server.build_service`).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from .protocol import (
    ToolContext,
    ToolError,
    ToolHandler,
    ToolRequest,
    ToolResult,
    ToolSpec,
)


# Minimal JSON-Schema type check. The registry does not implement the
# full draft; it enforces ``required`` and the ``type`` of each declared
# property. Tools still validate defensively — this is the runtime's
# first gate, not the only one.
_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _validate_arguments(
    arguments: Mapping[str, Any], schema: Mapping[str, Any]
) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors: list[str] = []
    if not isinstance(arguments, Mapping):
        return ["arguments must be an object"]
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    for key in required:
        if key not in arguments:
            errors.append(f"missing required argument: {key!r}")
    for key, value in arguments.items():
        prop = properties.get(key)
        if prop is None:
            # Undeclared argument: reject so the model can't smuggle
            # fields the tool did not advertise (e.g. a caller-supplied
            # path into create_artifact).
            errors.append(f"unexpected argument: {key!r}")
            continue
        expected = prop.get("type")
        if not expected:
            continue
        py_type = _JSON_TYPE_MAP.get(expected)
        if py_type is None:
            continue
        # bool is a subclass of int; keep them distinct so an integer
        # field does not silently accept True/False.
        if expected == "integer" and isinstance(value, bool):
            errors.append(f"{key!r} must be an integer, not a boolean")
            continue
        if not isinstance(value, py_type):
            errors.append(
                f"{key!r} must be of type {expected}, got {type(value).__name__}"
            )
    return errors


class ToolRegistry:
    """Registers tools and dispatches validated requests."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        """Register a tool spec and its handler under ``spec.name``."""
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def get(self, name: str) -> ToolSpec | None:
        """Return the spec for ``name``, or ``None`` if not registered."""
        return self._specs.get(name)

    def list(self, *, only_enabled: bool = True) -> list[ToolSpec]:
        """Return specs, optionally filtered to enabled ones."""
        return [
            spec
            for spec in self._specs.values()
            if (spec.enabled or not only_enabled)
        ]

    def function_schemas(self) -> list[dict[str, Any]]:
        """Wire-format schemas for every enabled tool, for the model."""
        return [spec.to_function_schema() for spec in self.list(only_enabled=True)]

    def execute(
        self, request: ToolRequest, context: ToolContext
    ) -> ToolResult:
        """Validate and dispatch one request.

        Never raises on bad input or handler failure: a structured
        :class:`ToolError` is returned instead, so the turn loop and
        receipts always get a well-formed result to record. Genuine
        programming errors (a handler raising ``TypeError``) are wrapped
        as ``tool_error``.
        """
        start = time.perf_counter()
        spec = self._specs.get(request.name)
        if spec is None:
            return self._fail(
                request,
                ToolError(
                    code="tool_not_found",
                    message=f"no tool named {request.name!r}",
                ),
                start,
            )
        if not spec.enabled:
            return self._fail(
                request,
                ToolError(
                    code="tool_disabled",
                    message=f"tool {request.name!r} is disabled",
                ),
                start,
            )
        errors = _validate_arguments(request.arguments, spec.input_schema)
        if errors:
            return self._fail(
                request,
                ToolError(
                    code="invalid_arguments",
                    message="; ".join(errors),
                    details={"errors": errors},
                ),
                start,
            )
        try:
            result = self._handlers[request.name](request, context)
        except ToolResultException as exc:  # tool-emitted structured failure
            return self._fail(request, exc.error, start)
        except Exception as exc:  # pragma: no cover - defensive
            return self._fail(
                request,
                ToolError(
                    code="tool_error",
                    message=f"{type(exc).__name__}: {exc}",
                ),
                start,
            )
        duration = int((time.perf_counter() - start) * 1000)
        # Stamp provenance the handler should not have to know about.
        return ToolResult(
            call_id=request.call_id,
            name=request.name,
            ok=result.ok,
            content=result.content,
            artifact_ids=result.artifact_ids,
            error=result.error,
            duration_ms=duration if not result.duration_ms else result.duration_ms,
        )

    def _fail(
        self, request: ToolRequest, error: ToolError, start: float
    ) -> ToolResult:
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            call_id=request.call_id,
            name=request.name,
            ok=False,
            error=error,
            duration_ms=duration,
        )


class ToolResultException(Exception):
    """Raised by a handler to emit a structured failure cleanly.

    Handlers may also return a ``ToolResult(ok=False, ...)`` directly;
    this exception is the convenience for handlers that raise.
    """

    def __init__(self, error: ToolError) -> None:
        super().__init__(error.message)
        self.error = error
