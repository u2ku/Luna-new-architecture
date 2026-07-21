"""Shared model provider interface.

This module defines the plug shape every model provider must implement.
It contains no provider-specific code — only the contracts the rest of
the runtime relies on to swap providers without changes elsewhere.

The contract:

    request  = ModelRequest(messages, tools?, ...)
    response = provider.complete(request) -> ModelResponse
    response = ModelResponse(content, tool_calls?, finish_reason, usage, model)

Providers are responsible for translating between this shape and their
native API. They must never leak provider-specific fields back to callers;
anything provider-specific belongs on ``ModelResponse.raw``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Finish reasons
# ---------------------------------------------------------------------------


class FinishReason(str, Enum):
    """Why the model stopped generating.

    Providers map their native stop reasons onto these. Anything that does
    not cleanly fit collapses to ``ERROR`` with details in ``ModelResponse.raw``.
    """

    STOP = "stop"               # natural stop or hit a stop sequence
    LENGTH = "length"           # hit max_tokens
    TOOL_CALLS = "tool_calls"   # model emitted one or more tool calls
    CONTENT_FILTER = "content_filter"  # provider blocked the response
    ERROR = "error"             # unrecoverable provider error


# ---------------------------------------------------------------------------
# Request / response structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model is allowed to call.

    ``parameters`` is a JSON Schema object describing the tool's input.
    The schema is the same shape every provider sends upstream.
    """

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    strict: bool = False


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation emitted by the model.

    ``arguments`` is the parsed argument object (already JSON-decoded).
    Callers must not assume keys are present — validate against the
    tool's schema before acting on it.
    """

    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Message:
    """A single message in the conversation.

    Roles follow the OpenAI convention: ``system``, ``user``, ``assistant``,
    ``tool``. ``tool`` messages carry ``tool_call_id`` referencing the call
    they respond to. ``assistant`` messages may carry ``tool_calls``.
    """

    role: str
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelRequest:
    """A request to generate a model response.

    ``messages`` is the full conversation including any prior tool turns.
    Providers do not see prior context — the runtime is responsible for
    building a request that fits the model's context window.
    """

    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()
    response_format: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    """Token accounting for a single completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ModelResponse:
    """A single model completion.

    Either ``content`` is non-empty (text reply), or ``tool_calls`` is
    non-empty (model wants to invoke tools), or both — providers should
    never produce neither unless ``finish_reason`` is ``ERROR``.
    """

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ModelError(Exception):
    """Base class for provider failures."""


class ModelAuthError(ModelError):
    """Invalid or missing credentials."""


class ModelRateLimit(ModelError):
    """Provider throttled the request."""


class ModelTimeout(ModelError):
    """Request timed out before a response arrived."""


class ModelUnavailable(ModelError):
    """Provider is unreachable or returned a 5xx."""


class ModelProtocolError(ModelError):
    """Response shape did not match what the provider expected."""


# ---------------------------------------------------------------------------
# Provider base
# ---------------------------------------------------------------------------


class ModelProvider(ABC):
    """Abstract base for every model provider.

    Subclasses must define ``name`` and implement ``complete``. They must
    raise one of the ``ModelError`` subclasses on failure; the runtime
    will not catch generic exceptions from here.
    """

    #: Stable identifier used in config and logs (e.g. ``"openai"``).
    name: str = ""

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Generate a single completion for ``request``.

        Implementations must be deterministic with respect to their input
        (given the same model, seed, and provider state) so the runtime
        can replay the ledger and reach the same world state.
        """
        raise NotImplementedError
