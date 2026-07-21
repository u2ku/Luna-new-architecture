"""OpenAI Chat Completions provider.

Translates between the shared :mod:`luna.models.base` interface and the
OpenAI ``/v1/chat/completions`` HTTP API. Uses only the standard library
so the runtime stays dependency-free; swap in the official ``openai``
package later if streaming or richer features are needed.

The internal :class:`_OpenAIChatClient` is also reused by
:mod:`luna.models.whooshd`, since whooshd speaks the same wire format.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .base import (
    FinishReason,
    Message,
    ModelAuthError,
    ModelError,
    ModelProtocolError,
    ModelProvider,
    ModelRateLimit,
    ModelRequest,
    ModelResponse,
    ModelTimeout,
    ModelUnavailable,
    ToolCall,
    ToolSpec,
    Usage,
)


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 60.0
_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,  # legacy
    "content_filter": FinishReason.CONTENT_FILTER,
}


def _log_request(
    log_path: Path | None,
    url: str,
    payload: Mapping[str, Any],
    *,
    status: int | None,
    body: Any,
) -> None:
    """Append one JSONL record of a chat-completions call to ``log_path``.

    Called by :meth:`_OpenAIChatClient.post` so the file is a faithful
    capture of what went on the wire, not a reconstruction from the
    ledger. Failures of the logger itself are swallowed — never let
    debug plumbing take down a real model call.
    """
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "url": url,
            "status": status,
            "request": payload,
            "response": body,
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # debug plumbing must not affect the model call


def _debug_log_path() -> Path | None:
    """Read $LUNA_DEBUG_PROMPT_LOG and return a Path, or None.

    Both providers call this from __post_init__; off by default.
    """
    raw = os.environ.get("LUNA_DEBUG_PROMPT_LOG")
    if not raw:
        return None
    return Path(raw)


@dataclass
class _OpenAIChatClient:
    """Stateless HTTP client for an OpenAI-compatible chat completions API.

    Configured per-provider with its own ``base_url``, auth scheme, and
    defaults. Knows nothing about whooshd vs OpenAI — it just speaks the
    shared wire format.
    """

    base_url: str
    auth_header: str  # fully-formed "Authorization: ..." value, or ""
    default_model: str | None
    timeout: float = DEFAULT_TIMEOUT
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    # When set, every request and response is appended to this JSONL
    # path BEFORE the request goes on the wire. Off by default; the
    # luna-server sets it via LUNA_DEBUG_PROMPT_LOG when debugging.
    debug_log_path: Path | None = None

    def post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.auth_header:
            headers["Authorization"] = self.auth_header
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        log_path = self.debug_log_path
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                status = resp.status
        except urllib.error.HTTPError as e:
            _log_request(log_path, url, payload, status=e.code, body=None)
            _raise_for_status(e)
        except urllib.error.URLError as e:
            _log_request(log_path, url, payload, status=None, body=None)
            raise ModelUnavailable(f"chat client: connection error: {e.reason}") from e
        except TimeoutError as e:
            _log_request(log_path, url, payload, status=None, body=None)
            raise ModelTimeout(f"chat client: timed out after {self.timeout}s") from e
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            _log_request(log_path, url, payload, status=status, body={"_raw": raw})
            raise ModelProtocolError(f"chat client: non-JSON response: {e}") from e
        _log_request(log_path, url, payload, status=status, body=body)
        return body

    def encode(self, request: ModelRequest) -> dict[str, Any]:
        model = request.model or self.default_model
        if not model:
            raise ModelError("no model specified (set default_model or request.model)")
        out: dict[str, Any] = {
            "model": model,
            "messages": [_encode_message(m) for m in request.messages],
        }
        if request.tools:
            out["tools"] = [_encode_tool(t) for t in request.tools]
        if request.temperature is not None:
            out["temperature"] = request.temperature
        if request.max_tokens is not None:
            out["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            out["top_p"] = request.top_p
        if request.stop:
            out["stop"] = list(request.stop)
        if request.response_format is not None:
            out["response_format"] = dict(request.response_format)
        return out

    def decode(self, body: Mapping[str, Any]) -> ModelResponse:
        try:
            choices = body["choices"]
            message = choices[0]["message"]
            finish_raw = choices[0].get("finish_reason") or "stop"
        except (KeyError, IndexError, TypeError) as e:
            raise ModelProtocolError(f"chat client: malformed response: {e}") from e

        finish = _FINISH_REASON_MAP.get(finish_raw, FinishReason.ERROR)
        content = message.get("content") or ""
        tool_calls = tuple(
            _decode_tool_call(tc) for tc in (message.get("tool_calls") or [])
        )
        usage = _decode_usage(body.get("usage") or {})

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=usage,
            model=str(body.get("model", "")),
            raw=dict(body),
        )


def _raise_for_status(err: urllib.error.HTTPError) -> None:
    status = err.code
    body = err.read().decode("utf-8", errors="replace")
    if status in (401, 403):
        raise ModelAuthError(f"chat client: auth failed ({status}): {body}") from err
    if status == 429:
        raise ModelRateLimit(f"chat client: rate limited (429): {body}") from err
    if status == 408:
        raise ModelTimeout(f"chat client: timeout (408): {body}") from err
    if 500 <= status < 600:
        raise ModelUnavailable(f"chat client: server error ({status}): {body}") from err
    raise ModelProtocolError(f"chat client: HTTP {status}: {body}") from err


# ---------------------------------------------------------------------------
# Public provider: OpenAI
# ---------------------------------------------------------------------------


@dataclass
class OpenAIProvider(ModelProvider):
    """Model provider for the OpenAI Chat Completions API.

    Parameters
    ----------
    api_key:
        Bearer token. Defaults to ``$OPENAI_API_KEY``; the provider will
        refuse to send a request without one.
    base_url:
        Override for the API root. Useful for OpenAI-compatible gateways.
    default_model:
        Model used when ``ModelRequest.model`` is ``None``.
    timeout:
        Per-request timeout in seconds.
    organization:
        Optional ``OpenAI-Organization`` header value.
    """

    name: str = "openai"
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    default_model: str | None = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    organization: str | None = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        self._client = _OpenAIChatClient(
            base_url=self.base_url,
            auth_header=f"Bearer {self.api_key}" if self.api_key else "",
            default_model=self.default_model,
            timeout=self.timeout,
            extra_headers=(
                {"OpenAI-Organization": self.organization}
                if self.organization
                else {}
            ),
            debug_log_path=_debug_log_path(),
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        if not self.api_key:
            raise ModelAuthError("OpenAIProvider: api_key not set")
        return self._client.decode(self._client.post(self._client.encode(request)))


# ---------------------------------------------------------------------------
# Pure helpers — exported for tests
# ---------------------------------------------------------------------------


def _encode_message(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.name is not None:
        out["name"] = m.name
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in m.tool_calls
        ]
    return out


def _encode_tool(t: ToolSpec) -> dict[str, Any]:
    fn: dict[str, Any] = {
        "name": t.name,
        "description": t.description,
        "parameters": dict(t.parameters) if t.parameters else {},
    }
    if t.strict:
        fn["strict"] = True
    return {"type": "function", "function": fn}


def _decode_tool_call(tc: Mapping[str, Any]) -> ToolCall:
    fn = tc.get("function") or {}
    raw_args = fn.get("arguments") or "{}"
    if isinstance(raw_args, dict):
        args: Any = raw_args
    else:
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
    return ToolCall(
        id=tc.get("id", ""),
        name=fn.get("name", ""),
        arguments=args if isinstance(args, Mapping) else {"_value": args},
    )


def _decode_usage(usage: Mapping[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        total_tokens=int(usage.get("total_tokens", 0) or 0),
    )
