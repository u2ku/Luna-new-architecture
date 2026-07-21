"""Whoosh'd local inference provider.

Talks to a running :mod:`whooshd` daemon over its OpenAI-compatible
``/v1/chat/completions`` endpoint. Whoosh'd is a local-first inference
broker for MLX, GGUF, VLM, and stub backends — see
``/Users/pieratradio/whooshd`` (or https://github.com/...) for the full
server surface and runtime validation guides.

This provider is a thin configuration wrapper around the shared
:class:`_OpenAIChatClient` from :mod:`luna.models.openai`, since whooshd
speaks the OpenAI chat completions wire format. The only differences
from OpenAI are:

* the base URL (``http://127.0.0.1:8000/v1`` by default);
* no required auth (whooshd is local-only — pass any non-empty string
  if your deployment enforces one);
* the default model is ``stub-model`` for testing without a real model.

Known limitations of whooshd itself (not of this adapter):

* Tool / function calling is **not implemented** upstream. ``tools`` are
  forwarded on the wire, but whooshd will not emit ``tool_calls`` in the
  response and will not consume them. The runtime must not depend on
  tool calling when the active provider is whooshd.
* Auth hardening is not implemented upstream. Do not expose whooshd
  outside ``127.0.0.1`` without a reverse proxy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .base import ModelProvider, ModelRequest, ModelResponse
from .openai import _OpenAIChatClient


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "stub-model"
DEFAULT_TIMEOUT = 120.0  # local models can be slow on first warmup
DEFAULT_API_KEY = "local"  # whooshd ignores this, but keep header well-formed


@dataclass
class WhooshdProvider(ModelProvider):
    """Model provider for a local Whoosh'd inference daemon.

    Parameters
    ----------
    base_url:
        Root of the whooshd HTTP API, e.g. ``http://127.0.0.1:8000/v1``.
        Override ``WHOOSH_BASE_URL`` to set globally.
    default_model:
        Model used when ``ModelRequest.model`` is ``None``. Default is
        ``stub-model`` (works against the daemon's stub adapter with no
        model download). Set to e.g. ``mlx-community/Llama-3.2-3B-Instruct-4bit``
        for a real MLX run.
    api_key:
        Sent as the ``Authorization: Bearer ...`` header. Whoosh'd
        ignores the value, but some HTTP clients reject a missing header.
    timeout:
        Per-request timeout in seconds. Local models can be slow on
        cold start, so the default is generous.
    """

    name: str = "whooshd"
    base_url: str = DEFAULT_BASE_URL
    default_model: str | None = DEFAULT_MODEL
    api_key: str | None = DEFAULT_API_KEY
    timeout: float = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        if self.base_url == DEFAULT_BASE_URL:
            env_url = os.environ.get("WHOOSH_BASE_URL")
            if env_url:
                self.base_url = env_url
        if self.api_key is None:
            env_key = os.environ.get("WHOOSH_API_KEY")
            self.api_key = env_key if env_key else DEFAULT_API_KEY
        self._client = _OpenAIChatClient(
            base_url=self.base_url,
            auth_header=f"Bearer {self.api_key}",
            default_model=self.default_model,
            timeout=self.timeout,
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        return self._client.decode(self._client.post(self._client.encode(request)))

    # -- Whooshd-specific helpers -----------------------------------------

    def health_url(self) -> str:
        """Return the URL to ``GET /health`` on the daemon.

        Useful for the runtime's gate to verify the daemon is up before
        scheduling a model call.
        """
        return self.base_url.rstrip("/").removesuffix("/v1") + "/health"

    def ready_url(self) -> str:
        """Return the URL to ``GET /ready`` on the daemon (200 or 503)."""
        return self.base_url.rstrip("/").removesuffix("/v1") + "/ready"
