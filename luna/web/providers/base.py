"""Web search provider interface.

A provider translates one search query into an ordered list of
provider-neutral :class:`~luna.web.types.ProviderSearchResult`. It must:

* raise :class:`ProviderNotConfigured` from :meth:`search` when its own
  configuration (URL / API key) is missing — so the tool can surface
  ``available: False`` rather than crash;
* raise :class:`ProviderError` from :meth:`search` when a configured
  provider genuinely fails (network, auth, malformed response) — so the
  tool surfaces ``failed``;
* never leak native fields (``favicon``, ``engine``, raw JSON) into the
  returned records — only ``title``, ``url``, ``snippet``,
  ``published_at``, ``rank``.

:meth:`health` is a cheap reachability/config check used by the tool to
distinguish "no provider configured" from "provider down".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..types import ProviderSearchResult


class ProviderError(Exception):
    """A configured provider failed during a search (network/auth/parse)."""

    def __init__(self, message: str, *, code: str = "provider_failed") -> None:
        super().__init__(message)
        self.code = code


class ProviderNotConfigured(ProviderError):
    """The provider's required configuration is missing."""

    def __init__(self, message: str = "provider not configured") -> None:
        super().__init__(message, code="provider_not_configured")


class WebSearchProvider(ABC):
    """Interface every search provider implements."""

    #: Stable identifier written into receipts (e.g. ``"searxng"``).
    name: str = ""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        limit: int,
        domains: Sequence[str] | None,
        exclude_domains: Sequence[str] | None,
        recency_days: int | None,
    ) -> list[ProviderSearchResult]:
        """Run one search. See module docstring for failure semantics."""
        raise NotImplementedError

    @abstractmethod
    def health(self) -> bool:
        """True if the provider is configured and reachable enough to try."""
        raise NotImplementedError
