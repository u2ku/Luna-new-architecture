"""Luna web research tools: bounded public-web search and page fetch.

This package adds two read-only network tools (``search_web`` and
``fetch_webpage``) to the existing tool framework. It introduces no
second tool framework: the specs and handlers live in
:mod:`luna.tools.web_tools` and dispatch through the shared registry
and executor with the same paired ``tool_call`` / ``tool_result``
receipts as the archive tools.

The package itself is organised so that provider-specific code
(:mod:`luna.web.providers`), network policy (:mod:`luna.web.security`),
and the HTTP transport + extraction (:mod:`luna.web.fetch`) stay out of
each other's way and out of the tool executor.
"""

from __future__ import annotations

from .fetch import (
    DEFAULT_MAX_LINKS,
    HttpFetchResponse,
    HttpTransport,
    RequestsHttpTransport,
    default_resolver,
    fetch_webpage,
)
from .security import (
    UrlValidationError,
    ValidatedUrl,
    classify_address,
    default_resolver as security_default_resolver,
    validate_url,
)
from .types import FetchOutcome, FetchResult, ProviderSearchResult, SearchResultItem

__all__ = [
    "DEFAULT_MAX_LINKS",
    "FetchOutcome",
    "FetchResult",
    "HttpFetchResponse",
    "HttpTransport",
    "ProviderSearchResult",
    "RequestsHttpTransport",
    "SearchResultItem",
    "UrlValidationError",
    "ValidatedUrl",
    "classify_address",
    "default_resolver",
    "fetch_webpage",
    "validate_url",
]
