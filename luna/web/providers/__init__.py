"""Web search providers.

Each provider (:class:`~luna.web.providers.base.WebSearchProvider`)
normalises its native response into the provider-neutral
:class:`~luna.web.types.ProviderSearchResult` so provider-specific
payloads never escape into the model-facing result or the receipts.
"""

from __future__ import annotations

from .base import ProviderError, ProviderNotConfigured, WebSearchProvider
from .brave import BraveSearchProvider
from .searxng import SearxngSearchProvider

__all__ = [
    "BraveSearchProvider",
    "ProviderError",
    "ProviderNotConfigured",
    "SearxngSearchProvider",
    "WebSearchProvider",
]
