"""Bounded result types for the web research tools.

These are the internal, provider-neutral records the fetch layer and the
search providers produce. The tool handlers (:mod:`luna.tools.web_tools`)
convert them into the model-facing dictionaries documented in
``docs/web-tools.md``. Provider-specific payloads never reach these
types — providers normalise into :class:`ProviderSearchResult` before
returning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderSearchResult:
    """One search hit, as a provider returns it (already normalised).

    ``rank`` is the provider's own ordering (1-indexed); the search tool
    re-assigns final ranks after de-duplicating canonical URLs.
    """

    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    rank: int = 0


@dataclass(frozen=True)
class SearchResultItem:
    """One model-facing search result."""

    result_id: str
    rank: int
    title: str
    url: str
    display_domain: str
    snippet: str
    published_at: str | None = None


@dataclass(frozen=True)
class FetchResult:
    """The bounded outcome of fetching one webpage.

    The model-facing ``fetch_webpage`` output is built from this. It
    never carries raw HTML — only extracted, bounded text.
    """

    requested_url: str
    final_url: str
    title: str
    content_type: str
    status_code: int
    retrieved_at: str
    text: str
    text_chars: int
    content_hash: str
    truncated: bool
    links: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FetchOutcome:
    """Either a successful :class:`FetchResult` or a structured failure.

    Kept separate from :class:`~luna.tools.protocol.ToolResult` so the
    fetch layer has no dependency on the tools framework — the handler
    translates. ``ok`` False means the fetch itself failed (network,
    unsupported content, invalid URL); ``available`` False is reserved
    for the search tool's "no provider configured" case and is not used
    here.
    """

    ok: bool
    result: FetchResult | None = None
    error_code: str = ""
    error_message: str = ""
    # Receipt summary fields (bounded, no body text):
    bytes_received: int = 0

    def to_content(self) -> dict[str, Any]:
        """Model-facing content dict for a successful fetch."""
        r = self.result
        assert r is not None
        return {
            "requested_url": r.requested_url,
            "final_url": r.final_url,
            "title": r.title,
            "content_type": r.content_type,
            "status_code": r.status_code,
            "retrieved_at": r.retrieved_at,
            "text": r.text,
            "text_chars": r.text_chars,
            "content_hash": r.content_hash,
            "truncated": r.truncated,
            "links": list(r.links),
        }
