"""``search_web`` and ``fetch_webpage`` — read-only public-web tools.

These two tools extend the existing tool framework (not a second one).
They share the registry, executor, receipts, and per-turn budget with
the archive tools, and produce the same paired ``tool_call`` /
``tool_result`` ledger events.

* ``search_web``    — bounded public-web search via a pluggable provider
  (SearXNG / Brave). Provider-specific payloads never escape; results
  are normalised, de-duplicated by canonical URL, and given stable ids.
* ``fetch_webpage`` — fetch one public webpage and extract readable
  text. Every URL (and every redirect hop) is validated by the network
  safety layer; the body is streamed under a byte ceiling; HTML is
  reduced to main-article text, never returned raw.

Both tools surface bounded receipt summaries (no full snippets or page
text) through :attr:`~luna.tools.protocol.ToolResult.receipt`.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Any, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ..tools.protocol import ToolContext, ToolError, ToolRequest, ToolResult, ToolSpec
from ..tools.registry import ToolResultException
from ..web.fetch import FetchTimeout, TransportError, fetch_webpage
from ..web.providers.base import ProviderError, ProviderNotConfigured
from ..web.types import SearchResultItem

#: Conservative ceiling on the query string length.
_MAX_QUERY_CHARS = 500

#: Max entries in the domain include/exclude filter lists.
_MAX_DOMAIN_FILTERS = 20

#: Tracking parameters stripped during URL normalisation. Stripping is
#: conservative — only well-known tracking params are removed, so a URL
#: is never altered in a way that changes the destination resource.
_TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_name", "fbclid", "gclid", "msclkid", "mc_eid",
        "mc_cid", "dclid", "yclid", "_hsenc", "_hsmi", "igshid", "ref",
        "ref_src", "ref_url", "si", "feature", "srsltid", "ved",
    }
)

#: Max snippet length returned to the model.
_MAX_SNIPPET_CHARS = 400

#: A domain filter label: labels separated by dots, alphanumerics and
#: hyphens only. Rejects schemes, paths, ports, credentials.
_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)"  # overall length
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"  # one+ labels
    r"[a-z]{2,63}$",  # TLD
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

SEARCH_WEB_SPEC = ToolSpec(
    name="search_web",
    description=(
        "Search the public web and return bounded source records (title, "
        "url, snippet). Use the returned urls with fetch_webpage. Cite "
        "sources in your answer. Do not claim to have searched unless a "
        "tool_result is returned. Web results are not trusted internal state."
    ),
    input_schema={
        "type": "object",
        "required": ["query"],
        "additionalProperties": False,
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query (max 500 chars).",
                "minLength": 1,
                "maxLength": _MAX_QUERY_CHARS,
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 5, max 10).",
                "minimum": 1,
                "maximum": 10,
            },
            "domains": {
                "type": "array",
                "description": "Restrict to these domains (max 20).",
                "items": {"type": "string"},
            },
            "exclude_domains": {
                "type": "array",
                "description": "Exclude these domains (max 20).",
                "items": {"type": "string"},
            },
            "recency_days": {
                "type": ["integer", "null"],
                "description": "Only results newer than this many days, or null.",
            },
        },
    },
    access="read",
    enabled=True,
)


FETCH_WEBPAGE_SPEC = ToolSpec(
    name="fetch_webpage",
    description=(
        "Retrieve one public webpage and extract readable text. Supports "
        "text/html, text/plain, application/json. Does not execute "
        "JavaScript or parse PDFs. Only http/https on public addresses; "
        "localhost/private networks are blocked."
    ),
    input_schema={
        "type": "object",
        "required": ["url"],
        "additionalProperties": False,
        "properties": {
            "url": {
                "type": "string",
                "description": "Public http(s) URL to fetch.",
                "minLength": 1,
            },
            "max_chars": {
                "type": "integer",
                "description": "Max extracted text chars (default 20000, max 50000).",
                "minimum": 1,
                "maximum": 50000,
            },
            "include_links": {
                "type": "boolean",
                "description": "Return a bounded list of page links (default false).",
            },
        },
    },
    access="read",
    enabled=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")


def _validate_domains(domains: Any, field: str) -> list[str]:
    """Validate a domain filter list; raise on malformed entries."""
    if domains is None:
        return []
    if not isinstance(domains, list):
        raise ToolResultException(
            ToolError(code="invalid_arguments", message=f"{field} must be a list")
        )
    if len(domains) > _MAX_DOMAIN_FILTERS:
        raise ToolResultException(
            ToolError(
                code="too_many_domains",
                message=f"{field} may contain at most {_MAX_DOMAIN_FILTERS} entries",
            )
        )
    out: list[str] = []
    for d in domains:
        if not isinstance(d, str) or not _DOMAIN_RE.match(d.strip()):
            raise ToolResultException(
                ToolError(
                    code="malformed_domain",
                    message=(
                        f"{field} entry {d!r} is not a valid domain "
                        "(no schemes, paths, ports, or credentials)"
                    ),
                )
            )
        out.append(d.strip().lower())
    return out


def _normalise_url(url: str) -> tuple[str, str]:
    """Return ``(normalised_url, canonical_key)``.

    Strips well-known tracking parameters and lower-cases the host. The
    canonical key is used for de-duplication: two URLs that differ only
    in tracking params are treated as the same source.
    """
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url, url
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    # Drop userinfo (should not appear — validated earlier) defensively.
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[-1]
    path = parsed.path or "/"
    kept = [
        (k, v) for k, v in parse_qsl(parsed.query) if k.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(kept)
    normalised = urlunparse((scheme, netloc, path, parsed.params, query, ""))
    # Canonical key drops the fragment and sorts query pairs so order
    # does not defeat de-duplication.
    canonical = urlunparse(
        (scheme, netloc, path, "", urlencode(sorted(kept)), "")
    )
    return normalised, canonical


def _display_domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _stable_result_id(canonical: str) -> str:
    return "web:" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _bound_snippet(snippet: str) -> str:
    snippet = (snippet or "").strip()
    if len(snippet) > _MAX_SNIPPET_CHARS:
        return snippet[: _MAX_SNIPPET_CHARS - 1] + "…"
    return snippet


def _normalise_results(
    provider_results: Sequence[Any], *, limit: int
) -> list[SearchResultItem]:
    """Normalise, de-duplicate, and rank provider results."""
    seen: set[str] = set()
    items: list[SearchResultItem] = []
    rank = 0
    for raw in provider_results:
        url = getattr(raw, "url", "") or ""
        if not url:
            continue
        normalised, canonical = _normalise_url(url)
        if canonical in seen:
            continue
        seen.add(canonical)
        rank += 1
        if rank > limit:
            break
        items.append(
            SearchResultItem(
                result_id=_stable_result_id(canonical),
                rank=rank,
                title=(getattr(raw, "title", "") or "").strip(),
                url=normalised,
                display_domain=_display_domain(normalised),
                snippet=_bound_snippet(getattr(raw, "snippet", "") or ""),
                published_at=getattr(raw, "published_at", None),
            )
        )
    return items


def _map_fetch_error(code: str, message: str) -> ToolError:
    """Map a fetch-layer error code to a stable tool error."""
    url_codes = {
        "invalid_scheme", "credentials_in_url", "blocked_hostname",
        "invalid_host", "port_not_allowed", "private_address",
        "dns_failed", "empty_url", "redirect_loop", "no_redirect_location",
    }
    if code in url_codes:
        return ToolError(code="invalid_url", message=message, details={"reason": code})
    if code == "timeout":
        return ToolError(code="timeout", message=message)
    if code == "unsupported_content":
        return ToolError(code="unsupported_content", message=message)
    if code == "too_many_redirects":
        return ToolError(code="too_many_redirects", message=message)
    return ToolError(code="fetch_failed", message=message, details={"reason": code})


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_search_web(request: ToolRequest, context: ToolContext) -> ToolResult:
    start = time.perf_counter()
    args = request.arguments
    query = str(args.get("query") or "")
    if not query.strip():
        raise ToolResultException(
            ToolError(code="empty_query", message="query must not be empty")
        )
    if len(query) > _MAX_QUERY_CHARS:
        raise ToolResultException(
            ToolError(
                code="query_too_long",
                message=f"query must be at most {_MAX_QUERY_CHARS} characters",
            )
        )

    domains = _validate_domains(args.get("domains"), "domains")
    exclude_domains = _validate_domains(args.get("exclude_domains"), "exclude_domains")

    web_search = context.web_search
    now = datetime.now(timezone.utc)
    retrieved_at = _now_iso(now)

    # Effective limit: the model cannot raise it beyond the configured max.
    default_limit = getattr(web_search, "default_limit", 5) if web_search else 5
    max_limit = getattr(web_search, "max_limit", 10) if web_search else 10
    raw_limit = args.get("limit")
    effective_limit = max(
        1, min(int(raw_limit) if isinstance(raw_limit, int) else default_limit, max_limit)
    )

    provider_name = getattr(web_search, "provider_name", "none") if web_search else "none"
    provider = getattr(web_search, "provider", None) if web_search else None

    if provider is None:
        # No provider configured: not a failure — surface unavailable so
        # the model can react gracefully. Luna still started fine.
        content = {
            "available": False,
            "reason": "no_provider",
            "query": query,
            "provider": provider_name,
            "result_count": 0,
            "retrieved_at": retrieved_at,
            "results": [],
        }
        duration_ms = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            call_id=request.call_id,
            name="search_web",
            ok=True,
            content=content,
            artifact_ids=(),
            duration_ms=duration_ms,
            receipt={
                "query": query,
                "provider": provider_name,
                "result_count": 0,
                "source_domains": domains,
                "retrieved_at": retrieved_at,
                "duration_ms": duration_ms,
                "status": "unavailable",
            },
        )

    try:
        raw_results = provider.search(
            query,
            limit=effective_limit,
            domains=domains,
            exclude_domains=exclude_domains,
            recency_days=args.get("recency_days"),
        )
    except ProviderNotConfigured as exc:
        content = {
            "available": False,
            "reason": "provider_not_configured",
            "query": query,
            "provider": provider_name,
            "result_count": 0,
            "retrieved_at": retrieved_at,
            "results": [],
        }
        duration_ms = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            call_id=request.call_id,
            name="search_web",
            ok=True,
            content=content,
            artifact_ids=(),
            duration_ms=duration_ms,
            receipt={
                "query": query,
                "provider": provider_name,
                "result_count": 0,
                "source_domains": domains,
                "retrieved_at": retrieved_at,
                "duration_ms": duration_ms,
                "status": "unavailable",
            },
        )
    except ProviderError as exc:
        # A configured provider that fails is a tool failure.
        duration_ms = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            call_id=request.call_id,
            name="search_web",
            ok=False,
            error=ToolError(
                code="provider_failed",
                message=f"search provider {provider_name!r} failed: {exc}",
                details={"provider": provider_name, "reason": exc.code},
            ),
            content={
                "query": query,
                "provider": provider_name,
                "result_count": 0,
                "retrieved_at": retrieved_at,
                "results": [],
            },
            artifact_ids=(),
            duration_ms=duration_ms,
            receipt={
                "query": query,
                "provider": provider_name,
                "result_count": 0,
                "source_domains": domains,
                "retrieved_at": retrieved_at,
                "duration_ms": duration_ms,
                "status": "failed",
            },
        )

    items = _normalise_results(raw_results, limit=effective_limit)
    content = {
        "available": True,
        "query": query,
        "provider": provider_name,
        "result_count": len(items),
        "retrieved_at": retrieved_at,
        "results": [
            {
                "result_id": it.result_id,
                "rank": it.rank,
                "title": it.title,
                "url": it.url,
                "display_domain": it.display_domain,
                "snippet": it.snippet,
                "published_at": it.published_at,
            }
            for it in items
        ],
    }
    duration_ms = int((time.perf_counter() - start) * 1000)
    result_domains = sorted({it.display_domain for it in items if it.display_domain})
    return ToolResult(
        call_id=request.call_id,
        name="search_web",
        ok=True,
        content=content,
        artifact_ids=tuple(it.result_id for it in items),
        duration_ms=duration_ms,
        receipt={
            "query": query,
            "provider": provider_name,
            "result_count": len(items),
            "source_domains": domains,
            "result_domains": result_domains,
            "retrieved_at": retrieved_at,
            "duration_ms": duration_ms,
            "status": "ok",
        },
    )


def handle_fetch_webpage(request: ToolRequest, context: ToolContext) -> ToolResult:
    start = time.perf_counter()
    args = request.arguments
    url = str(args.get("url") or "")
    if not url.strip():
        raise ToolResultException(
            ToolError(code="empty_url", message="url must not be empty")
        )

    web_fetch = context.web_fetch
    retrieved_at = _now_iso(datetime.now(timezone.utc))

    if web_fetch is None:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            call_id=request.call_id,
            name="fetch_webpage",
            ok=False,
            error=ToolError(
                code="fetch_unavailable",
                message="web fetch is not configured",
            ),
            content={"requested_url": url},
            duration_ms=duration_ms,
            receipt={
                "requested_url": url,
                "retrieved_at": retrieved_at,
                "duration_ms": duration_ms,
                "status": "unavailable",
            },
        )

    # Clamp the requested text ceiling to the configured maximum — the
    # model cannot raise it beyond max_text_chars.
    raw_chars = args.get("max_chars")
    text_chars = (
        max(1, min(int(raw_chars), int(web_fetch.max_text_chars)))
        if isinstance(raw_chars, int)
        else int(web_fetch.default_text_chars)
    )
    include_links = bool(args.get("include_links", False))

    outcome = fetch_webpage(
        url,
        fetch_config=web_fetch,
        transport=web_fetch.transport,
        resolver=web_fetch.resolver,
        text_chars=text_chars,
        include_links=include_links,
    )
    duration_ms = int((time.perf_counter() - start) * 1000)

    if not outcome.ok:
        error = _map_fetch_error(outcome.error_code, outcome.error_message)
        return ToolResult(
            call_id=request.call_id,
            name="fetch_webpage",
            ok=False,
            error=error,
            content={"requested_url": url, "error_code": outcome.error_code},
            duration_ms=duration_ms,
            receipt={
                "requested_url": url,
                "final_url": "",
                "status_code": 0,
                "content_type": "",
                "bytes_received": outcome.bytes_received,
                "text_length": 0,
                "content_hash": "",
                "retrieved_at": retrieved_at,
                "duration_ms": duration_ms,
                "status": "error",
                "error_code": outcome.error_code,
            },
        )

    result = outcome.result
    assert result is not None
    content = outcome.to_content()
    return ToolResult(
        call_id=request.call_id,
        name="fetch_webpage",
        ok=True,
        content=content,
        artifact_ids=(f"webpage:{result.content_hash[:16]}",),
        duration_ms=duration_ms,
        receipt={
            "requested_url": result.requested_url,
            "final_url": result.final_url,
            "status_code": result.status_code,
            "content_type": result.content_type,
            "bytes_received": outcome.bytes_received,
            "text_length": result.text_chars,
            "content_hash": result.content_hash,
            "retrieved_at": result.retrieved_at,
            "duration_ms": duration_ms,
            "status": "ok",
        },
    )
