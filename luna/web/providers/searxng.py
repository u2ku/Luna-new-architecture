"""SearXNG search provider.

Talks to a self-hosted SearXNG instance's JSON API
(``GET <base>/search?format=json``). SearXNG is the default-recommended
provider because it is self-hostable and needs no API key. Domain
include/exclude filters are applied as ``site:`` / ``-site:`` query
operators so the provider makes no assumptions about an instance's
installed engines.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..types import ProviderSearchResult
from .base import ProviderError, ProviderNotConfigured, WebSearchProvider

#: Signature: ``(url, params, headers, timeout) -> (status_code, parsed_json)``.
#: The default uses :mod:`requests`; tests inject a fake. Raises are
#: propagated so the provider can map them to :class:`ProviderError`.
GetJson = Callable[[str, dict, dict, float], "tuple[int, Any]"]


def _default_get_json(url: str, params: dict, headers: dict, timeout: float) -> tuple[int, Any]:
    import requests  # lazy: only live mode needs the network library

    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    return resp.status_code, resp.json()


def _augment_query(
    query: str,
    domains: Sequence[str] | None,
    exclude_domains: Sequence[str] | None,
) -> str:
    """Add ``site:`` / ``-site:`` operators to the query."""
    parts: list[str] = [query]
    for d in domains or ():
        parts.append(f"site:{d}")
    for d in exclude_domains or ():
        parts.append(f"-site:{d}")
    return " ".join(p for p in parts if p)


def _time_range(recency_days: int | None) -> str | None:
    if recency_days is None:
        return None
    if recency_days <= 1:
        return "day"
    if recency_days <= 7:
        return "week"
    if recency_days <= 30:
        return "month"
    return "year"


class SearxngSearchProvider(WebSearchProvider):
    name = "searxng"

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 15.0,
        get_json: GetJson | None = None,
    ) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        self.timeout = float(timeout)
        self._get_json = get_json or _default_get_json

    def health(self) -> bool:
        return bool(self.base_url)

    def search(
        self,
        query: str,
        *,
        limit: int,
        domains: Sequence[str] | None,
        exclude_domains: Sequence[str] | None,
        recency_days: int | None,
    ) -> list[ProviderSearchResult]:
        if not self.base_url:
            raise ProviderNotConfigured("searxng base url is not configured")
        params: dict[str, Any] = {
            "q": _augment_query(query, domains, exclude_domains),
            "format": "json",
            "safesearch": 1,
        }
        time_range = _time_range(recency_days)
        if time_range:
            params["time_range"] = time_range
        headers = {"Accept": "application/json"}
        try:
            status, data = self._get_json(
                f"{self.base_url}/search", params, headers, self.timeout
            )
        except ProviderError:
            raise
        except Exception as exc:  # network / timeout / JSON errors
            raise ProviderError(f"searxng request failed: {exc}", code="request_failed")
        if status != 200 or not isinstance(data, dict):
            raise ProviderError(
                f"searxng returned status {status}", code="http_error"
            )
        raw = data.get("results") or []
        out: list[ProviderSearchResult] = []
        for i, item in enumerate(raw[: max(1, limit)]):
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            if not url:
                continue
            out.append(
                ProviderSearchResult(
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=(item.get("content") or "").strip(),
                    published_at=item.get("publishedDate"),
                    rank=i + 1,
                )
            )
        return out
