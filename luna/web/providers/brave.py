"""Brave Search API provider.

Talks to the Brave Web Search endpoint
(``GET https://api.search.brave.com/res/v1/web/search``) authenticated
with an API key sent in the ``X-Subscription-Token`` header. As with
SearXNG, domain filters are applied as ``site:`` / ``-site:`` query
operators and only the provider-neutral fields are returned.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..types import ProviderSearchResult
from .base import ProviderError, ProviderNotConfigured, WebSearchProvider
from .searxng import GetJson, _augment_query, _default_get_json

#: Brave's published base. Kept as a default so a bare API key works
#: without extra configuration; an explicit base_url override still wins.
_DEFAULT_BRAVE_BASE = "https://api.search.brave.com"


def _freshness(recency_days: int | None) -> str | None:
    if recency_days is None:
        return None
    if recency_days <= 1:
        return "pd"
    if recency_days <= 7:
        return "pw"
    if recency_days <= 30:
        return "pm"
    return "py"


class BraveSearchProvider(WebSearchProvider):
    name = "brave"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BRAVE_BASE,
        timeout: float = 15.0,
        get_json: GetJson | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or _DEFAULT_BRAVE_BASE).strip().rstrip("/")
        self.timeout = float(timeout)
        self._get_json = get_json or _default_get_json

    def health(self) -> bool:
        return bool(self.api_key)

    def search(
        self,
        query: str,
        *,
        limit: int,
        domains: Sequence[str] | None,
        exclude_domains: Sequence[str] | None,
        recency_days: int | None,
    ) -> list[ProviderSearchResult]:
        if not self.api_key:
            raise ProviderNotConfigured("brave api key is not configured")
        params: dict[str, Any] = {
            "q": _augment_query(query, domains, exclude_domains),
            "count": max(1, min(int(limit), 20)),
        }
        freshness = _freshness(recency_days)
        if freshness:
            params["freshness"] = freshness
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        try:
            status, data = self._get_json(
                f"{self.base_url}/res/v1/web/search",
                params,
                headers,
                self.timeout,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"brave request failed: {exc}", code="request_failed")
        if status == 401 or status == 403:
            raise ProviderError("brave rejected the api key", code="auth_error")
        if status != 200 or not isinstance(data, dict):
            raise ProviderError(
                f"brave returned status {status}", code="http_error"
            )
        web = data.get("web") or {}
        raw = web.get("results") or []
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
                    snippet=(item.get("description") or "").strip(),
                    published_at=item.get("page_age") or item.get("published"),
                    rank=i + 1,
                )
            )
        return out
