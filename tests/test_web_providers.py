"""Tests for the SearXNG and Brave search providers.

Uses an injected ``get_json`` so no real HTTP is contacted. Asserts that
provider-specific payloads do not escape and that the auth/config
failure modes map to the right provider exceptions.
"""

from __future__ import annotations

import pytest

from luna.web.providers.base import ProviderError, ProviderNotConfigured
from luna.web.providers.brave import BraveSearchProvider
from luna.web.providers.searxng import SearxngSearchProvider


def _fake_get(payload, *, status=200):
    calls = []

    def fn(url, params, headers, timeout):
        calls.append({"url": url, "params": params, "headers": headers})
        return status, payload

    fn.calls = calls
    return fn


SEARXNG_PAYLOAD = {
    "results": [
        {"title": "First", "url": "https://a.example.com/1",
         "content": "snippet one", "publishedDate": "2026-01-01T00:00:00Z"},
        {"title": "Second", "url": "https://b.example.com/2", "content": "snippet two"},
        {"title": "NoUrl", "url": "", "content": "dropped"},
        {"not_a": "dict"},
    ]
}


def test_searxng_search_normalises_results():
    fn = _fake_get(SEARXNG_PAYLOAD)
    p = SearxngSearchProvider("https://searx.example.com", get_json=fn)
    results = p.search("drones", limit=5, domains=["caa.govt.nz"],
                       exclude_domains=None, recency_days=None)
    assert len(results) == 2  # no-url and non-dict entries dropped
    assert results[0].title == "First"
    assert results[0].url == "https://a.example.com/1"
    assert results[0].snippet == "snippet one"
    assert results[0].rank == 1
    # domain filter applied as a site: operator in the query
    assert "site:caa.govt.nz" in fn.calls[0]["params"]["q"]


def test_searxng_not_configured():
    p = SearxngSearchProvider("", get_json=_fake_get({}))
    with pytest.raises(ProviderNotConfigured):
        p.search("q", limit=5, domains=None, exclude_domains=None, recency_days=None)
    assert p.health() is False


def test_searxng_http_error():
    p = SearxngSearchProvider("https://searx.example.com", get_json=_fake_get({}, status=503))
    with pytest.raises(ProviderError):
        p.search("q", limit=5, domains=None, exclude_domains=None, recency_days=None)


def test_searxng_request_failure_mapped():
    def boom(url, params, headers, timeout):
        raise RuntimeError("network down")

    p = SearxngSearchProvider("https://searx.example.com", get_json=boom)
    with pytest.raises(ProviderError):
        p.search("q", limit=5, domains=None, exclude_domains=None, recency_days=None)


BRAVE_PAYLOAD = {
    "web": {
        "results": [
            {"title": "Brave One", "url": "https://a.example.com/b1",
             "description": "desc one", "page_age": "2026-01-02"},
            {"title": "Brave Two", "url": "https://b.example.com/b2", "description": "desc two"},
        ]
    }
}


def test_brave_search_normalises_results():
    fn = _fake_get(BRAVE_PAYLOAD)
    p = BraveSearchProvider("key-123", get_json=fn)
    results = p.search("drones", limit=5, domains=None, exclude_domains=["bad.com"],
                       recency_days=7)
    assert len(results) == 2
    assert results[0].title == "Brave One"
    assert results[0].published_at == "2026-01-02"
    assert results[0].rank == 1
    # the api key is sent in the subscription header
    assert fn.calls[0]["headers"]["X-Subscription-Token"] == "key-123"
    # exclude domain applied as -site:
    assert "-site:bad.com" in fn.calls[0]["params"]["q"]
    # freshness bucket for a week
    assert fn.calls[0]["params"]["freshness"] == "pw"


def test_brave_not_configured():
    p = BraveSearchProvider("", get_json=_fake_get({}))
    with pytest.raises(ProviderNotConfigured):
        p.search("q", limit=5, domains=None, exclude_domains=None, recency_days=None)
    assert p.health() is False


def test_brave_auth_error_mapped():
    p = BraveSearchProvider("bad-key", get_json=_fake_get({}, status=401))
    with pytest.raises(ProviderError):
        p.search("q", limit=5, domains=None, exclude_domains=None, recency_days=None)


def test_provider_specific_payload_does_not_escape():
    # The raw payload fields (favicon, engine, etc.) must not appear on
    # the returned ProviderSearchResult.
    fn = _fake_get({
        "results": [{
            "title": "T", "url": "https://a.example.com", "content": "s",
            "engine": "google", "favicon_url": "https://x/fav.ico",
            "positions": [1, 2], "category": "general",
        }]
    })
    p = SearxngSearchProvider("https://searx.example.com", get_json=fn)
    results = p.search("q", limit=5, domains=None, exclude_domains=None, recency_days=None)
    assert results[0].title == "T"
    assert not hasattr(results[0], "engine")
    assert not hasattr(results[0], "favicon_url")
