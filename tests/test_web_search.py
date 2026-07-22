"""Tests for the search_web tool: validation, provider unavailable/failed,
result normalisation, de-duplication, stable ids, and receipts. Uses a
fake provider so no real network is contacted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest  # noqa: F401  (kept for potential markers)

from luna.ledger import WorldLedger
from luna.tools.config import WebSearchConfig
from luna.tools.executor import build_registry, execute_with_receipts
from luna.tools.protocol import ToolContext, ToolRequest
from luna.web.providers.base import ProviderError, WebSearchProvider
from luna.web.types import ProviderSearchResult


def _search_cfg(provider=None, provider_name="fake") -> WebSearchConfig:
    return WebSearchConfig(
        provider_name=provider_name,
        default_limit=5,
        max_limit=10,
        timeout_seconds=15,
        searxng_url="",
        brave_api_key="",
        provider=provider,
    )


def _ctx(search_cfg, fetch_cfg=None) -> ToolContext:
    return ToolContext(
        archive_root=Path("/nonexistent"),
        artifact_output_root=Path("/tmp/x"),
        search_default_limit=8,
        search_max_limit=20,
        read_default_lines=200,
        read_max_lines=500,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
        stream_id="web::test:test",
        turn_id="turn-1",
        web_search=search_cfg,
        web_fetch=fetch_cfg,
    )


class FakeProvider(WebSearchProvider):
    name = "fake"

    def __init__(self, results, *, raise_exc=None) -> None:
        self._results = results
        self._raise = raise_exc
        self.calls = []

    def health(self) -> bool:
        return True

    def search(self, query, *, limit, domains, exclude_domains, recency_days):
        self.calls.append(
            {"query": query, "limit": limit, "domains": domains, "recency_days": recency_days}
        )
        if self._raise is not None:
            raise self._raise
        return list(self._results)[:limit]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_empty_search_query_rejected():
    reg = build_registry()
    result = reg.execute(
        ToolRequest(name="search_web", arguments={"query": "   "}, call_id="c1"),
        _ctx(_search_cfg(provider=FakeProvider([]))),
    )
    assert result.ok is False
    assert result.error.code == "empty_query"


def test_search_limit_enforcement_clamps_to_max():
    reg = build_registry()
    provider = FakeProvider(
        [ProviderSearchResult(title=f"t{i}", url=f"https://x{i}.example.com/p",
                              snippet="s") for i in range(20)]
    )
    result = reg.execute(
        ToolRequest(name="search_web", arguments={"query": "q", "limit": 999},
                    call_id="c1"),
        _ctx(_search_cfg(provider=provider)),
    )
    assert result.ok
    assert result.content["result_count"] == 10  # clamped to max_limit
    assert provider.calls[0]["limit"] == 10


def test_search_limit_defaults():
    reg = build_registry()
    provider = FakeProvider(
        [ProviderSearchResult(title="t", url="https://x.example.com/p", snippet="s")]
    )
    result = reg.execute(
        ToolRequest(name="search_web", arguments={"query": "q"}, call_id="c1"),
        _ctx(_search_cfg(provider=provider)),
    )
    assert result.ok
    assert provider.calls[0]["limit"] == 5  # default


def test_domain_filter_rejects_malformed():
    reg = build_registry()
    result = reg.execute(
        ToolRequest(name="search_web",
                    arguments={"query": "q", "domains": ["not a domain"]},
                    call_id="c1"),
        _ctx(_search_cfg(provider=FakeProvider([]))),
    )
    assert result.ok is False
    assert result.error.code == "malformed_domain"


def test_domain_filter_rejects_scheme_path_port():
    reg = build_registry()
    for bad in ("https://x.com", "x.com/path", "x.com:8080", "user@x.com"):
        result = reg.execute(
            ToolRequest(name="search_web", arguments={"query": "q", "domains": [bad]},
                        call_id="c1"),
            _ctx(_search_cfg(provider=FakeProvider([]))),
        )
        assert result.ok is False
        assert result.error.code == "malformed_domain"


def test_domain_filter_rejects_too_many():
    reg = build_registry()
    result = reg.execute(
        ToolRequest(name="search_web",
                    arguments={"query": "q", "domains": [f"d{i}.example.com" for i in range(21)]},
                    call_id="c1"),
        _ctx(_search_cfg(provider=FakeProvider([]))),
    )
    assert result.ok is False
    assert result.error.code == "too_many_domains"


def test_domain_filter_accepts_valid():
    reg = build_registry()
    provider = FakeProvider([])
    reg.execute(
        ToolRequest(name="search_web",
                    arguments={"query": "q", "domains": ["aviation.govt.nz", "caa.govt.nz"]},
                    call_id="c1"),
        _ctx(_search_cfg(provider=provider)),
    )
    assert provider.calls[0]["domains"] == ["aviation.govt.nz", "caa.govt.nz"]


# ---------------------------------------------------------------------------
# Provider unavailable / failed
# ---------------------------------------------------------------------------


def test_provider_unavailable_when_not_configured():
    reg = build_registry()
    result = reg.execute(
        ToolRequest(name="search_web", arguments={"query": "q"}, call_id="c1"),
        _ctx(_search_cfg(provider=None, provider_name="none")),
    )
    assert result.ok is True
    assert result.content["available"] is False
    assert result.content["reason"] == "no_provider"
    assert result.content["result_count"] == 0


def test_provider_failed_returns_error():
    reg = build_registry()
    provider = FakeProvider([], raise_exc=ProviderError("boom", code="http_error"))
    result = reg.execute(
        ToolRequest(name="search_web", arguments={"query": "q"}, call_id="c1"),
        _ctx(_search_cfg(provider=provider)),
    )
    assert result.ok is False
    assert result.error.code == "provider_failed"


def test_empty_result_set_is_success_with_zero():
    reg = build_registry()
    provider = FakeProvider([])
    result = reg.execute(
        ToolRequest(name="search_web", arguments={"query": "q"}, call_id="c1"),
        _ctx(_search_cfg(provider=provider)),
    )
    assert result.ok is True
    assert result.content["available"] is True
    assert result.content["result_count"] == 0


# ---------------------------------------------------------------------------
# Normalisation, de-duplication, stable ids
# ---------------------------------------------------------------------------


def test_result_normalisation_strips_tracking_params():
    reg = build_registry()
    provider = FakeProvider([
        ProviderSearchResult(
            title="T", url="https://example.com/p?utm_source=x&id=42&fbclid=abc",
            snippet="  " + "s" * 500 + "  ",
        )
    ])
    result = reg.execute(
        ToolRequest(name="search_web", arguments={"query": "q"}, call_id="c1"),
        _ctx(_search_cfg(provider=provider)),
    )
    item = result.content["results"][0]
    assert "utm_source" not in item["url"]
    assert "fbclid" not in item["url"]
    assert "id=42" in item["url"]
    assert item["display_domain"] == "example.com"
    assert len(item["snippet"]) <= 400  # bounded


def test_duplicate_url_removal():
    reg = build_registry()
    provider = FakeProvider([
        ProviderSearchResult(title="A", url="https://example.com/a?utm_source=x", snippet="s"),
        ProviderSearchResult(title="B", url="https://example.com/a", snippet="s2"),
        ProviderSearchResult(title="C", url="https://example.com/b", snippet="s3"),
    ])
    result = reg.execute(
        ToolRequest(name="search_web", arguments={"query": "q"}, call_id="c1"),
        _ctx(_search_cfg(provider=provider)),
    )
    urls = [r["url"] for r in result.content["results"]]
    assert len(urls) == 2
    assert "https://example.com/a" in urls
    assert "https://example.com/b" in urls


def test_stable_result_identifiers():
    reg = build_registry()
    provider = FakeProvider([
        ProviderSearchResult(title="A", url="https://example.com/a", snippet="s"),
    ])
    r1 = reg.execute(
        ToolRequest(name="search_web", arguments={"query": "q"}, call_id="c1"),
        _ctx(_search_cfg(provider=provider)),
    )
    provider2 = FakeProvider([
        ProviderSearchResult(title="A", url="https://example.com/a", snippet="s"),
    ])
    r2 = reg.execute(
        ToolRequest(name="search_web", arguments={"query": "q"}, call_id="c2"),
        _ctx(_search_cfg(provider=provider2)),
    )
    id1 = r1.content["results"][0]["result_id"]
    id2 = r2.content["results"][0]["result_id"]
    assert id1 == id2
    assert id1.startswith("web:")
    assert r1.artifact_ids == (id1,)


def test_provider_ranking_preserved():
    reg = build_registry()
    provider = FakeProvider([
        ProviderSearchResult(title="A", url="https://a.example.com", snippet="s"),
        ProviderSearchResult(title="B", url="https://b.example.com", snippet="s"),
        ProviderSearchResult(title="C", url="https://c.example.com", snippet="s"),
    ])
    result = reg.execute(
        ToolRequest(name="search_web", arguments={"query": "q", "limit": 2}, call_id="c1"),
        _ctx(_search_cfg(provider=provider)),
    )
    ranks = [r["rank"] for r in result.content["results"]]
    assert ranks == [1, 2]
    assert result.content["results"][0]["title"] == "A"


def test_provider_specific_payload_does_not_escape():
    reg = build_registry()
    provider = FakeProvider([
        ProviderSearchResult(title="A", url="https://example.com/a", snippet="s"),
    ])
    result = reg.execute(
        ToolRequest(name="search_web", arguments={"query": "q"}, call_id="c1"),
        _ctx(_search_cfg(provider=provider)),
    )
    blob = json.dumps(result.content)
    assert "favicon" not in blob
    assert "engine" not in blob


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------


def test_search_receipt_pairing_and_summary(tmp_path):
    ledger = WorldLedger(path=tmp_path / "world.jsonl", lock_path=tmp_path / "w.lock")
    provider = FakeProvider([
        ProviderSearchResult(title="A", url="https://example.com/a", snippet="s"),
    ])
    reg = build_registry()
    result = execute_with_receipts(
        reg,
        ToolRequest(name="search_web", arguments={"query": "q"}, call_id="c1"),
        _ctx(_search_cfg(provider=provider, provider_name="fake")),
        ledger,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
    )
    assert result.ok
    events = ledger.tail(10)
    call = next(e for e in events if e["type"] == "tool_call")
    res = next(e for e in events if e["type"] == "tool_result")
    assert res["payload"]["call_event_id"] == call["event_id"]
    assert res["payload"]["status"] == "ok"
    assert res["payload"]["tool"] == "search_web"
    assert res["payload"]["receipt"]["provider"] == "fake"
    assert res["payload"]["receipt"]["result_count"] == 1
    assert res["payload"]["receipt"]["status"] == "ok"
    assert res["payload"]["receipt"]["query"] == "q"


def test_search_receipt_sanitisation_no_snippets(tmp_path):
    ledger = WorldLedger(path=tmp_path / "world.jsonl", lock_path=tmp_path / "w.lock")
    provider = FakeProvider([
        ProviderSearchResult(title="A", url="https://example.com/a", snippet="TOPSECRET-DATA-XYZ"),
    ])
    reg = build_registry()
    execute_with_receipts(
        reg,
        ToolRequest(name="search_web", arguments={"query": "q"}, call_id="c1"),
        _ctx(_search_cfg(provider=provider)),
        ledger,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
    )
    events = ledger.tail(10)
    call = next(e for e in events if e["type"] == "tool_call")
    res = next(e for e in events if e["type"] == "tool_result")
    # No full snippet text in either receipt.
    assert "TOPSECRET-DATA-XYZ" not in json.dumps(call["payload"])
    assert "TOPSECRET-DATA-XYZ" not in json.dumps(res["payload"])


def test_fetch_receipt_sanitisation_no_page_text(tmp_path):
    from luna.tools.config import WebFetchConfig
    from luna.web.fetch import HttpFetchResponse, HttpTransport

    class _T(HttpTransport):
        def get(self, url, *, headers, connect_timeout, read_timeout, max_bytes):
            body = ("<html><body><article><p>" + "SECRET-BODY-123 " * 50
                    + "</p></article></body></html>")
            b = body.encode("utf-8")
            return HttpFetchResponse(200, {"content-type": "text/html"}, b, url)

    fetch_cfg = WebFetchConfig(
        connect_timeout_seconds=5, read_timeout_seconds=15, total_timeout_seconds=20,
        max_redirects=5, max_response_bytes=2_000_000, default_text_chars=200,
        max_text_chars=500, allowed_ports=(80, 443), user_agent="t", max_links=50,
        transport=_T(), resolver=lambda h: ["8.8.8.8"],
    )
    ledger = WorldLedger(path=tmp_path / "world.jsonl", lock_path=tmp_path / "w.lock")
    reg = build_registry()
    execute_with_receipts(
        reg,
        ToolRequest(name="fetch_webpage", arguments={"url": "https://example.com/p"},
                    call_id="c1"),
        _ctx(_search_cfg(provider=None), fetch_cfg=fetch_cfg),
        ledger,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
    )
    events = ledger.tail(10)
    res = next(e for e in events if e["type"] == "tool_result")
    # The bounded result payload fed to the model is bounded; the receipt
    # must never contain the full page text.
    assert "SECRET-BODY-123" not in json.dumps(res["payload"])
    assert "content_hash" in res["payload"]["receipt"]
    assert "text_length" in res["payload"]["receipt"]
