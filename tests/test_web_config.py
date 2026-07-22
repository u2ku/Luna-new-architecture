"""Tests for web config loading and provider/transport wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from luna.tools.config import (
    WebSearchConfig,
    build_search_provider,
    build_web_config,
    expand_env,
    load_web_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_expand_env_default_form(monkeypatch):
    monkeypatch.delenv("LUNA_WEB_SEARCH_PROVIDER", raising=False)
    assert expand_env("${LUNA_WEB_SEARCH_PROVIDER:-none}", {}) == "none"


def test_expand_env_default_overridden_by_env(monkeypatch):
    monkeypatch.setenv("LUNA_WEB_SEARCH_PROVIDER", "searxng")
    assert expand_env("${LUNA_WEB_SEARCH_PROVIDER:-none}", {}) == "searxng"


def test_load_web_config_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("LUNA_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("LUNA_SEARXNG_URL", raising=False)
    monkeypatch.delenv("LUNA_BRAVE_SEARCH_API_KEY", raising=False)
    search, fetch, limits = load_web_config(REPO_ROOT, data_root=tmp_path)
    assert search.provider_name == "none"
    assert search.default_limit == 5
    assert search.max_limit == 10
    assert search.timeout_seconds == 15
    assert fetch.connect_timeout_seconds == 5
    assert fetch.read_timeout_seconds == 15
    assert fetch.total_timeout_seconds == 20
    assert fetch.max_redirects == 5
    assert fetch.max_response_bytes == 2_000_000
    assert fetch.default_text_chars == 20_000
    assert fetch.max_text_chars == 50_000
    assert fetch.allowed_ports == (80, 443)
    assert fetch.user_agent == "LunaRuntime/0.1"
    assert limits.max_search_calls == 3
    assert limits.max_fetch_calls == 5
    assert limits.max_combined_web_calls == 8
    assert limits.max_combined_webpage_text == 50_000


def test_build_search_provider_none_when_unconfigured():
    cfg = WebSearchConfig(
        provider_name="none", default_limit=5, max_limit=10, timeout_seconds=15,
        searxng_url="", brave_api_key="",
    )
    assert build_search_provider(cfg) is None


def test_build_search_provider_searxng(monkeypatch):
    monkeypatch.setenv("LUNA_WEB_SEARCH_PROVIDER", "searxng")
    monkeypatch.setenv("LUNA_SEARXNG_URL", "https://searx.example.com")
    monkeypatch.delenv("LUNA_BRAVE_SEARCH_API_KEY", raising=False)
    cfg = load_web_config(REPO_ROOT)[0]
    provider = build_search_provider(cfg)
    assert provider is not None
    assert provider.name == "searxng"


def test_build_search_provider_searxng_without_url_is_none(monkeypatch):
    monkeypatch.setenv("LUNA_WEB_SEARCH_PROVIDER", "searxng")
    monkeypatch.delenv("LUNA_SEARXNG_URL", raising=False)
    cfg = load_web_config(REPO_ROOT)[0]
    assert build_search_provider(cfg) is None


def test_build_search_provider_brave(monkeypatch):
    monkeypatch.setenv("LUNA_WEB_SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("LUNA_BRAVE_SEARCH_API_KEY", "key-abc")
    monkeypatch.delenv("LUNA_SEARXNG_URL", raising=False)
    cfg = load_web_config(REPO_ROOT)[0]
    provider = build_search_provider(cfg)
    assert provider is not None
    assert provider.name == "brave"


def test_build_web_config_wires_transport_and_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("LUNA_WEB_SEARCH_PROVIDER", "searxng")
    monkeypatch.setenv("LUNA_SEARXNG_URL", "https://searx.example.com")
    web = build_web_config(REPO_ROOT, data_root=tmp_path)
    assert web.search.provider is not None
    assert web.fetch.transport is not None
    assert web.fetch.resolver is not None


def test_build_web_config_no_provider_starts_clean(monkeypatch, tmp_path):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("LUNA_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("LUNA_SEARXNG_URL", raising=False)
    web = build_web_config(REPO_ROOT, data_root=tmp_path)
    assert web.search.provider is None  # unavailable, not a crash
    assert web.fetch.transport is not None  # live transport still wired
