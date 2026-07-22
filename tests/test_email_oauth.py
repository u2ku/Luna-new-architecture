"""Tests for luna.channels.email.oauth — XOAUTH2 token refresh + cache."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from luna.channels.email.oauth import (
    CachedTokenSource,
    OAuthConfig,
    OAuthError,
    TokenResponse,
    access_token_from_refresh,
    xoauth2_string,
)


class _FakeResp:
    """Minimal urllib.request.urlopen-compatible response context manager."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


# ── xoauth2_string ─────────────────────────────────────────────────────


def test_xoauth2_string_format() -> None:
    out = xoauth2_string("user@example.com", "tok-abc")
    assert out == "user=user@example.com\x01auth=Bearer tok-abc\x01\x01"


# ── access_token_from_refresh ──────────────────────────────────────────


def test_refresh_posts_correct_request() -> None:
    config = OAuthConfig(
        client_id="client",
        client_secret="secret",
        refresh_token="rtok",
    )
    captured: dict = {}

    def fake_urlopen(req, timeout=0):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["method"] = req.method
        captured["data"] = req.data.decode("utf-8")
        captured["headers"] = dict(req.headers)
        return _FakeResp({"access_token": "fresh-tok", "expires_in": 3600})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        resp = access_token_from_refresh(config, _now=1000.0)

    assert resp.access_token == "fresh-tok"
    assert resp.expires_at == 4600.0
    assert captured["url"] == "https://oauth2.googleapis.com/token"
    assert captured["method"] == "POST"
    assert "grant_type=refresh_token" in captured["data"]
    assert "client_id=client" in captured["data"]
    assert "client_secret=secret" in captured["data"]
    assert "refresh_token=rtok" in captured["data"]
    # urllib.request.Request capitalizes header keys via add_header.
    headers_lower = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers_lower["content-type"] == "application/x-www-form-urlencoded"


def test_refresh_raises_on_missing_config() -> None:
    with pytest.raises(OAuthError, match="not fully populated"):
        access_token_from_refresh(OAuthConfig("", "", ""))


def test_refresh_raises_on_error_response() -> None:
    def fake_urlopen(req, timeout=0):  # noqa: ARG001
        return _FakeResp({"error": "invalid_grant", "error_description": "Bad token"})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(OAuthError, match="invalid_grant"):
            access_token_from_refresh(
                OAuthConfig("c", "s", "rt"), _now=1000.0
            )


def test_refresh_raises_on_no_access_token() -> None:
    def fake_urlopen(req, timeout=0):  # noqa: ARG001
        return _FakeResp({"expires_in": 3600})  # no access_token field

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(OAuthError, match="no access_token"):
            access_token_from_refresh(
                OAuthConfig("c", "s", "rt"), _now=1000.0
            )


# ── CachedTokenSource ──────────────────────────────────────────────────


def test_cache_returns_fresh_token_without_calling_network() -> None:
    config = OAuthConfig("c", "s", "rt")
    src = CachedTokenSource(config)
    src._cached = TokenResponse(access_token="cached", expires_at=time.time() + 3600)
    with patch("urllib.request.urlopen") as mocked:
        assert src.get() == "cached"
        assert not mocked.called


def test_cache_refreshes_when_expired() -> None:
    config = OAuthConfig("c", "s", "rt")
    src = CachedTokenSource(config)
    src._cached = TokenResponse(access_token="old", expires_at=500.0)  # 1970
    fake = _FakeResp({"access_token": "fresh", "expires_in": 3600})
    with patch("urllib.request.urlopen", return_value=fake) as mocked:
        assert src.get() == "fresh"
        assert mocked.called


def test_cache_refreshes_within_margin() -> None:
    """Token expiring within REFRESH_MARGIN should be refreshed, not used."""
    config = OAuthConfig("c", "s", "rt")
    src = CachedTokenSource(config)
    src._cached = TokenResponse(
        access_token="soon", expires_at=time.time() + 5  # < REFRESH_MARGIN
    )
    fake = _FakeResp({"access_token": "fresh", "expires_in": 3600})
    with patch("urllib.request.urlopen", return_value=fake) as mocked:
        assert src.get() == "fresh"
        assert mocked.called


def test_cache_invalidate_clears() -> None:
    config = OAuthConfig("c", "s", "rt")
    src = CachedTokenSource(config)
    src._cached = TokenResponse(access_token="x", expires_at=time.time() + 3600)
    src.invalidate()
    assert src._cached is None


# ── OAuthConfig.is_configured ──────────────────────────────────────────


@pytest.mark.parametrize(
    "client_id,client_secret,refresh_token,expected",
    [
        ("a", "b", "c", True),
        ("", "b", "c", False),
        ("a", "", "c", False),
        ("a", "b", "", False),
        ("", "", "", False),
    ],
)
def test_is_configured(
    client_id: str, client_secret: str, refresh_token: str, expected: bool
) -> None:
    config = OAuthConfig(client_id, client_secret, refresh_token)
    assert config.is_configured() is expected


# ── from_env ───────────────────────────────────────────────────────────


def test_from_env_reads_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("LUNA_EMAIL_OAUTH_CLIENT_ID", "client-from-env")
    monkeypatch.setenv("LUNA_EMAIL_OAUTH_CLIENT_SECRET", "secret-from-env")
    monkeypatch.setenv("LUNA_EMAIL_OAUTH_REFRESH_TOKEN", "rtok-from-env")
    monkeypatch.setenv("LUNA_EMAIL_OAUTH_SCOPES", "https://example.com/scope")

    config = OAuthConfig.from_env()
    assert config.client_id == "client-from-env"
    assert config.client_secret == "secret-from-env"
    assert config.refresh_token == "rtok-from-env"
    assert config.scopes == "https://example.com/scope"


def test_from_env_defaults_to_read_and_compose(monkeypatch) -> None:
    for key in (
        "LUNA_EMAIL_OAUTH_CLIENT_ID",
        "LUNA_EMAIL_OAUTH_CLIENT_SECRET",
        "LUNA_EMAIL_OAUTH_REFRESH_TOKEN",
        "LUNA_EMAIL_OAUTH_SCOPES",
    ):
        monkeypatch.delenv(key, raising=False)

    config = OAuthConfig.from_env()
    # The default scope is the union the email channel needs:
    # inbox sync (gmail.readonly) + drafts/send (gmail.compose).
    assert "gmail.readonly" in config.scopes
    assert "gmail.compose" in config.scopes
    assert not config.is_configured()
