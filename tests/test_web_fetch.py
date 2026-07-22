"""Tests for the webpage fetch layer: redirects, byte/timeout limits,
extraction of HTML / plain text / JSON, charset handling, and the
unsupported-content path. Uses a fake HTTP transport and mocked DNS so
no real network is contacted.
"""

from __future__ import annotations

from typing import Any

import pytest

from luna.tools.config import WebFetchConfig
from luna.web.fetch import (
    FetchTimeout,
    HttpFetchResponse,
    HttpTransport,
    TransportError,
    fetch_webpage,
)


PUBLIC = lambda host: ["8.8.8.8"]


def _fetch_config(transport, max_bytes=2_000_000, max_redirects=5, max_text=50000,
                  default_text=20000, allowed_ports=(80, 443)) -> WebFetchConfig:
    return WebFetchConfig(
        connect_timeout_seconds=5,
        read_timeout_seconds=15,
        total_timeout_seconds=20,
        max_redirects=max_redirects,
        max_response_bytes=max_bytes,
        default_text_chars=default_text,
        max_text_chars=max_text,
        allowed_ports=allowed_ports,
        user_agent="LunaRuntime-test",
        max_links=50,
        transport=transport,
        resolver=PUBLIC,
    )


class FakeTransport(HttpTransport):
    """Routes URLs to canned (status, content_type, body) tuples.

    A body larger than ``max_bytes`` is truncated to ``max_bytes + 1`` to
    mimic the live transport's streaming cutoff. Records every call so a
    test can assert on the requested URL and byte ceiling.
    """

    def __init__(self, routes: dict[str, Any] | None = None, *, default=None,
                 raise_exc: Exception | None = None) -> None:
        self.routes = routes or {}
        self.default = default  # (status, ct, body)
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def get(self, url, *, headers, connect_timeout, read_timeout, max_bytes):
        self.calls.append(
            {"url": url, "max_bytes": max_bytes, "headers": dict(headers)}
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        entry = self.routes.get(url)
        if entry is None:
            entry = self.default or (200, "text/html", b"")
        status, ct, body = entry
        if isinstance(body, str):
            body = body.encode("utf-8")
        truncated = False
        if len(body) > max_bytes:
            body = body[: max_bytes + 1]
            truncated = True
        resp_headers = {"content-type": ct} if ct else {}
        return HttpFetchResponse(
            status_code=status,
            headers=resp_headers,
            body=body,
            request_url=url,
            body_truncated=truncated,
        )


HTML = (
    "<html><head><title>Example Page</title></head><body>"
    "<nav>home about</nav>"
    "<script>var x = 1;</script>"
    "<article><h1>Main Title</h1><p>First paragraph of content.</p>"
    "<p>Second paragraph.</p></article>"
    "<footer>footer noise</footer></body></html>"
)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_html_text_extraction():
    t = FakeTransport({"https://example.com/page": (200, "text/html", HTML)})
    outcome = fetch_webpage(
        "https://example.com/page", fetch_config=_fetch_config(t), transport=t
    )
    assert outcome.ok
    r = outcome.result
    assert r.title == "Example Page"
    assert "Main Title" in r.text
    assert "First paragraph of content." in r.text
    assert "Second paragraph." in r.text
    # nav / script / footer removed
    assert "var x" not in r.text
    assert "footer noise" not in r.text
    assert r.content_type == "text/html"
    assert r.status_code == 200
    assert len(r.content_hash) == 64  # sha256 hex
    assert r.truncated is False


def test_plain_text_extraction():
    t = FakeTransport(
        {"https://example.com/note": (200, "text/plain", "line one\n\nline two")}
    )
    outcome = fetch_webpage(
        "https://example.com/note", fetch_config=_fetch_config(t), transport=t
    )
    assert outcome.ok
    assert "line one" in outcome.result.text
    assert "line two" in outcome.result.text


def test_json_extraction():
    payload = '{"name": "luna", "nested": {"a": 1, "b": [1, 2]}}'
    t = FakeTransport(
        {"https://example.com/data.json": (200, "application/json", payload)}
    )
    outcome = fetch_webpage(
        "https://example.com/data.json", fetch_config=_fetch_config(t), transport=t
    )
    assert outcome.ok
    assert '"name": "luna"' in outcome.result.text


def test_character_truncation():
    body = "<p>" + ("word " * 5000) + "</p>"
    t = FakeTransport({"https://example.com/big": (200, "text/html", body)})
    outcome = fetch_webpage(
        "https://example.com/big",
        fetch_config=_fetch_config(t, default_text=100, max_text=100),
        transport=t,
        text_chars=100,
    )
    assert outcome.ok
    assert outcome.result.truncated is True
    assert outcome.result.text_chars == 100


def test_malformed_html_still_extracts():
    body = "<html><body><p>unclosed paragraph <b>bold"
    t = FakeTransport({"https://example.com/bad": (200, "text/html", body)})
    outcome = fetch_webpage(
        "https://example.com/bad", fetch_config=_fetch_config(t), transport=t
    )
    assert outcome.ok
    assert "unclosed paragraph" in outcome.result.text
    assert "bold" in outcome.result.text


def test_charset_handling():
    body = "<html><head><meta charset='iso-8859-1'></head><body><p>caf\xe9</p></body></html>"
    body_bytes = body.encode("iso-8859-1")
    t = FakeTransport(
        {"https://example.com/enc": (200, "text/html", body_bytes)}
    )
    outcome = fetch_webpage(
        "https://example.com/enc", fetch_config=_fetch_config(t), transport=t
    )
    assert outcome.ok
    assert "café" in outcome.result.text


def test_content_type_charset_header():
    body = "caf\xe9".encode("iso-8859-1")
    t = FakeTransport(
        {"https://example.com/enc2": (200, "text/html; charset=iso-8859-1", body)}
    )
    outcome = fetch_webpage(
        "https://example.com/enc2", fetch_config=_fetch_config(t), transport=t
    )
    assert outcome.ok
    assert "café" in outcome.result.text


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------


def test_redirect_followed_and_revalidated():
    t = FakeTransport({
        "https://example.com/old": (301, "text/html", b""),
        "https://example.com/new": (200, "text/html", "<html><body><p>final</p></body></html>"),
    })
    # Provide a Location by overriding the default route entries.
    # FakeTransport returns canned (status,ct,body); we need headers.
    # Patch by using a small subclass that returns Location on 301.
    class _Redirect(FakeTransport):
        def get(self, url, **kw):
            self.calls.append({"url": url, "max_bytes": kw["max_bytes"], "headers": dict(kw["headers"])})
            if url == "https://example.com/old":
                return HttpFetchResponse(
                    status_code=301,
                    headers={"content-type": "text/html", "location": "/new"},
                    body=b"",
                    request_url=url,
                )
            return super().get(url, **kw)

    t = _Redirect({"https://example.com/new": (200, "text/html", "<html><body><p>final</p></body></html>")})
    outcome = fetch_webpage(
        "https://example.com/old", fetch_config=_fetch_config(t), transport=t
    )
    assert outcome.ok
    assert outcome.result.final_url == "https://example.com/new"
    assert "final" in outcome.result.text


def test_redirect_into_private_address_rejected():
    class _RedirectToPrivate(FakeTransport):
        def get(self, url, **kw):
            if url == "https://example.com/redir":
                return HttpFetchResponse(
                    status_code=302,
                    headers={"location": "http://10.0.0.1/secret"},
                    body=b"",
                    request_url=url,
                )
            return super().get(url, **kw)

    t = _RedirectToPrivate(
        {"https://example.com/redir": (200, "text/html", b"")},
        default=(200, "text/html", b"should not reach"),
    )
    outcome = fetch_webpage(
        "https://example.com/redir", fetch_config=_fetch_config(t), transport=t
    )
    assert not outcome.ok
    assert outcome.error_code == "private_address"


def test_redirect_count_enforced():
    # 3 hops with max_redirects=2 → too_many_redirects on the 3rd.
    hops = {
        "https://example.com/r0": (302, "text/html", ""),
        "https://example.com/r1": (302, "text/html", ""),
        "https://example.com/r2": (302, "text/html", ""),
    }

    class _Chain(FakeTransport):
        def __init__(self):
            super().__init__()
            self._locs = {
                "https://example.com/r0": "https://example.com/r1",
                "https://example.com/r1": "https://example.com/r2",
                "https://example.com/r2": "https://example.com/r3",
            }

        def get(self, url, **kw):
            self.calls.append({"url": url, "max_bytes": kw["max_bytes"]})
            if url in self._locs:
                return HttpFetchResponse(
                    status_code=302,
                    headers={"location": self._locs[url]},
                    body=b"",
                    request_url=url,
                )
            return HttpFetchResponse(200, {"content-type": "text/html"}, b"end", url)

    t = _Chain()
    outcome = fetch_webpage(
        "https://example.com/r0",
        fetch_config=_fetch_config(t, max_redirects=2),
        transport=t,
    )
    assert not outcome.ok
    assert outcome.error_code == "too_many_redirects"


def test_redirect_loop_detected():
    class _Loop(FakeTransport):
        def get(self, url, **kw):
            if url == "https://example.com/a":
                return HttpFetchResponse(
                    302, {"location": "https://example.com/b"}, b"", url
                )
            return HttpFetchResponse(
                302, {"location": "https://example.com/a"}, b"", url
            )

    t = _Loop()
    outcome = fetch_webpage(
        "https://example.com/a", fetch_config=_fetch_config(t, max_redirects=5), transport=t
    )
    assert not outcome.ok
    assert outcome.error_code == "redirect_loop"


# ---------------------------------------------------------------------------
# Limits & failures
# ---------------------------------------------------------------------------


def test_response_byte_enforcement():
    big = b"x" * 1000
    t = FakeTransport({"https://example.com/big": (200, "text/plain", big)})
    outcome = fetch_webpage(
        "https://example.com/big",
        fetch_config=_fetch_config(t, max_bytes=100),
        transport=t,
    )
    assert outcome.ok
    # body was truncated to max_bytes + 1 by the transport
    assert outcome.bytes_received == 101
    assert t.calls[0]["max_bytes"] == 100


def test_timeout_handling():
    t = FakeTransport(default=(200, "text/html", b""))
    t.raise_exc = FetchTimeout("read")
    outcome = fetch_webpage(
        "https://example.com/slow", fetch_config=_fetch_config(t), transport=t
    )
    assert not outcome.ok
    assert outcome.error_code == "timeout"


def test_transport_error_handling():
    t = FakeTransport(default=(200, "text/html", b""))
    t.raise_exc = TransportError("connection refused", code="connection_failed")
    outcome = fetch_webpage(
        "https://example.com/x", fetch_config=_fetch_config(t), transport=t
    )
    assert not outcome.ok
    assert outcome.error_code == "connection_failed"


def test_unsupported_content_type():
    t = FakeTransport(
        {"https://example.com/doc.pdf": (200, "application/pdf", b"%PDF-1.4 ...")}
    )
    outcome = fetch_webpage(
        "https://example.com/doc.pdf", fetch_config=_fetch_config(t), transport=t
    )
    assert not outcome.ok
    assert outcome.error_code == "unsupported_content"


def test_unsupported_image_rejected():
    t = FakeTransport(
        {"https://example.com/img.png": (200, "image/png", b"\x89PNG\r\n")}
    )
    outcome = fetch_webpage(
        "https://example.com/img.png", fetch_config=_fetch_config(t), transport=t
    )
    assert not outcome.ok
    assert outcome.error_code == "unsupported_content"


def test_credential_url_rejected_before_request():
    t = FakeTransport(default=(200, "text/html", b"x"))
    outcome = fetch_webpage(
        "http://user:pass@example.com/", fetch_config=_fetch_config(t), transport=t
    )
    assert not outcome.ok
    assert outcome.error_code == "credentials_in_url"
    # the transport must not have been contacted
    assert t.calls == []


def test_localhost_rejected_before_request():
    t = FakeTransport(default=(200, "text/html", b"x"))
    outcome = fetch_webpage(
        "http://localhost/", fetch_config=_fetch_config(t), transport=t
    )
    assert not outcome.ok
    assert outcome.error_code == "blocked_hostname"
    assert t.calls == []


def test_fixed_user_agent_sent():
    t = FakeTransport({"https://example.com/p": (200, "text/html", b"<p>x</p>")})
    fetch_webpage(
        "https://example.com/p", fetch_config=_fetch_config(t), transport=t
    )
    assert t.calls[0]["headers"]["User-Agent"] == "LunaRuntime-test"


def test_include_links_returns_bounded_list():
    body = (
        '<html><body><article><p>text</p>'
        '<a href="/a">A</a><a href="/b">B</a>'
        '<a href="https://other.com/c">C</a><a href="javascript:bad">x</a>'
        "</article></body></html>"
    )
    t = FakeTransport({"https://example.com/p": (200, "text/html", body)})
    outcome = fetch_webpage(
        "https://example.com/p",
        fetch_config=_fetch_config(t),
        transport=t,
        include_links=True,
    )
    assert outcome.ok
    links = outcome.result.links
    assert "https://example.com/a" in links
    assert "https://other.com/c" in links
    assert not any(l.startswith("javascript") for l in links)


def test_links_not_returned_by_default():
    body = '<html><body><a href="/x">x</a></body></html>'
    t = FakeTransport({"https://example.com/p": (200, "text/html", body)})
    outcome = fetch_webpage(
        "https://example.com/p", fetch_config=_fetch_config(t), transport=t
    )
    assert outcome.ok
    assert outcome.result.links == ()
