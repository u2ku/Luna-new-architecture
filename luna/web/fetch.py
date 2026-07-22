"""Webpage fetch + readable-text extraction.

The fetch layer is the bridge between the network safety layer
(:mod:`luna.web.security`) and the tool handler
(:mod:`luna.tools.web_tools`). It:

* follows redirects **manually**, re-validating every hop so a redirect
  into a private address is caught at the boundary;
* bounds the response body in a streaming read (stops at the configured
  byte ceiling);
* extracts readable text from HTML / plain text / JSON, never returning
  raw HTML to the model;
* computes a content hash and, optionally, a bounded link list.

The HTTP transport is an injected interface (:class:`HttpTransport`)
so tests use a :class:`FakeHttpTransport` and never contact the real
internet. The live transport (:class:`RequestsHttpTransport`) imports
:mod:`requests` lazily.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

from .security import (
    UrlValidationError,
    ValidatedUrl,
    default_resolver,
    validate_url,
)
from .types import FetchOutcome, FetchResult

#: Default cap on the number of links returned when ``include_links`` is set.
DEFAULT_MAX_LINKS = 50

#: Redirect status codes that trigger a manual hop.
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

#: Content types the tool will extract text from. Anything else yields an
#: ``unsupported_content`` outcome (PDFs, images, audio, video, archives,
#: binaries).
_SUPPORTED_CONTENT_TYPES: frozenset[str] = frozenset(
    {"text/html", "text/plain", "application/json"}
)


# ---------------------------------------------------------------------------
# Transport errors
# ---------------------------------------------------------------------------


class FetchTimeout(Exception):
    """A connect or read timeout exceeded the configured ceiling."""

    def __init__(self, phase: str = "read") -> None:
        super().__init__(f"{phase} timeout")
        self.phase = phase


class TransportError(Exception):
    """A network-level failure (connection refused, DNS, TLS, ...)."""

    def __init__(self, message: str, *, code: str = "fetch_failed") -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Transport interface + live implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HttpFetchResponse:
    """One single-hop HTTP response (no redirect following)."""

    status_code: int
    headers: dict[str, str]
    body: bytes
    request_url: str
    body_truncated: bool = False


class HttpTransport(ABC):
    """Single-hop HTTP GET with bounded streaming and explicit timeouts."""

    @abstractmethod
    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        connect_timeout: float,
        read_timeout: float,
        max_bytes: int,
    ) -> HttpFetchResponse:
        raise NotImplementedError


class RequestsHttpTransport(HttpTransport):
    """Live transport backed by :mod:`requests`.

    Redirects are disabled (``allow_redirects=False``) so the fetch loop
    can re-validate each ``Location``. The body is streamed in chunks and
    the read stops the moment the byte ceiling is exceeded. :mod:`requests`
    is imported lazily so the module loads without it (tests never need
    it).
    """

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        connect_timeout: float,
        read_timeout: float,
        max_bytes: int,
    ) -> HttpFetchResponse:
        import requests  # lazy: only live mode needs the network library
        from requests.exceptions import (
            ConnectionError as RConnectionError,
            Timeout as RTimeout,
            RequestException,
        )

        # requests' timeout is (connect, read). The configured total is
        # enforced as a ceiling on the read phase when it is the tighter
        # bound — a true absolute deadline would need to interrupt a
        # blocking socket read, which requests does not support.
        effective_read = read_timeout
        try:
            resp = requests.get(
                url,
                headers=headers,
                allow_redirects=False,
                stream=True,
                timeout=(connect_timeout, effective_read),
            )
        except RTimeout as exc:
            raise FetchTimeout("read" if "read" in str(exc).lower() else "connect")
        except RConnectionError as exc:
            raise TransportError(f"connection failed: {exc}", code="connection_failed")
        except RequestException as exc:
            raise TransportError(f"request failed: {exc}")

        body = bytearray()
        truncated = False
        try:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                if isinstance(chunk, str):  # pragma: no cover - defensive
                    chunk = chunk.encode("utf-8")
                body.extend(chunk)
                if len(body) > max_bytes:
                    # Keep one byte past the ceiling so ``truncated`` is
                    # truthy and the reported byte count is honest.
                    del body[max_bytes + 1 :]
                    truncated = True
                    break
        except RTimeout:
            raise FetchTimeout("read")
        except RequestException as exc:
            raise TransportError(f"stream failed: {exc}", code="stream_failed")

        headers_out = {str(k).lower(): str(v) for k, v in resp.headers.items()}
        return HttpFetchResponse(
            status_code=int(resp.status_code),
            headers=headers_out,
            body=bytes(body),
            request_url=url,
            body_truncated=truncated,
        )


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

#: Tags whose entire subtree is dropped (never decoded into text).
_SKIP_TAGS: frozenset[str] = frozenset(
    {"script", "style", "noscript", "svg", "canvas", "template", "form", "select"}
)

#: Tags treated as navigation / advertising clutter — subtree dropped.
_CLUTTER_TAGS: frozenset[str] = frozenset({"nav", "aside", "footer"})

#: Tags that introduce a paragraph / line boundary in the extracted text.
_BLOCK_TAGS: frozenset[str] = frozenset(
    {
        "p", "div", "section", "article", "li", "ul", "ol", "br",
        "tr", "table", "header", "pre", "blockquote", "figure",
        "figcaption", "h1", "h2", "h3", "h4", "h5", "h6", "dl", "dt",
        "dd", "main",
    }
)

#: class/id substrings that mark advertising / nav / chrome containers.
_CLUTTER_CLASS_RE = re.compile(
    r"(?:^|[\s_-])(?:nav|menu|sidebar|advert|ads?|promo|banner|cookie"
    r"|consent|popup|modal|social|share|footer|header|breadcrumb)(?:[\s_-]|$)",
    re.IGNORECASE,
)


class _ReadableExtractor(HTMLParser):
    """Extract title, main text, and links from an HTML document.

    Strategy: drop ``script``/``style``/``noscript``/``svg``/``canvas``
    and identifiable navigation/ad containers entirely; keep block
    boundaries as newlines; collapse repeated whitespace when finished.
    Falls back to whatever text survives (i.e. cleaned body text) when
    no ``<article>``/``<main>`` is identifiable.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._pieces: list[str] = []
        self._links: list[str] = []
        self._title_parts: list[str] = []
        self._skip_depth = 0  # script/style/clutter subtree depth
        self._in_title = 0
        self._pending_href: str | None = None

    # -- tag boundaries ------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if self._is_clutter(t, attrs):
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if t == "title":
            self._in_title += 1
        if t == "a":
            href = dict(attrs).get("href")
            if href:
                self._pending_href = href
        if t in _BLOCK_TAGS and self._pieces and not self._pieces[-1].endswith("\n"):
            self._pieces.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Self-closing tags (e.g. <br/>, <img/>). <br/> forces a break.
        t = tag.lower()
        if self._skip_depth > 0:
            return
        if t == "br" and self._pieces and not self._pieces[-1].endswith("\n"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if self._skip_depth > 0:
            # Only decrement when this endtag closes the clutter that
            # opened the skip — HTMLParser gives no nesting certainty for
            # malformed HTML, so we track depth conservatively: any
            # endtag of a clutter/skip tag peels one layer.
            if t in _SKIP_TAGS or t in _CLUTTER_TAGS:
                self._skip_depth = max(0, self._skip_depth - 1)
            return
        if t == "title":
            self._in_title = max(0, self._in_title - 1)
        if t == "a":
            if self._pending_href:
                self._links.append(self._pending_href)
                self._pending_href = None
        if t in _BLOCK_TAGS and self._pieces and not self._pieces[-1].endswith("\n"):
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        if data.strip():
            self._pieces.append(data)

    # -- helpers / output ---------------------------------------------

    @staticmethod
    def _is_clutter(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in _SKIP_TAGS or tag in _CLUTTER_TAGS:
            return True
        for name, value in attrs:
            if name in ("class", "id", "role") and value:
                if name == "role" and value.lower() in {
                    "navigation", "complementary", "banner", "search",
                }:
                    return True
                if name in ("class", "id") and _CLUTTER_CLASS_RE.search(value):
                    return True
        return False

    def title(self) -> str:
        return " ".join(p.strip() for p in self._title_parts if p.strip()).strip()

    def text(self) -> str:
        raw = "".join(self._pieces)
        # Collapse runs of spaces/tabs but keep paragraph breaks.
        raw = re.sub(r"[ \t\f\v]+", " ", raw)
        raw = re.sub(r" *\n[ \n]*", "\n", raw)
        return raw.strip()

    def links(self) -> list[str]:
        return list(self._links)


def _decode_body(body: bytes, content_type: str) -> tuple[str, str]:
    """Decode ``body`` to text, returning ``(text, charset_used)``.

    Charset priority: the ``Content-Type`` header's ``charset=``;
    an HTML ``<meta charset>``/``<meta http-equiv>``; then UTF-8 with
    replacement (never raises on bad bytes).
    """
    ct = content_type.lower()
    charset = ""
    if "charset=" in ct:
        charset = ct.split("charset=", 1)[1].split(";")[0].strip().strip('"')
    if not charset and "html" in ct:
        # Sniff a meta charset from the first few KB.
        head = body[:4096].decode("ascii", errors="ignore").lower()
        m = re.search(r'charset=["\']?([\w\-]+)', head)
        if m:
            charset = m.group(1)
    if not charset:
        charset = "utf-8"
    try:
        return body.decode(charset, errors="replace"), charset
    except (LookupError, TypeError):
        return body.decode("utf-8", errors="replace"), "utf-8"


def _content_type(raw: str) -> str:
    """Normalise a Content-Type header to a bare type (no params)."""
    return (raw or "").split(";")[0].strip().lower()


def _extract_links(html: str, base: str, max_links: int) -> tuple[str, ...]:
    """Return up to ``max_links`` absolute URLs from ``<a href>`` tags."""
    extractor = _ReadableExtractor()
    try:
        extractor.feed(html)
    except Exception:
        pass
    out: list[str] = []
    seen: set[str] = set()
    for href in extractor.links():
        absolute = urljoin(base, href)
        if absolute in seen:
            continue
        scheme = urlparse(absolute).scheme.lower()
        if scheme not in ("http", "https"):
            continue
        seen.add(absolute)
        out.append(absolute)
        if len(out) >= max_links:
            break
    return tuple(out)


def _html_title_and_text(html: str) -> tuple[str, str]:
    extractor = _ReadableExtractor()
    try:
        extractor.feed(html)
    except Exception:
        # Even malformed input has already produced partial data;
        # return whatever was collected.
        pass
    try:
        extractor.close()
    except Exception:
        pass
    return extractor.title(), extractor.text()


def _json_to_text(body_text: str) -> str:
    """Pretty-print JSON into bounded readable text."""
    try:
        parsed = json.loads(body_text)
    except (json.JSONDecodeError, ValueError):
        return body_text
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def _collapse_plain(text: str) -> str:
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]*", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------


def _now_iso(now: datetime | None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")


def fetch_webpage(
    url: str,
    *,
    fetch_config: Any,
    transport: HttpTransport | None = None,
    resolver=default_resolver,
    now: datetime | None = None,
    text_chars: int | None = None,
    include_links: bool = False,
) -> FetchOutcome:
    """Fetch one public webpage and extract readable text.

    ``fetch_config`` is a duck-typed object (the real one is
    :class:`~luna.tools.config.WebFetchConfig`) supplying the limits:
    ``connect_timeout_seconds``, ``read_timeout_seconds``,
    ``total_timeout_seconds``, ``max_redirects``, ``max_response_bytes``,
    ``max_text_chars``, ``default_text_chars``, ``allowed_ports``,
    ``user_agent``, and ``max_links``.

    Never raises on expected failures: it returns a
    :class:`FetchOutcome` with ``ok=False`` and a stable ``error_code``.
    """
    connect = float(fetch_config.connect_timeout_seconds)
    read = float(fetch_config.read_timeout_seconds)
    total = float(fetch_config.total_timeout_seconds)
    max_redirects = int(fetch_config.max_redirects)
    max_bytes = int(fetch_config.max_response_bytes)
    max_text = int(fetch_config.max_text_chars)
    default_text = int(fetch_config.default_text_chars)
    allowed_ports = list(fetch_config.allowed_ports)
    user_agent = str(fetch_config.user_agent)
    max_links = int(getattr(fetch_config, "max_links", DEFAULT_MAX_LINKS))

    # Effective text ceiling: the model cannot raise this beyond the
    # configured maximum. ``text_chars`` defaults to the default cap.
    if text_chars is None:
        effective_text = default_text
    else:
        effective_text = max(1, min(int(text_chars), max_text))

    if transport is None:
        transport = fetch_config.transport  # live transport

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/json,text/plain;q=0.5,*/*;q=0.1",
        "Accept-Encoding": "identity",  # no gzip so byte/charset is honest
    }

    current = url
    redirects = 0
    last_response: HttpFetchResponse | None = None
    visited: set[str] = set()

    while True:
        try:
            validated: ValidatedUrl = validate_url(
                current, allowed_ports=allowed_ports, resolver=resolver
            )
        except UrlValidationError as exc:
            return FetchOutcome(
                ok=False,
                error_code=exc.code,
                error_message=exc.message,
                bytes_received=0,
            )

        if validated.url in visited:
            return FetchOutcome(
                ok=False,
                error_code="redirect_loop",
                error_message="a redirect loop was detected",
                bytes_received=0,
            )
        visited.add(validated.url)

        try:
            response = transport.get(
                validated.url,
                headers=headers,
                connect_timeout=min(connect, total),
                read_timeout=min(read, total),
                max_bytes=max_bytes,
            )
        except FetchTimeout as exc:
            return FetchOutcome(
                ok=False,
                error_code="timeout",
                error_message=f"{exc.phase} timeout exceeded",
                bytes_received=0,
            )
        except TransportError as exc:
            return FetchOutcome(
                ok=False,
                error_code=exc.code,
                error_message=str(exc),
                bytes_received=0,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return FetchOutcome(
                ok=False,
                error_code="fetch_failed",
                error_message=f"{type(exc).__name__}: {exc}",
                bytes_received=0,
            )

        last_response = response

        if response.status_code in _REDIRECT_STATUSES:
            location = response.headers.get("location")
            if not location:
                return FetchOutcome(
                    ok=False,
                    error_code="no_redirect_location",
                    error_message="redirect response had no Location header",
                    bytes_received=len(response.body),
                )
            if redirects >= max_redirects:
                return FetchOutcome(
                    ok=False,
                    error_code="too_many_redirects",
                    error_message=f"exceeded the {max_redirects}-redirect limit",
                    bytes_received=len(response.body),
                )
            current = urljoin(validated.url, location)
            redirects += 1
            continue

        break  # final (non-redirect) response

    assert last_response is not None
    resp = last_response
    raw_content_type = resp.headers.get("content-type", "")
    content_type = _content_type(raw_content_type)
    bytes_received = len(resp.body)

    if content_type not in _SUPPORTED_CONTENT_TYPES:
        return FetchOutcome(
            ok=False,
            error_code="unsupported_content",
            error_message=(
                f"content type {content_type or 'unknown'!r} is not supported "
                "(only text/html, text/plain, application/json)"
            ),
            bytes_received=bytes_received,
        )

    body_text, _charset = _decode_body(resp.body, raw_content_type)

    title = ""
    if content_type == "text/html":
        title, extracted = _html_title_and_text(body_text)
    elif content_type == "text/plain":
        extracted = _collapse_plain(body_text)
    else:  # application/json
        extracted = _collapse_plain(_json_to_text(body_text))

    truncated = len(extracted) > effective_text
    final_text = extracted[:effective_text]
    content_hash = hashlib.sha256(final_text.encode("utf-8")).hexdigest()

    links: tuple[str, ...] = ()
    if include_links and content_type == "text/html":
        links = _extract_links(body_text, resp.request_url, max_links)

    result = FetchResult(
        requested_url=url,
        final_url=resp.request_url,
        title=title,
        content_type=content_type or "application/octet-stream",
        status_code=resp.status_code,
        retrieved_at=_now_iso(now),
        text=final_text,
        text_chars=len(final_text),
        content_hash=content_hash,
        truncated=truncated,
        links=links,
    )
    return FetchOutcome(
        ok=True,
        result=result,
        bytes_received=bytes_received,
    )
