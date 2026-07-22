"""Network safety layer for the web research tools.

Every URL the ``fetch_webpage`` tool requests — and every URL a redirect
points at — is validated here *before* a connection is opened. The layer
enforces the public-web-only contract:

* only ``http`` / ``https`` schemes (no ``file``, ``data``,
  ``javascript``, ``ftp``, ``gopher``, ``smb``, ``mailto``);
* no userinfo (usernames or passwords) in the URL;
* no ``localhost`` or ``.local`` hostnames;
* the resolved destination must not be loopback, private, link-local,
  multicast, reserved, unspecified, or an IPv4-mapped private IPv6
  address;
* explicit ports must be in the configured allowlist (80 and 443 by
  default; extra ports configurable but denied by default).

DNS resolution is the one network touch-point the layer needs, and it
is injected (``resolver``) so tests can mock it without contacting the
real internet. The default resolver uses :func:`socket.getaddrinfo`.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Callable, Sequence
from urllib.parse import urlparse

#: Hosts that are never resolved — rejected by name before DNS.
_BLOCKED_HOSTNAMES: frozenset[str] = frozenset({"localhost"})

#: Schemes a public-web fetch is allowed to use.
_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

#: Schemes explicitly rejected (documented in the spec). Any scheme not
#: in ``_ALLOWED_SCHEMES`` is rejected; this set just makes the intent
#: explicit and testable.
_REJECTED_SCHEMES: frozenset[str] = frozenset(
    {"file", "data", "javascript", "ftp", "gopher", "smb", "mailto"}
)


class UrlValidationError(Exception):
    """Raised when a URL is not safe to fetch.

    ``code`` is a stable machine identifier (e.g. ``"private_address"``,
    ``"invalid_scheme"``); ``message`` is human-readable.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


#: Signature of a DNS resolver. Returns the list of address strings the
#: host resolves to. The default uses ``socket.getaddrinfo``; tests inject
#: a fake. Never raises on a bare lookup — a resolution failure is turned
#: into a :class:`UrlValidationError` by :func:`validate_url`.
Resolver = Callable[[str], list[str]]


def default_resolver(host: str) -> list[str]:
    """Resolve ``host`` to address strings via :func:`socket.getaddrinfo`.

    Returns an empty list on failure (``validate_url`` turns an empty
    resolution into a ``dns_failed`` error). Only A/AAAA records are
    returned; the port is irrelevant to safety classification.
    """
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError):
        return []
    seen: list[str] = []
    for family, _stype, _proto, _canon, sockaddr in infos:
        if family == socket.AF_INET:
            ip = sockaddr[0]
        elif family == socket.AF_INET6:
            ip = sockaddr[0]
            # Strip a scope id if present (``fe80::1%eth0``).
            if "%" in ip:
                ip = ip.split("%", 1)[0]
        else:
            continue
        if ip not in seen:
            seen.append(ip)
    return seen


@dataclass(frozen=True)
class ValidatedUrl:
    """A URL that passed every safety check."""

    scheme: str
    host: str
    port: int | None
    resource: str  # path + query + fragment
    url: str  # the normalised URL actually requested


def _is_blocked_hostname(host: str) -> bool:
    low = host.lower().strip(".")
    if low in _BLOCKED_HOSTNAMES:
        return True
    return low.endswith(".local")


def classify_address(ip_str: str) -> str:
    """Classify an address string into a safety category.

    Returns one of: ``unspecified``, ``loopback``, ``link_local``,
    ``multicast``, ``reserved``, ``private``, ``ipv4_mapped_private``,
    or ``public``. The first matching (most specific) category wins.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return "invalid"

    # IPv4-mapped IPv6 (``::ffff:a.b.c.d``). If the embedded IPv4 is in
    # any forbidden range, treat it as mapped-private so a request for
    # ``::ffff:127.0.0.1`` does not slip past.
    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped
        if mapped is not None:
            if mapped.is_loopback or mapped.is_private or mapped.is_link_local:
                return "ipv4_mapped_private"

    if ip.is_unspecified:
        return "unspecified"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link_local"
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved:
        return "reserved"
    if ip.is_private:
        return "private"
    return "public"


#: Categories that must never be fetched.
_FORBIDDEN_CATEGORIES: frozenset[str] = frozenset(
    {
        "unspecified",
        "loopback",
        "link_local",
        "multicast",
        "reserved",
        "private",
        "ipv4_mapped_private",
        "invalid",
    }
)


def is_address_allowed(ip_str: str) -> bool:
    """True if a resolved address is on the public internet."""
    return classify_address(ip_str) == "public"


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def validate_url(
    url: str,
    *,
    allowed_ports: Sequence[int],
    resolver: Resolver = default_resolver,
) -> ValidatedUrl:
    """Validate a URL is safe to fetch; raise :class:`UrlValidationError` if not.

    Called before the initial request and after every redirect, so a
    redirect into a private address is caught at the boundary.
    """
    if not isinstance(url, str) or not url.strip():
        raise UrlValidationError("empty_url", "url must not be empty")

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if not scheme:
        raise UrlValidationError("invalid_scheme", "url must have a scheme")
    if scheme in _REJECTED_SCHEMES:
        raise UrlValidationError(
            "invalid_scheme", f"scheme {scheme!r} is not allowed"
        )
    if scheme not in _ALLOWED_SCHEMES:
        raise UrlValidationError(
            "invalid_scheme", f"only http and https schemes are allowed, got {scheme!r}"
        )

    # Userinfo (username/password) in the URL is credentials on a link —
    # rejected outright. ``urlparse`` exposes them only when present.
    if parsed.username or parsed.password:
        raise UrlValidationError(
            "credentials_in_url",
            "urls with usernames or passwords are not allowed",
        )

    host = (parsed.hostname or "").lower().strip(".")
    if not host:
        raise UrlValidationError("invalid_host", "url must have a host")

    if _is_blocked_hostname(host):
        raise UrlValidationError(
            "blocked_hostname", f"hostname {host!r} is not allowed"
        )

    # Explicit port must be in the allowlist. An implicit port (the
    # scheme default) is allowed — 80/443 are inherently public.
    port: int | None = parsed.port
    if port is not None and port not in allowed_ports:
        raise UrlValidationError(
            "port_not_allowed",
            f"port {port} is not in the allowed list {sorted(allowed_ports)}",
        )

    # If the host is already a literal IP, classify it directly. Otherwise
    # resolve and require *every* resolved address to be public — a host
    # that resolves to even one private address is rejected.
    literal = _maybe_literal_ip(host)
    if literal is not None:
        if not is_address_allowed(literal):
            raise UrlValidationError(
                "private_address",
                f"address {literal} is not on the public internet",
            )
    else:
        addresses = resolver(host)
        if not addresses:
            raise UrlValidationError(
                "dns_failed", f"could not resolve hostname {host!r}"
            )
        for addr in addresses:
            if not is_address_allowed(addr):
                raise UrlValidationError(
                    "private_address",
                    f"{host} resolved to a non-public address {addr}",
                )

    # Rebuild a normalised URL with no userinfo, lower-cased host.
    resource = parsed.path or "/"
    if parsed.query:
        resource += "?" + parsed.query
    if parsed.fragment:
        resource += "#" + parsed.fragment
    final_port = "" if port is None or port == _default_port(scheme) else f":{port}"
    normalised = f"{scheme}://{host}{final_port}{resource}"
    return ValidatedUrl(
        scheme=scheme,
        host=host,
        port=port,
        resource=resource,
        url=normalised,
    )


def _maybe_literal_ip(host: str) -> str | None:
    """Return the address string if ``host`` is a literal IP, else None."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return None
    return host
