"""Tests for the network safety layer (luna.web.security)."""

from __future__ import annotations

import pytest

from luna.web.security import (
    UrlValidationError,
    classify_address,
    is_address_allowed,
    validate_url,
)

PUBLIC = ["8.8.8.8", "1.1.1.1"]


def _resolves_to(public=PUBLIC):
    return lambda host: list(public)


# ---------------------------------------------------------------------------
# Scheme validation
# ---------------------------------------------------------------------------


def test_https_url_validates():
    v = validate_url("https://example.com/path", allowed_ports=[80, 443], resolver=_resolves_to())
    assert v.scheme == "https"
    assert v.host == "example.com"
    assert v.url == "https://example.com/path"


def test_http_url_validates():
    v = validate_url("http://example.com", allowed_ports=[80, 443], resolver=_resolves_to())
    assert v.scheme == "http"
    # bare path normalised to "/"
    assert v.url == "http://example.com/"


@pytest.mark.parametrize(
    "scheme", ["file", "data", "javascript", "ftp", "gopher", "smb", "mailto"]
)
def test_unsupported_scheme_rejected(scheme):
    with pytest.raises(UrlValidationError) as exc:
        validate_url(f"{scheme}://example.com/", allowed_ports=[80, 443], resolver=_resolves_to())
    assert exc.value.code == "invalid_scheme"


def test_missing_scheme_rejected():
    with pytest.raises(UrlValidationError) as exc:
        validate_url("example.com/path", allowed_ports=[80, 443], resolver=_resolves_to())
    assert exc.value.code == "invalid_scheme"


# ---------------------------------------------------------------------------
# Credentials / userinfo
# ---------------------------------------------------------------------------


def test_credential_bearing_url_rejected():
    with pytest.raises(UrlValidationError) as exc:
        validate_url(
            "http://user:pass@example.com/", allowed_ports=[80, 443], resolver=_resolves_to()
        )
    assert exc.value.code == "credentials_in_url"


def test_username_only_url_rejected():
    with pytest.raises(UrlValidationError) as exc:
        validate_url(
            "http://user@example.com/", allowed_ports=[80, 443], resolver=_resolves_to()
        )
    assert exc.value.code == "credentials_in_url"


# ---------------------------------------------------------------------------
# Hostname blocks
# ---------------------------------------------------------------------------


def test_localhost_rejected():
    with pytest.raises(UrlValidationError) as exc:
        validate_url("http://localhost/", allowed_ports=[80, 443], resolver=lambda h: ["127.0.0.1"])
    assert exc.value.code == "blocked_hostname"


def test_dotlocal_rejected():
    with pytest.raises(UrlValidationError) as exc:
        validate_url("http://printer.local/", allowed_ports=[80, 443], resolver=_resolves_to())
    assert exc.value.code == "blocked_hostname"


# ---------------------------------------------------------------------------
# Address classification
# ---------------------------------------------------------------------------


def test_private_ipv4_rejected():
    for ip in ("10.0.0.1", "172.16.0.1", "192.168.1.1"):
        with pytest.raises(UrlValidationError) as exc:
            validate_url(f"http://example.com/", allowed_ports=[80, 443], resolver=lambda h: [ip])
        assert exc.value.code == "private_address"


def test_loopback_ipv4_rejected():
    with pytest.raises(UrlValidationError) as exc:
        validate_url("http://example.com/", allowed_ports=[80, 443], resolver=lambda h: ["127.0.0.1"])
    assert exc.value.code == "private_address"


def test_private_ipv6_rejected():
    with pytest.raises(UrlValidationError) as exc:
        validate_url(
            "http://example.com/", allowed_ports=[80, 443], resolver=lambda h: ["fc00::1"]
        )
    assert exc.value.code == "private_address"


def test_loopback_ipv6_rejected():
    with pytest.raises(UrlValidationError) as exc:
        validate_url(
            "http://example.com/", allowed_ports=[80, 443], resolver=lambda h: ["::1"]
        )
    assert exc.value.code == "private_address"


def test_link_local_rejected():
    for ip in ("169.254.1.1", "fe80::1"):
        with pytest.raises(UrlValidationError) as exc:
            validate_url(
                "http://example.com/", allowed_ports=[80, 443], resolver=lambda h: [ip]
            )
        assert exc.value.code == "private_address"


def test_ipv4_mapped_private_ipv6_rejected():
    with pytest.raises(UrlValidationError) as exc:
        validate_url(
            "http://example.com/", allowed_ports=[80, 443], resolver=lambda h: ["::ffff:127.0.0.1"]
        )
    assert exc.value.code == "private_address"
    assert classify_address("::ffff:10.0.0.1") == "ipv4_mapped_private"


def test_multicast_reserved_unspecified_rejected():
    for ip in ("224.0.0.1", "240.0.0.1", "0.0.0.0"):
        with pytest.raises(UrlValidationError) as exc:
            validate_url(
                "http://example.com/", allowed_ports=[80, 443], resolver=lambda h: [ip]
            )
        assert exc.value.code == "private_address"


def test_literal_ip_host_classified_directly():
    with pytest.raises(UrlValidationError) as exc:
        validate_url("http://10.0.0.1/", allowed_ports=[80, 443], resolver=_resolves_to())
    assert exc.value.code == "private_address"
    # a public literal IP is allowed
    v = validate_url("http://8.8.8.8/", allowed_ports=[80, 443], resolver=_resolves_to())
    assert v.host == "8.8.8.8"


def test_one_private_among_many_resolutions_rejected():
    # a host resolving to both public and private must be rejected
    with pytest.raises(UrlValidationError) as exc:
        validate_url(
            "http://example.com/",
            allowed_ports=[80, 443],
            resolver=lambda h: ["8.8.8.8", "10.0.0.1"],
        )
    assert exc.value.code == "private_address"


def test_dns_failure_rejected():
    with pytest.raises(UrlValidationError) as exc:
        validate_url(
            "http://nonexistent.invalid/", allowed_ports=[80, 443], resolver=lambda h: []
        )
    assert exc.value.code == "dns_failed"


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


def test_default_ports_allowed():
    v = validate_url("https://example.com/", allowed_ports=[80, 443], resolver=_resolves_to())
    assert v.port is None
    v2 = validate_url("http://example.com:80/", allowed_ports=[80, 443], resolver=_resolves_to())
    assert v2.port == 80


def test_extra_port_denied_by_default():
    with pytest.raises(UrlValidationError) as exc:
        validate_url("http://example.com:8080/", allowed_ports=[80, 443], resolver=_resolves_to())
    assert exc.value.code == "port_not_allowed"


def test_configurable_extra_port_allowed():
    v = validate_url(
        "http://example.com:8080/", allowed_ports=[80, 443, 8080], resolver=_resolves_to()
    )
    assert v.port == 8080


def test_classify_public():
    assert classify_address("8.8.8.8") == "public"
    assert is_address_allowed("1.1.1.1") is True
    assert is_address_allowed("10.0.0.1") is False
