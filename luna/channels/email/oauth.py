"""Google OAuth2 (XOAUTH2) for the email channel's SMTP auth.

Gmail's SMTP server supports two auth methods:

* **App password** — username + 16-char app password, sent via
  ``smtp.login()``. Simple, but Google's anti-abuse policy
  increasingly pushes users to OAuth.
* **OAuth2** — a short-lived access token, sent via
  ``AUTH XOAUTH2`` after STARTTLS. The access token is obtained
  by exchanging a long-lived ``refresh_token`` at Google's token
  endpoint. Refresh tokens do not expire unless explicitly revoked.

This module supports OAuth2 (the modern path). App-password auth
still works through :mod:`.config` + :mod:`.sender` if the runtime
operator has not migrated.

OAuth flow
----------

1. **One time**: register a Google Cloud OAuth client (Desktop
   app type). Run :mod:`scripts.email_bootstrap_oauth` once to
   walk the consent flow and capture the ``refresh_token``.
2. **Per send**: call :func:`access_token_from_refresh` to get a
   fresh ``access_token`` (cached until ~5 minutes from expiry to
   avoid pointless refresh calls), then use the
   :data:`XOAUTH2_FORMAT` string in the SMTP ``AUTH`` command.

Credentials in env vars
-----------------------

Set in ``.env`` (gitignored), placeholders in ``.env.example``::

    LUNA_EMAIL_OAUTH_CLIENT_ID=…apps.googleusercontent.com
    LUNA_EMAIL_OAUTH_CLIENT_SECRET=…
    LUNA_EMAIL_OAUTH_REFRESH_TOKEN=1//0e…
    LUNA_EMAIL_OAUTH_SCOPES=https://mail.google.com/

The ``scopes`` value is the OAuth scope to ask for. ``https://mail.google.com/``
grants full Gmail access; ``https://www.googleapis.com/auth/gmail.send``
grants send-only. The bootstrap script defaults to send-only.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# XOAUTH2 SASL: ``user=<email>\x01auth=Bearer <token>\x01\x01``.
# See https://developers.google.com/gmail/imap/xoauth2-protocol
XOAUTH2_FORMAT = "user={user}\x01auth=Bearer {token}\x01\x01"


@dataclass(frozen=True)
class OAuthConfig:
    """Google OAuth2 client + refresh token.

    Attributes
    ----------
    client_id, client_secret:
        From the Google Cloud console's "OAuth 2.0 Client IDs"
        page (Desktop app type). Treat client_secret as a secret.
    refresh_token:
        Long-lived token obtained by the one-time consent flow
        (see ``scripts/email_bootstrap_oauth.py``). Does not
        expire unless the user revokes it or the refresh-token
        quota is exceeded.
    scopes:
        Space-separated OAuth scopes. ``https://mail.google.com/``
        for full Gmail, ``https://www.googleapis.com/auth/gmail.send``
        for send-only.
    token_uri:
        Google's token endpoint. Defaults to the production URL;
        override only for testing.
    """

    client_id: str
    client_secret: str
    refresh_token: str
    scopes: str = "https://www.googleapis.com/auth/gmail.send"
    token_uri: str = GOOGLE_TOKEN_URL

    @classmethod
    def from_env(cls) -> "OAuthConfig":
        return cls(
            client_id=os.environ.get("LUNA_EMAIL_OAUTH_CLIENT_ID", "").strip(),
            client_secret=os.environ.get(
                "LUNA_EMAIL_OAUTH_CLIENT_SECRET", ""
            ).strip(),
            refresh_token=os.environ.get(
                "LUNA_EMAIL_OAUTH_REFRESH_TOKEN", ""
            ).strip(),
            scopes=os.environ.get(
                "LUNA_EMAIL_OAUTH_SCOPES",
                "https://www.googleapis.com/auth/gmail.send",
            ).strip(),
        )

    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret and self.refresh_token)


@dataclass(frozen=True)
class TokenResponse:
    """A freshly-issued access token plus its expiry.

    Returned by :func:`access_token_from_refresh`. ``expires_at``
    is an absolute ``time.time()`` value, not a duration, so
    caches can compare directly.
    """

    access_token: str
    expires_at: float


def access_token_from_refresh(
    config: OAuthConfig,
    *,
    _now: float | None = None,
) -> TokenResponse:
    """Exchange a refresh token for a fresh access token.

    Hits Google's token endpoint with the standard
    ``grant_type=refresh_token`` request. Raises
    :class:`OAuthError` on any failure.
    """
    if not config.is_configured():
        raise OAuthError("OAuthConfig is not fully populated")
    now = _now if _now is not None else time.time()
    body = urllib.parse.urlencode(
        {
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "refresh_token": config.refresh_token,
            "grant_type": "refresh_token",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        config.token_uri,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise OAuthError(f"token endpoint returned {e.code}: {body}") from e
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        raise OAuthError(f"token endpoint unreachable: {e}") from e
    if "error" in payload:
        raise OAuthError(
            f"token endpoint returned error: {payload.get('error')!r} "
            f"{payload.get('error_description', '')!r}"
        )
    token = payload.get("access_token")
    if not token:
        raise OAuthError(f"token endpoint returned no access_token: {payload!r}")
    # ``expires_in`` is the access_token lifetime in seconds from
    # ``now``. Google typically returns 3599 (one hour minus a
    # second). Fall back to 3600 if the field is missing.
    expires_in = int(payload.get("expires_in", 3600))
    return TokenResponse(access_token=token, expires_at=now + expires_in)


def xoauth2_string(user: str, access_token: str) -> str:
    """Build the SASL string for the SMTP ``AUTH XOAUTH2`` command."""
    return XOAUTH2_FORMAT.format(user=user, token=access_token)


class OAuthError(RuntimeError):
    """Raised when an OAuth flow or token exchange fails."""


class CachedTokenSource:
    """Caches access tokens so we don't refresh on every send.

    Google access tokens live ~1 hour. The cache is in-memory and
    process-local; for a long-running daemon this is fine. If
    multiple Luna processes are running, each will refresh
    independently — that's safe (refresh tokens are designed for
    concurrent use).
    """

    # Refresh a little before the real expiry to avoid a request
    # landing in the 0–30 second pre-expiry window where the
    # server might already consider the token invalid.
    REFRESH_MARGIN = 30.0

    def __init__(self, config: OAuthConfig) -> None:
        self._config = config
        self._cached: TokenResponse | None = None

    def get(self) -> str:
        now = time.time()
        if (
            self._cached is not None
            and self._cached.expires_at - self.REFRESH_MARGIN > now
        ):
            return self._cached.access_token
        self._cached = access_token_from_refresh(self._config, _now=now)
        return self._cached.access_token

    def invalidate(self) -> None:
        """Drop the cached token. Useful on AUTH failure."""
        self._cached = None

