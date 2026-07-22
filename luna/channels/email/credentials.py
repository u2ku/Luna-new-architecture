"""Load Gmail OAuth credentials and tokens from ``LunaData/secrets/gmail/``.

The Google OAuth flow writes two files into the secrets directory:

* ``credentials.json`` — the "installed app" client. Has an
  ``installed`` key with ``client_id``, ``client_secret``,
  ``redirect_uris``, ``auth_uri``, ``token_uri``.
* ``token.json`` — the token bundle from a successful auth.
  Has ``token`` (access_token), ``refresh_token``,
  ``token_uri``, ``client_id``, ``client_secret``, ``scopes``,
  ``expiry``.

Both files are read-only at runtime — Luna never writes to them.
The bootstrap script (:mod:`scripts.email_bootstrap_oauth`)
creates them.

Neither path is hardcoded. Tests and the CLI can override via
``LUNA_GMAIL_SECRETS_DIR``; the default is
``$LUNA_DATA_ROOT/secrets/gmail``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .oauth import OAuthConfig, TokenResponse, access_token_from_refresh


DEFAULT_SECRETS_SUBDIR = "secrets/gmail"


def default_secrets_dir() -> Path:
    """Return the canonical secrets directory, or env override."""
    override = os.environ.get("LUNA_GMAIL_SECRETS_DIR", "").strip()
    if override:
        return Path(override)
    root = os.environ.get("LUNA_DATA_ROOT", "").strip()
    if not root:
        return Path("LunaData") / DEFAULT_SECRETS_SUBDIR
    return Path(root) / DEFAULT_SECRETS_SUBDIR


@dataclass(frozen=True)
class GmailCredentials:
    """Resolved Gmail OAuth credentials and a live access token.

    ``access_token()`` is the only method the Gmail API client
    needs. It returns a fresh, cached access token (refreshing
    from the refresh_token in ``token.json`` when needed).
    """

    client_id: str
    client_secret: str
    refresh_token: str
    scopes: tuple[str, ...]
    token_uri: str
    expiry: str  # ISO timestamp from token.json; informational

    @classmethod
    def from_secrets_dir(cls, secrets_dir: Path | None = None) -> "GmailCredentials":
        """Load credentials from ``credentials.json`` and ``token.json``."""
        secrets = Path(secrets_dir) if secrets_dir else default_secrets_dir()
        creds_path = secrets / "credentials.json"
        token_path = secrets / "token.json"
        if not creds_path.is_file():
            raise FileNotFoundError(
                f"missing Gmail OAuth credentials: {creds_path} — "
                f"create the file or run scripts/email_bootstrap_oauth.py"
            )
        if not token_path.is_file():
            raise FileNotFoundError(
                f"missing Gmail OAuth token: {token_path} — run "
                f"scripts/email_bootstrap_oauth.py to walk the consent flow"
            )
        creds = json.loads(creds_path.read_text(encoding="utf-8"))
        token = json.loads(token_path.read_text(encoding="utf-8"))

        # credentials.json uses Google's "installed app" envelope:
        #   {"installed": {"client_id": ..., "client_secret": ..., ...}}
        installed = creds.get("installed", creds)
        client_id = installed.get("client_id") or token.get("client_id", "")
        client_secret = installed.get("client_secret") or token.get("client_secret", "")

        if not client_id or not client_secret:
            raise ValueError(
                f"client_id / client_secret missing in {creds_path}"
            )
        if not token.get("refresh_token"):
            raise ValueError(
                f"refresh_token missing in {token_path} — re-run "
                f"the bootstrap with prompt=consent"
            )

        scopes_raw = token.get("scopes") or installed.get("scopes") or []
        if isinstance(scopes_raw, str):
            scopes = tuple(scopes_raw.split())
        else:
            scopes = tuple(scopes_raw)

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=token["refresh_token"],
            scopes=scopes,
            token_uri=token.get("token_uri") or installed.get("token_uri", ""),
            expiry=token.get("expiry", ""),
        )

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def _oauth_config(self) -> OAuthConfig:
        return OAuthConfig(
            client_id=self.client_id,
            client_secret=self.client_secret,
            refresh_token=self.refresh_token,
            scopes=" ".join(self.scopes) or "https://www.googleapis.com/auth/gmail.readonly",
            token_uri=self.token_uri or "https://oauth2.googleapis.com/token",
        )

    def access_token(self) -> str:
        """Return a fresh, cached access token (refreshes if expired)."""
        cached = getattr(self, "_cached_token", None)
        now = time.time()
        if cached and cached.expires_at - 30 > now:
            return cached.access_token
        resp = access_token_from_refresh(self._oauth_config(), _now=now)
        object.__setattr__(self, "_cached_token", resp)
        return resp.access_token
