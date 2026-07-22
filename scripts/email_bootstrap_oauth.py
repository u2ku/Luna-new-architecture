"""One-time OAuth consent flow for the email channel.

Walks the Google OAuth2 authorization code flow and writes the
resulting token bundle to ``$LUNA_DATA_ROOT/secrets/gmail/token.json``.
The bundle is what :class:`GmailCredentials` reads back at runtime.

Prerequisites
------------

1. Create a Google Cloud project.
2. Enable the Gmail API.
3. Create an OAuth 2.0 Client ID of type **Desktop app** under
   "APIs & Services" → "Credentials". Download the client_id and
   client_secret.
4. Add ``http://localhost:8765/callback`` to the list of
   "Authorized redirect URIs" for that client.

Usage
-----

    LUNA_EMAIL_OAUTH_CLIENT_ID=…apps.googleusercontent.com \
    LUNA_EMAIL_OAUTH_CLIENT_SECRET=… \
    python3 scripts/email_bootstrap_oauth.py

The script will:
    1. Open your browser to Google's consent screen
    2. After you approve, Google redirects to a local HTTP server
    3. The script exchanges the auth code for tokens and writes
       the full token bundle to
       ``$LUNA_DATA_ROOT/secrets/gmail/token.json``
    4. Also prints the ``LUNA_EMAIL_OAUTH_REFRESH_TOKEN`` value
       for callers that want to use env-var auth instead of the
       on-disk bundle (e.g. serverless deployments).

The script is deliberately offline (no extra dependencies) — it
uses Python's stdlib ``http.server`` and ``webbrowser``.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
# Union of scopes the email channel needs end-to-end:
#   gmail.readonly — list + read messages (inbox sync)
#   gmail.compose  — create / update drafts, send mail (outbox flush)
DEFAULT_SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly "
    "https://www.googleapis.com/auth/gmail.compose"
)
REDIRECT_URI = "http://localhost:8765/callback"
REDIRECT_PORT = 8765


def _default_secrets_dir() -> Path:
    """Mirror the runtime's default for the secrets dir."""
    override = os.environ.get("LUNA_GMAIL_SECRETS_DIR", "").strip()
    if override:
        return Path(override)
    root = os.environ.get("LUNA_DATA_ROOT", "").strip()
    if not root:
        return Path("LunaData") / "secrets" / "gmail"
    return Path(root) / "secrets" / "gmail"


def _build_auth_url(client_id: str, scopes: str) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": scopes,
        "access_type": "offline",
        "prompt": "consent",  # force a refresh_token every time
    }
    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


def _exchange_code_for_tokens(
    client_id: str, client_secret: str, code: str
) -> dict:
    body = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        GOOGLE_TOKEN_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_code(
    client_id: str, client_secret: str, scopes: str
) -> dict:
    """Open browser, run a localhost callback server, return tokens."""
    received: dict = {}
    done = Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — http.server convention
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            if "code" in qs:
                received["code"] = qs["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<h2>Authorized. You can close this tab.</h2>"
                )
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"missing code")
            done.set()

        def log_message(self, *_):  # silence stderr
            return

    auth_url = _build_auth_url(client_id, scopes)
    print(f"  opening browser to: {auth_url}")
    if not webbrowser.open(auth_url):
        print("  (browser didn't open automatically; copy the URL above)")

    server = HTTPServer(("127.0.0.1", REDIRECT_PORT), Handler)
    server.timeout = 120
    print("  waiting for callback on http://localhost:8765/callback …")
    while not done.is_set():
        server.handle_request()
    server.server_close()
    if "code" not in received:
        raise SystemExit("did not receive an authorization code")
    return _exchange_code_for_tokens(
        client_id, client_secret, received["code"]
    )


def main() -> int:
    client_id = os.environ.get("LUNA_EMAIL_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.environ.get("LUNA_EMAIL_OAUTH_CLIENT_SECRET", "").strip()
    scopes = os.environ.get("LUNA_EMAIL_OAUTH_SCOPES", DEFAULT_SCOPES).strip()

    if not client_id or not client_secret:
        print(
            "LUNA_EMAIL_OAUTH_CLIENT_ID and LUNA_EMAIL_OAUTH_CLIENT_SECRET "
            "must be set",
            file=sys.stderr,
        )
        return 2

    print(f"  client_id:     {client_id}")
    print(f"  scopes:        {scopes}")
    print(f"  redirect_uri:  {REDIRECT_URI}")
    print()

    tokens = _wait_for_code(client_id, client_secret, scopes)
    refresh = tokens.get("refresh_token")
    if not refresh:
        print(
            "  no refresh_token in the response. Common causes:",
            "  • the consent screen was previously granted (revoke at "
            "https://myaccount.google.com/permissions and re-run)",
            "  • the OAuth client is a 'Web' type, not 'Desktop' (Desktop "
            "apps get refresh tokens reliably)",
            sep="\n",
            file=sys.stderr,
        )
        return 1

    # Build the on-disk token bundle in google-auth's standard shape
    # so GmailCredentials.from_secrets_dir can read it back unchanged.
    expires_in = int(tokens.get("expires_in", 3600))
    expiry = (
        datetime.now(timezone.utc).timestamp() + expires_in
    )
    expiry_iso = datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    bundle = {
        "token": tokens.get("access_token", ""),
        "refresh_token": refresh,
        "token_uri": GOOGLE_TOKEN_URL,
        "client_id": client_id,
        "client_secret": client_secret,
        "scopes": scopes.split(),
        "universe_domain": "googleapis.com",
        "account": "",
        "expiry": expiry_iso,
    }

    secrets_dir = _default_secrets_dir()
    secrets_dir.mkdir(parents=True, exist_ok=True)
    token_path = secrets_dir / "token.json"
    token_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        token_path.chmod(0o600)
    except OSError:
        pass

    print()
    print(f"  ✓ token bundle written to {token_path}")
    print()
    print("  Runtime picks this up automatically. To verify:")
    print(
        "    python3 -c \"from luna.channels.email import "
        "GmailCredentials; c = GmailCredentials.from_secrets_dir(); "
        "print(c.access_token()[:24] + '…')\""
    )
    print()
    print("  (Optional) for env-only auth, also save the refresh token:")
    print(f"    LUNA_EMAIL_OAUTH_REFRESH_TOKEN={refresh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
