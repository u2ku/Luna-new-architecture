"""One-time OAuth consent flow for the email channel.

Walks the Google OAuth2 authorization code flow and prints the
``refresh_token`` you save into ``.env`` as
``LUNA_EMAIL_OAUTH_REFRESH_TOKEN``.

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
    3. The script exchanges the auth code for tokens and prints
       the refresh_token

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
from http.server import BaseHTTPRequestHandler, HTTPServer
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

    print()
    print("  ✓ got tokens. Save these into .env (gitignored):")
    print()
    print(f"    LUNA_EMAIL_OAUTH_REFRESH_TOKEN={refresh}")
    print()
    print(
        "  Verify the token works with: "
        "luna.channels.email.oauth.access_token_from_refresh(...)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
