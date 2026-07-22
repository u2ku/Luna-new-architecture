"""Email channel adapter.

Three transports share the same event shape (``world_event``):

* **Gmail API** (read + outbox + send) — preferred; OAuth2 via
  ``LunaData/secrets/gmail/{credentials,token}.json``
* **SMTP** (send only) — fallback when the API isn't configured.
  App password or OAuth2 XOAUTH2.
* **IMAP** (read) — not implemented yet; the ``receiver`` module
  is shared with the API path and is ready for it.

The adapter derives ``stream_id`` from ``From`` + a thread key
(typically ``In-Reply-To`` or the first ``References`` entry,
falling back to ``Message-ID``).

OAuth scopes
------------

The Gmail default scope is the union the channel needs end-to-end:

* ``https://www.googleapis.com/auth/gmail.readonly`` — list +
  read messages (the inbox sync)
* ``https://www.googleapis.com/auth/gmail.compose``  — create
  / update drafts, send mail (the outbox flush)

Override via ``LUNA_EMAIL_OAUTH_SCOPES`` in ``.env``. For
send-only deployments, set
``LUNA_EMAIL_OAUTH_SCOPES="https://www.googleapis.com/auth/gmail.send"``.

Re-authorization
----------------

If you change scopes, delete ``LunaData/secrets/gmail/token.json``
and re-run ``scripts/email_bootstrap_oauth.py`` so Google issues
a token with the new scope set.
"""

from .config import SmtpConfig, default_from_address
from .credentials import GmailCredentials, default_secrets_dir
from .gmail import GmailAPIError, GmailClient
from .inbox import InboxSync
from .oauth import (
    CachedTokenSource,
    OAuthConfig,
    OAuthError,
    TokenResponse,
    access_token_from_refresh,
    xoauth2_string,
)
from .receiver import EmailReceiver, _parse_address, _thread_id
from .sender import EmailSender, build_reply, send_via_smtp
from .store import EmailStore


__all__ = [
    "SmtpConfig",
    "default_from_address",
    "GmailCredentials",
    "default_secrets_dir",
    "GmailClient",
    "GmailAPIError",
    "OAuthConfig",
    "OAuthError",
    "TokenResponse",
    "access_token_from_refresh",
    "xoauth2_string",
    "CachedTokenSource",
    "EmailStore",
    "EmailReceiver",
    "EmailSender",
    "build_reply",
    "send_via_smtp",
    "_parse_address",
    "_thread_id",
    "InboxSync",
]
