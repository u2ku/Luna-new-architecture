"""Email channel adapter.

Translate between RFC 822 / RFC 5322 email and Luna world events.

Inbound: a parsed ``email.message.EmailMessage`` (or bytes/string).
The adapter derives ``stream_id`` from ``From`` + a thread key
(typically ``In-Reply-To`` or the first ``References`` entry,
falling back to ``Message-ID``).

Outbound: a properly-threaded ``MIMEMultipart`` message with the
Luna response as the body and the proper ``In-Reply-To`` /
``References`` headers so the recipient's mail client groups the
reply with the prior conversation.

Deployment
----------

SMTP credentials are NEVER in source. Set them in ``.env`` (which
is gitignored) using the placeholders in ``.env.example``. Two
auth modes are supported; OAuth wins when both are configured.

* **App password** — set ``LUNA_EMAIL_SMTP_USERNAME`` and
  ``LUNA_EMAIL_SMTP_PASSWORD``. For Gmail, generate an app
  password at https://myaccount.google.com/apppasswords.
* **OAuth2 XOAUTH2** — register a Google Cloud OAuth client
  (Desktop app type), run ``scripts/email_bootstrap_oauth.py``
  once to walk the consent flow and capture the
  ``refresh_token``, then set ``LUNA_EMAIL_OAUTH_CLIENT_ID`` /
  ``_SECRET`` / ``_REFRESH_TOKEN`` (and optionally
  ``_SCOPES``; default is send-only).

``Security`` is one of:

    * ``starttls`` (default) — plaintext on connect, ``STARTTLS`` upgrade
    * ``tls``                 — implicit TLS (port 465)
    * ``none``                — plaintext (do not use on the public internet)

The SQLite index lives at ``$LUNA_DATA_ROOT/email/email.db``; the
directory and schema are created on first open.
"""
