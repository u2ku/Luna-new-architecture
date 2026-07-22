"""Email channel adapter.

Translate between RFC 822 / RFC 5322 email and Luna world events.

Inbound: a parsed ``email.message.EmailMessage`` (or a dict with the
fields the adapter needs). The adapter derives ``stream_id`` from
``From`` + a thread key (typically ``In-Reply-To`` or the
``References`` header chain, falling back to ``Subject``).

Outbound: a ``MIMEMultipart`` message with the Luna response as
the body and the proper ``In-Reply-To`` / ``References`` headers to
keep the conversation threaded.

Deployment
----------

SMTP credentials are NEVER in source. Set them in ``.env`` (which
is gitignored) using the placeholders in ``.env.example``::

    LUNA_EMAIL_SMTP_HOST=smtp.gmail.com
    LUNA_EMAIL_SMTP_PORT=587
    LUNA_EMAIL_SMTP_SECURITY=starttls
    LUNA_EMAIL_SMTP_USERNAME=your-smtp-username
    LUNA_EMAIL_SMTP_PASSWORD=your-app-password
    LUNA_EMAIL_FROM_ADDRESS=luna@yourdomain.com

Gmail specifically requires an app password (not the account
password) — generate one at
https://myaccount.google.com/apppasswords.

``Security`` is one of:
    * ``starttls`` (default) — plaintext on connect, ``STARTTLS`` upgrade
    * ``tls``                 — implicit TLS (port 465)
    * ``none``                — plaintext (do not use on the public internet)

The SQLite index lives at ``$LUNA_DATA_ROOT/email/email.db``; the
directory and schema are created on first open.
"""
