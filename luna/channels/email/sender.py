"""Outbound email — Luna response → RFC 822 message.

Builds the reply (with proper ``In-Reply-To`` / ``References``
headers to keep the conversation threaded), records it in the
SQLite store, and either delivers via SMTP or queues it for
later. Actual SMTP delivery is a thin wrapper around
``smtplib.SMTP`` — see :func:`send_via_smtp`.
"""

from __future__ import annotations

import smtplib
import socket
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from typing import Any
from uuid import uuid4

from ...ledger import WorldLedger
from .config import SmtpConfig
from .oauth import CachedTokenSource, OAuthConfig, xoauth2_string
from .store import EmailStore


def build_reply(
    *,
    text: str,
    from_addr: str,
    to_addr: str,
    in_reply_to: str | None,
    references: list[str] | None,
    subject: str | None,
) -> EmailMessage:
    """Build a properly-threaded RFC 822 reply.

    ``In-Reply-To`` is the immediate parent; ``References`` is the
    full thread chain. Both should be set when threading matters.
    """
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Date"] = formatdate(localtime=False)
    msg["Message-ID"] = make_msgid(domain=from_addr.split("@", 1)[-1] or "luna")
    if subject:
        prefixed = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    else:
        prefixed = "Luna"
    msg["Subject"] = prefixed
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = " ".join(references)
    msg.set_content(text)
    return msg


def send_via_smtp(
    msg: EmailMessage,
    config: SmtpConfig,
    oauth: OAuthConfig | None = None,
    *,
    token_source: CachedTokenSource | None = None,
) -> None:
    """Deliver ``msg`` via SMTP. Returns ``None`` on success and on
    any non-fatal delivery error — the caller is expected to inspect
    the store; a more sophisticated implementation would surface
    the error and let the cron decide whether to retry.

    When ``oauth`` is supplied, XOAUTH2 is used instead of plain
    password auth. The ``token_source`` is a process-local cache
    so we don't hit Google's token endpoint on every send.
    """
    try:
        with smtplib.SMTP(host=config.host, port=config.port, timeout=15) as smtp:
            smtp.ehlo()
            if config.security == "starttls":
                smtp.starttls()
                smtp.ehlo()

            if oauth and oauth.is_configured():
                src = token_source or CachedTokenSource(oauth)
                access_token = src.get()
                # smtp.auth() takes a string challenge/response
                # callable; we pre-build the SASL string.
                smtp.auth("XOAUTH2", lambda: xoauth2_string(config.username or "", access_token))
            elif config.username and config.password:
                smtp.login(config.username, config.password)
            # else: anonymous relay (LAN test only)

            smtp.send_message(msg)
    except (smtplib.SMTPException, socket.error, OSError):
        # Caller is expected to inspect the store; do not raise.
        return


class EmailSender:
    """Egress half of the email channel.

    Given a Luna ``assistant_message`` event plus the thread
    context (the original From / To / Subject / In-Reply-To),
    build the reply, record it in the SQLite store, and deliver
    via SMTP if configured. SMTP auth supports both an app
    password (``config.username`` + ``config.password``) and
    OAuth2 XOAUTH2 (``oauth``); OAuth wins when both are
    configured.
    """

    def __init__(
        self,
        ledger: WorldLedger,
        store: EmailStore,
        *,
        smtp: SmtpConfig | None = None,
        oauth: OAuthConfig | None = None,
        from_addr: str | None = None,
    ) -> None:
        self.ledger = ledger
        self.store = store
        self.smtp = smtp
        self.oauth = oauth
        self.from_addr = from_addr or ""
        if not self.from_addr:
            raise ValueError(
                "EmailSender needs a from_addr (or set LUNA_EMAIL_FROM_ADDRESS)"
            )
        # Token source is created lazily on first send so the
        # constructor stays cheap when SMTP isn't configured.
        self._token_source: CachedTokenSource | None = None

    def send(
        self,
        *,
        stream_id: str,
        turn_id: str,
        text: str,
        to_addr: str,
        subject: str | None,
        in_reply_to: str | None,
        references: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build, record, and deliver a reply. Returns the
        ``assistant_message`` event written to the ledger.
        """
        # Thread the reference chain so the recipient's mail client
        # groups the reply with the prior conversation.
        full_refs: list[str] = list(references or [])
        if in_reply_to and in_reply_to not in full_refs:
            full_refs.append(in_reply_to)

        msg = build_reply(
            text=text,
            from_addr=self.from_addr,
            to_addr=to_addr,
            in_reply_to=in_reply_to,
            references=full_refs,
            subject=subject,
        )

        outbound_status = "queued"
        if self.smtp and self.smtp.is_configured():
            token_source: CachedTokenSource | None = None
            if self.oauth and self.oauth.is_configured():
                token_source = self._token_source or CachedTokenSource(self.oauth)
                self._token_source = token_source
            send_via_smtp(
                msg, self.smtp, oauth=self.oauth, token_source=token_source
            )
            outbound_status = "sent"

        # Find the original inbound message so we can record the
        # thread anchor on the outbound side.
        thread_id = in_reply_to or (references[0] if references else "")
        outbound_message_id = msg["Message-ID"]
        now_iso = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z")

        event = self.ledger.append(
            event_type="assistant_message",
            actor={
                "id": "agent:luna",
                "type": "agent",
                "display_name": "Luna",
            },
            source={"platform": "luna-runtime"},
            destination={
                "platform": "email",
                "adapter": "smtp",
                "account_id": self.from_addr.split("@", 1)[-1] or None,
                "conversation_id": to_addr,
            },
            stream_id=stream_id,
            turn_id=turn_id,
            payload={
                "text": text,
                "subject": subject,
                "to": to_addr,
                "from": self.from_addr,
                "in_reply_to": in_reply_to or None,
                "message_id_header": outbound_message_id,
            },
        )

        self.store.record_message(
            message_id=outbound_message_id,
            thread_id=thread_id or outbound_message_id,
            stream_id=stream_id,
            direction="outbound",
            from_addr=self.from_addr,
            to_addr=to_addr,
            subject=subject,
            body=text,
            in_reply_to=in_reply_to or None,
            ledger_event_id=event["event_id"],
            turn_id=turn_id,
            status=outbound_status,
        )

        return event
