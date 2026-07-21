"""Inbound email → Luna event.

The receiver parses an RFC 822 / RFC 5322 message, derives the
canonical Luna stream_id from From + threading anchors, builds the
``user_message`` event, writes it to the world ledger, and records
the raw message in the SQLite store for dedup and threading.
"""

from __future__ import annotations

import email
import email.utils
from email.message import EmailMessage
from typing import Any
from uuid import uuid4

from ...ledger import WorldLedger
from .store import EmailStore


def _parse_address(header_value: str) -> str:
    """Return a bare address from a From/To header value.

    ``"Zac <zac@example.com>"`` → ``"zac@example.com"``. Empty
    string if the header is missing or unparseable.
    """
    if not header_value:
        return ""
    name, addr = email.utils.parseaddr(header_value)
    return (addr or "").strip().lower()


def _extract_body(message: EmailMessage) -> str:
    """Return the plain-text body, falling back to the first text part.

    Multipart/alternative and multipart/mixed are walked depth-first.
    HTML-only messages return an empty string — the runtime can
    extend with a sanitizer later.
    """
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                try:
                    return payload.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    return payload.decode("utf-8", errors="replace")
        return ""
    payload = message.get_payload(decode=True) or b""
    charset = message.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="replace")


def _thread_id(message: EmailMessage) -> str:
    """Derive a stable thread id from the message.

    Priority: ``In-Reply-To`` → first ``References`` entry → ``Message-ID``.
    The result is the canonical anchor that ties the inbound email
    to the rest of the conversation.
    """
    in_reply_to = (message.get("In-Reply-To") or "").strip()
    if in_reply_to:
        return in_reply_to
    refs = message.get("References", "")
    if refs:
        first = refs.split()[0].strip()
        if first:
            return first
    return (message.get("Message-ID") or "").strip()


class EmailReceiver:
    """Translate inbound SMTP mail into a canonical Luna event.

    The receiver is the ingest half of the email channel; the
    ``send`` method on the assistant reply is the egress half
    (:mod:`luna.channels.email.sender`).
    """

    def __init__(
        self,
        ledger: WorldLedger,
        store: EmailStore,
        *,
        agent_name: str = "Luna",
    ) -> None:
        self.ledger = ledger
        self.store = store
        self.agent_name = agent_name

    def receive(
        self,
        raw_message: bytes | str | EmailMessage,
    ) -> dict[str, Any]:
        """Process one inbound email end-to-end.

        Accepts raw bytes, a string, or a pre-parsed
        :class:`EmailMessage`. Returns the ``user_message`` event
        that was written to the world ledger.
        """
        if isinstance(raw_message, EmailMessage):
            message = raw_message
        else:
            message = email.message_from_bytes(
                raw_message.encode("utf-8") if isinstance(raw_message, str) else raw_message
            )

        message_id = (message.get("Message-ID") or "").strip()
        from_addr = _parse_address(message.get("From", ""))
        to_addr = _parse_address(message.get("To", ""))
        subject = (message.get("Subject") or "").strip()
        body = _extract_body(message)
        in_reply_to = (message.get("In-Reply-To") or "").strip()
        thread_id = _thread_id(message)

        if not message_id:
            raise ValueError("inbound email has no Message-ID header")
        if not from_addr:
            raise ValueError("inbound email has no From address")

        # stream_id is the security boundary: the From address and
        # the thread anchor together scope this conversation so it
        # can't bleed into another correspondent's thread.
        stream_id = f"email:{from_addr}:{thread_id}:{message_id}"
        turn_id = str(uuid4())

        event = self.ledger.append(
            event_type="user_message",
            actor={
                "id": f"identity:{from_addr}",
                "type": "human",
                "display_name": from_addr,
            },
            source={
                "platform": "email",
                "adapter": "smtp",
                "message_id": message_id,
                "external_actor_id": from_addr,
            },
            destination={"platform": "luna-runtime"},
            stream_id=stream_id,
            turn_id=turn_id,
            payload={
                "text": body,
                "subject": subject,
                "from": from_addr,
                "to": to_addr,
                "in_reply_to": in_reply_to or None,
                "message_id_header": message_id,
            },
        )

        # Record in the email store for dedup and threading. Status
        # 'processed' marks that the message has been written to the
        # ledger and is ready for the chat handler to consume.
        self.store.record_message(
            message_id=message_id,
            thread_id=thread_id,
            stream_id=stream_id,
            direction="inbound",
            from_addr=from_addr,
            to_addr=to_addr,
            subject=subject or None,
            body=body,
            in_reply_to=in_reply_to or None,
            ledger_event_id=event["event_id"],
            turn_id=turn_id,
            status="processed",
        )

        return event
