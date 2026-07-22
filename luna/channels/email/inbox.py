"""Inbox sync: Gmail API → SQLite → world ledger → Luna → outbox.

The first milestone on the email channel. For every new Gmail
message:

    1. Fetch the raw RFC 822 via the Gmail API
    2. Parse headers + body (reuse :mod:`luna.channels.email.receiver`)
    3. Compute the Luna ``stream_id`` + ``turn_id`` and write
       transport data (Gmail IDs, headers, snippet) to SQLite
    4. Append a canonical ``user_message`` event to the world ledger
    5. Hand the event to the chat service, which calls the model
       and appends the ``assistant_message`` event
    6. Write the reply to the outbox (``pending_replies``) with
       status ``'pending'`` — the outbox flush (a separate
       milestone) will pick it up and send via the Gmail API

A separate send-side milestone (outbox flush) is what actually
delivers the pending reply. The read side never sends.

The sync is safe to call repeatedly: every step is idempotent.
Gmail's per-message ID is the dedup key in ``gmail_messages``;
the RFC 822 ``Message-ID`` is the dedup key in ``messages`` and
``pending_replies`` (via ``source_message_id``).
"""

from __future__ import annotations

import email as _email
import logging
from typing import Any
from uuid import uuid4

from ...api.routes import ChatRequest, ChatService
from ...ledger import WorldLedger
from .gmail import GmailClient
from .receiver import _parse_address, _thread_id
from .store import EmailStore


log = logging.getLogger(__name__)


def _body_text(message: Any) -> str:
    """Extract plain-text body, falling back to empty for HTML-only."""
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


class InboxSync:
    """Read-side orchestrator. One instance per sync run."""

    def __init__(
        self,
        *,
        gmail: GmailClient,
        ledger: WorldLedger,
        store: EmailStore,
        chat: ChatService,
        agent_email: str,
    ) -> None:
        self.gmail = gmail
        self.ledger = ledger
        self.store = store
        self.chat = chat
        self.agent_email = agent_email

    def sync(
        self,
        *,
        query: str = "is:unread",
        max_results: int = 20,
    ) -> dict[str, int]:
        """Run one sync pass. Returns counts for the caller.

        Returns a dict with keys:
            * ``fetched``    — number of Gmail message IDs returned
            * ``processed``  — number of messages ingested end-to-end
            * ``skipped``    — already-seen (dedup hit)
            * ``failed``     — any per-message error (caught + logged)
        """
        ids, _ = self.gmail.list_messages(query=query, max_results=max_results)
        stats = {"fetched": len(ids), "processed": 0, "skipped": 0, "failed": 0}

        for gmail_message_id in ids:
            try:
                if self.store.get_gmail_message(gmail_message_id) is not None:
                    stats["skipped"] += 1
                    continue
                raw = self.gmail.get_message_raw(gmail_message_id)
                message = _email.message_from_bytes(raw)
                self._ingest(gmail_message_id, message)
                stats["processed"] += 1
            except Exception:
                log.exception("failed to process gmail message %s", gmail_message_id)
                stats["failed"] += 1
        return stats

    def _ingest(self, gmail_message_id: str, message: Any) -> dict[str, Any]:
        """One inbound email → ledger user_message + reply in outbox."""
        from_email = _parse_address(message.get("From", ""))
        to_email = _parse_address(message.get("To", "")) or self.agent_email
        subject = (message.get("Subject", "") or "").strip()
        rfc822_message_id = (message.get("Message-ID", "") or "").strip()
        in_reply_to = (message.get("In-Reply-To", "") or "").strip()
        thread_anchor = _thread_id(message)
        body = _body_text(message)
        snippet = body[:200]

        if not from_email:
            raise ValueError("email has no From address")
        if not rfc822_message_id:
            raise ValueError("email has no Message-ID header")

        turn_id = str(uuid4())
        # The stream is the conversation thread, not the individual
        # message. Two emails in the same thread (same correspondent
        # + same In-Reply-To/Message-ID anchor) collapse to one
        # stream so the model sees the prior turns as context.
        stream_id = f"email:{from_email}:{thread_anchor}"

        # 1. user_message in the world ledger
        user_event = self.ledger.append(
            event_type="user_message",
            actor={
                "id": f"identity:{from_email}",
                "type": "human",
                "display_name": from_email,
            },
            source={
                "platform": "email",
                "adapter": "gmail-api",
                "account_id": from_email.split("@", 1)[-1] or None,
                "conversation_id": from_email,
                "message_id": rfc822_message_id,
                "external_actor_id": from_email,
                "thread_id": thread_anchor,
            },
            destination={"platform": "luna-runtime"},
            stream_id=stream_id,
            turn_id=turn_id,
            payload={
                "text": body,
                "subject": subject,
                "from": from_email,
                "to": to_email,
                "in_reply_to": in_reply_to or None,
                "message_id_header": rfc822_message_id,
            },
        )

        # 2. transport data in SQLite (so we don't re-ingest on resync)
        self.store.record_gmail_message(
            gmail_message_id=gmail_message_id,
            rfc822_message_id=rfc822_message_id,
            thread_id=thread_anchor,
            account_email=self.agent_email,
            from_addr=from_email,
            subject=subject or None,
            snippet=snippet or None,
            received_at=user_event["timestamp"],
            stream_id=stream_id,
            ledger_event_id=user_event["event_id"],
        )
        # 3. messages / threads (the same rows receiver.py writes;
        #    keeps the SMTP path and the API path on the same shape)
        self.store.record_message(
            message_id=rfc822_message_id,
            thread_id=thread_anchor,
            stream_id=stream_id,
            direction="inbound",
            from_addr=from_email,
            to_addr=to_email,
            subject=subject or None,
            body=body,
            in_reply_to=in_reply_to or None,
            ledger_event_id=user_event["event_id"],
            turn_id=turn_id,
            status="processed",
        )

        # 4. Ask Luna for a reply, via the same ChatService the web
        #    UI uses. Pass the user_event we just wrote so the chat
        #    service uses the same event_id / stream_id / turn_id
        #    and SKIPS writing a duplicate user_event.
        chat_request = ChatRequest(
            text=user_event["payload"]["text"],
            session_id=thread_anchor,
            source="email",
            sender_id=from_email,
            sender_name=from_email,
            conversation_id=from_email,
            thread_id=thread_anchor,
            existing_user_event=user_event,
        )
        chat_response = self.chat.complete(chat_request)

        # 5. Outbox: store the reply as pending. NOT sent. The
        #    outbox flush (next milestone) reads status='pending'
        #    rows and sends via the Gmail API.
        self.store.enqueue_reply(
            stream_id=stream_id,
            turn_id=turn_id,
            source_user_event_id=user_event["event_id"],
            source_message_id=rfc822_message_id,
            to_addr=from_email,
            from_addr=self.agent_email,
            subject=subject,
            in_reply_to=rfc822_message_id,
            references=[in_reply_to] if in_reply_to else None,
            body=chat_response.response,
        )

        log.info(
            "ingested gmail=%s rfc822=%s stream=%s user_event=%s reply_queued",
            gmail_message_id,
            rfc822_message_id,
            stream_id,
            user_event["event_id"],
        )
        return user_event
