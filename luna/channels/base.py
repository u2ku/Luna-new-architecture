"""Shared base class for channel adapters.

Every channel (``web``, ``slack``, ``google_chat``, ``email``, …)
implements the same two-direction contract: ingest a platform-native
message into a canonical Luna event, and egress a Luna response back
into a platform-native message. This module defines that contract;
per-channel implementations live in :mod:`luna.channels.<platform>`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping


class ChannelAdapter(ABC):
    """Base for every channel adapter.

    A channel adapter is a thin translation layer between a
    platform-native message (Slack JSON, an RFC 822 email, a webhook
    payload, an HTTP form) and the canonical Luna event shape
    (see ``schemas/world_event.schema.json``).

    Implementations MUST:

    * Pick a stable ``name`` and ``platform`` (used as the leading
      component of ``stream_id``).
    * Derive ``stream_id`` deterministically from platform-native
      routing identifiers (channel + thread, mailing list + subject
      hash, etc.) so the same conversation maps to the same stream
      across messages.
    * Tag every outbound message with a ``turn_id`` matching the
      inbound ``user_message`` so the ledger can group user/assistant
      pairs.

    Implementations SHOULD NOT:

    * Read "recent messages globally" — use ``recent_message_events``
      with a ``stream_id`` so channels cannot leak into each other.
    * Persist state outside the ledger; if a channel needs memory,
      write events for it.
    """

    #: Stable identifier used in logs and metrics (e.g. ``"web"``).
    name: ClassVar[str] = ""

    #: Platform name used as the leading component of ``stream_id``
    #: (e.g. ``"web"``, ``"slack"``, ``"google_chat"``, ``"email"``).
    platform: ClassVar[str] = ""

    @abstractmethod
    def build_ingest_event(
        self,
        native_message: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate a platform-native message into a Luna event dict.

        Returns the full event body — ``actor``, ``source``,
        ``destination``, ``stream_id``, ``turn_id``, ``payload`` —
        ready to be passed to :meth:`WorldLedger.append`.

        The returned event has no ``event_id``, ``seq``, or
        ``timestamp``; the ledger assigns those on write.
        """
        raise NotImplementedError

    @abstractmethod
    def build_egress_payload(
        self,
        assistant_event: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Translate a Luna ``assistant_message`` event into a platform-native payload.

        Returns whatever the channel needs to deliver the reply —
        a Slack message JSON, an email RFC 822 body, a JSON response
        body, etc. The channel is responsible for actually
        transmitting it.
        """
        raise NotImplementedError

    def stream_id(
        self,
        account_id: str | None,
        conversation_id: str,
        thread_id: str,
    ) -> str:
        """Canonical ``<platform>:<account>:<conversation>:<thread>`` id.

        Empty fields are kept as empty strings (``web:::sess-uuid``),
        not collapsed — a stable format makes the stream id easy
        to grep and parse.
        """
        if not self.platform:
            raise RuntimeError(
                f"{type(self).__name__} must set class attribute `platform`"
            )
        return f"{self.platform}:{account_id or ''}:{conversation_id}:{thread_id}"
