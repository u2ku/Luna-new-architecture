"""Channel adapters: web, slack, google_chat, email, …

A channel adapter bridges one external messaging surface (the web UI,
Slack, Google Chat, email, an ad-hoc API, etc.) and the Luna world
ledger. Every channel is responsible for two directions:

* **ingest** — turn a platform-native inbound message into a Luna
  ``user_message`` event with the canonical ``actor`` / ``source`` /
  ``destination`` / ``stream_id`` / ``turn_id`` shape. See
  :mod:`luna.channels.base`.
* **egress** — turn a Luna ``assistant_message`` event back into a
  platform-native outbound message and deliver it.

All channels share the same base contract; per-channel behaviour
lives in ``luna/channels/<platform>/``.
"""
