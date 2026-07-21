"""Google Chat channel adapter.

Translate between Google Chat webhook payloads (or the Chat API) and
Luna world events.

Inbound: a Google Chat ``ADDED_TO_SPACE`` / ``MESSAGE`` event. The
adapter derives ``stream_id`` from ``space.name`` + ``thread.name``
(or the message's ``message.thread.threadKey`` for threaded replies).

Outbound: a Google Chat message response body, returned synchronously
to the webhook caller or sent via the Chat API.
"""
