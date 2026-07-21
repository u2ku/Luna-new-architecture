"""Slack channel adapter.

Translate between Slack events API payloads and Luna world events.

Inbound: a Slack ``message`` (or ``app_mention``) from the Events API
or a slash-command payload. The adapter derives ``stream_id`` from
``team_id`` + ``channel_id`` + ``thread_ts`` so the same Slack thread
maps to the same Luna stream.

Outbound: a Slack ``chat.postMessage`` payload — text plus
``thread_ts`` to keep the reply in the same thread.
"""
