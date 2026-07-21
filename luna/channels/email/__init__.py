"""Email channel adapter.

Translate between RFC 822 / RFC 5322 email and Luna world events.

Inbound: a parsed ``email.message.EmailMessage`` (or a dict with the
fields the adapter needs). The adapter derives ``stream_id`` from
``From`` + a thread key (typically ``In-Reply-To`` or the
``References`` header chain, falling back to ``Subject``).

Outbound: a ``MIMEMultipart`` message with the Luna response as
the body and the proper ``In-Reply-To`` / ``References`` headers to
keep the conversation threaded.
"""
