"""Web channel — the private web UI.

The FastAPI chat handler currently lives in :mod:`luna.api.routes`
under ``/api/chat`` and writes events with ``source.platform ==
"web"``. This package is where a future ``WebChannel`` adapter
will live once the chat handler is split out of ``routes.py``.

For now this is a marker so the channel package layout is in
place. The web channel does NOT need to be re-implemented — the
existing routes.py is the web channel; the split is a refactor for
later.
"""
