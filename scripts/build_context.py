"""Build a model context from the live world.jsonl and print it.

Demonstrates that the context builder is reading the actual ledger,
not a stub. Run from the runtime root with LUNA_DATA_ROOT set::

    LUNA_DATA_ROOT=/path/to/LunaData python3 scripts/build_context.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from luna.context.builder import build_recent_messages, group_into_turns
from luna.context.recent_events import ledger_path


def main() -> int:
    path = ledger_path()
    print(f"ledger: {path}")
    if not path.is_file():
        print("  (not found — set LUNA_DATA_ROOT)", file=sys.stderr)
        return 1

    print()

    # --- 1. flat recent events ---
    # stream_id is required: events are scoped per-channel/thread so a
    # private web conversation can't bleed into a Slack context.
    # For the live ledger we use the first stream_id we find.
    import json as _json
    from luna.context.recent_events import _iter_events, ledger_path as _ledger_path
    stream_id: str | None = None
    for event in _iter_events(_ledger_path()):
        sid = event.get("stream_id")
        if sid:
            stream_id = sid
            break
    if stream_id is None:
        print("  (no stream_id'd events in ledger yet)", file=sys.stderr)
        return 1

    events = build_recent_messages(stream_id=stream_id, limit=25)
    print(f"# 1. build_recent_messages(stream_id={stream_id!r}, limit=25): {len(events)} events")
    print("#    (oldest → newest, the slice the model would see)")
    print()
    for e in events:
        role = "user " if e.is_user else "luna " if e.is_assistant else "?    "
        text = e.text[:70].replace("\n", " ")
        print(f"  seq={e.seq:3d}  {e.type:18s}  {role}  {text!r}")
    print()

    # --- 2. grouped into turns ---
    turns = group_into_turns(events)
    complete = sum(1 for t in turns if t.assistant is not None)
    incomplete = sum(1 for t in turns if t.assistant is None)
    print(f"# 2. group_into_turns(): {len(turns)} turns")
    print(f"#    {complete} complete, {incomplete} incomplete (user without reply)")
    print()
    for i, t in enumerate(turns, 1):
        u = t.user.text[:60].replace("\n", " ") if t.user else "(no user)"
        a = t.assistant.text[:60].replace("\n", " ") if t.assistant else "(no reply)"
        marker = " " if t.assistant else "*"
        print(f"  {marker} turn {i:2d}  user: {u!r}")
        print(f"             luna: {a!r}")
    print()

    # --- 3. ledger summary ---
    with path.open("r", encoding="utf-8") as fh:
        total = sum(1 for line in fh if line.strip())
    print(f"# 3. ledger: {total} total events on disk")
    print(f"#    builder pulled {len(events)} / {total} as message events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
