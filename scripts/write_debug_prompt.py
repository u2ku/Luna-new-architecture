"""Write the model request body to output/debug/last_request.txt.

Built to be safe to run on a cron every minute. Reads the live ledger,
constructs the request body the runtime would send to whooshd for the
most recent user turn, and writes it as pretty-printed JSON to the
debug file. Any error is logged to output/debug/cron.log so the cron
job can fail visibly without spamming the user.

Usage:
    python3 scripts/write_debug_prompt.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Project layout: this script lives at <root>/scripts/write_debug_prompt.py
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from luna.context.recent_events import ledger_path as default_ledger_path
from luna.context.builder import build_recent_messages, group_into_turns

OUTPUT_DIR = ROOT / "output" / "debug"
OUTPUT_FILE = OUTPUT_DIR / "last_request.txt"
LOG_FILE = OUTPUT_DIR / "cron.log"

SYSTEM_PROMPT = (
    "You are Luna.\n"
    "This is the new runtime test.\n"
    "Reply directly and clearly."
)

DEFAULT_MODEL = "gemma-4-26B-A4B-it-4bit"
TEMPERATURE = 0.3
MAX_TOKENS = 800
MAX_RECENT_EVENTS = 24  # leave room in the window for the system prompt + current turn


def _latest_user_text(events) -> str | None:
    """Find the most recent user message in the recent-event slice.

    If the trailing message is already a user message with no reply,
    that's the current turn. Otherwise the last user message is what
    the model would be responding to.
    """
    for event in reversed(events):
        if event.is_user and event.text:
            return event.text
    return None


def build_request_body() -> dict:
    """Construct the JSON body the runtime would POST to whooshd.

    Reads the live ledger, takes the last MAX_RECENT_EVENTS message
    events, and assembles a messages list with the system prompt at
    the head and the latest user turn at the tail.
    """
    events = build_recent_messages(limit=MAX_RECENT_EVENTS)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for e in events:
        role = "user" if e.is_user else "assistant" if e.is_assistant else None
        if role is None:
            continue
        messages.append({"role": role, "content": e.text})

    latest_user = _latest_user_text(events)
    if latest_user is not None:
        # Append the current turn as the last user message. If the
        # trailing event is already a user message, this would be a
        # duplicate — only append if it's different from the last
        # user message in the list.
        if not messages or messages[-1] != {"role": "user", "content": latest_user}:
            messages.append({"role": "user", "content": latest_user})

    return {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("write_debug_prompt")

    try:
        body = build_request_body()
        body["_meta"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ledger": str(default_ledger_path()),
            "recent_events_used": sum(
                1 for m in body["messages"] if m["role"] != "system"
            ),
        }
        OUTPUT_FILE.write_text(
            json.dumps(body, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log.info("wrote %s (%d messages)", OUTPUT_FILE, len(body["messages"]))
    except Exception:
        log.error("failed: %s", traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
