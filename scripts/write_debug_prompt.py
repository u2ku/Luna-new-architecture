"""Copy the last line of the model request log to output/debug/last_request.txt.

The request log is written by the model providers
(:mod:`luna.models.openai._OpenAIChatClient`) whenever
``$LUNA_DEBUG_PROMPT_LOG`` points at a JSONL path. Each line is one
``{ts, url, status, request, response}`` record — the literal HTTP
exchange, not a reconstruction.

This script just reads the last line of that JSONL file and copies it
to ``output/debug/last_request.txt``. That way ``last_request.txt`` is
always an exact mirror of what the model actually got on the most
recent call.

Safe to run on a cron every minute. Errors are logged to
``output/debug/cron.log``; the file is left untouched on failure.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "output" / "debug"
OUTPUT_FILE = OUTPUT_DIR / "last_request.txt"
LOG_FILE = OUTPUT_DIR / "cron.log"


def _log_path() -> Path | None:
    raw = __import__("os").environ.get("LUNA_DEBUG_PROMPT_LOG")
    if not raw:
        return None
    return Path(raw)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("write_debug_prompt")

    try:
        log_path = _log_path()
        if log_path is None:
            log.error("LUNA_DEBUG_PROMPT_LOG not set; nothing to mirror")
            return 1
        if not log_path.is_file():
            log.error("log path does not exist: %s", log_path)
            return 1

        # Read the last non-empty line. Newline-delimited JSON; the
        # last line is the most recent model call.
        last_line: str | None = None
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.strip():
                    last_line = line
        if last_line is None:
            log.error("log file is empty: %s", log_path)
            return 1

        # Validate the last line is real JSON; if not, surface the error
        # to the cron log instead of writing garbage to the debug file.
        try:
            record = json.loads(last_line)
        except json.JSONDecodeError as e:
            log.error("last line of %s is not valid JSON: %s", log_path, e)
            return 1

        # The file is the EXACT prompt the model gets. Strip the
        # wrapper — only the request body goes to last_request.txt.
        # The full record (ts, url, status, response) lives in the
        # JSONL log; the .txt is just the prompt.
        request_body = record.get("request")
        if not isinstance(request_body, dict):
            log.error("log entry has no 'request' field: %s", log_path)
            return 1

        OUTPUT_FILE.write_text(
            json.dumps(request_body, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        msg_count = len(request_body.get("messages", []))
        log.info(
            "mirrored last request body of %s to %s (%d messages)",
            log_path,
            OUTPUT_FILE,
            msg_count,
        )
    except Exception:
        log.error("failed: %s", traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
