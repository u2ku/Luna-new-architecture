"""Mirror the last model call to two debug files.

The model providers (:mod:`luna.models.openai._OpenAIChatClient`) write
every chat-completions call to a JSONL log whenever
``$LUNA_DEBUG_PROMPT_LOG`` points at a file. Each line is one
``{ts, url, status, request, response}`` record — the literal HTTP
exchange, not a reconstruction.

This script reads the last line and writes two debug files:

  * ``last_request.txt`` — the JSON request body the API saw. This
    is what goes over HTTP from the runtime to whooshd to mlx-vlm.
  * ``last_prompt_rendered.txt`` — the text gemma actually decodes,
    after mlx-vlm applies gemma's chat template. This is what the
    model itself processes.

Safe to run on a cron every minute. Errors are logged to
``output/debug/cron.log``; the files are left untouched on failure.
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
RENDERED_FILE = OUTPUT_DIR / "last_prompt_rendered.txt"
LOG_FILE = OUTPUT_DIR / "cron.log"


# Standard gemma 2/3/4 chat template (matches HuggingFace's
# google/gemma-*-IT chat_template.json). mlx-vlm applies this on
# the server side; we render it locally so the file matches what
# gemma actually sees, byte for byte.
_GEMMA_TURN_OPEN = {"system": "system", "user": "user", "assistant": "model"}


def _render_gemma(messages: list[dict]) -> str:
    out = "<bos>"
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role not in _GEMMA_TURN_OPEN:
            continue
        out += f"<start_of_turn>{_GEMMA_TURN_OPEN[role]}\n{content}<end_of_turn>\n"
    out += "<start_of_turn>model\n"  # generation prompt
    return out


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

        # Read the last non-empty line.
        last_line: str | None = None
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.strip():
                    last_line = line
        if last_line is None:
            log.error("log file is empty: %s", log_path)
            return 1

        try:
            record = json.loads(last_line)
        except json.JSONDecodeError as e:
            log.error("last line of %s is not valid JSON: %s", log_path, e)
            return 1

        request_body = record.get("request")
        if not isinstance(request_body, dict):
            log.error("log entry has no 'request' field: %s", log_path)
            return 1

        # 1) the JSON request body — what the API saw
        OUTPUT_FILE.write_text(
            json.dumps(request_body, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        msg_count = len(request_body.get("messages", []))

        # 2) the chat-template-rendered text — what gemma actually saw
        messages = request_body.get("messages", [])
        rendered = _render_gemma(messages)
        RENDERED_FILE.write_text(rendered, encoding="utf-8")

        log.info(
            "mirrored last call of %s: %s (%d msgs, %d chars rendered)",
            log_path,
            OUTPUT_FILE,
            msg_count,
            len(rendered),
        )
    except Exception:
        log.error("failed: %s", traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

