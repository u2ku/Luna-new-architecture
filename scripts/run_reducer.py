"""Run the reducer against the live ledger and print the result.

Usage:
    python3 scripts/run_reducer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running this file directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from luna.context.recent_events import ledger_path
from luna.reducer.current_state import handle as current_state_handle
from luna.reducer.reducer import Reducer


HANDLERS = {
    "user_message": current_state_handle,
    "assistant_message": current_state_handle,
}


def main() -> int:
    path = ledger_path()
    print(f"ledger: {path}")
    if not path.is_file():
        print("  (not found)", file=sys.stderr)
        return 1

    reducer = Reducer(handlers=HANDLERS)

    present = reducer.discover_event_types(path)
    registered = set(HANDLERS)
    print()
    print(f"event types in ledger: {sorted(present)}")
    print(f"handlers registered:   {sorted(registered)}")
    print(
        f"filtered (no handler): {sorted(present - registered) or '(none)'}"
    )
    print()

    result = reducer.reduce(path)
    print("stats:")
    for key, value in sorted(result.stats.items()):
        print(f"  {key:24s} {value}")
    print()

    print("derived state:")
    serializable = {
        k: (
            v.__dict__
            if hasattr(v, "__dict__") and not isinstance(v, type)
            else v
        )
        for k, v in result.state.items()
    }
    text = json.dumps(serializable, indent=2, ensure_ascii=False, default=str)
    # Truncate very long assistant text for readability.
    print(text[:2000] + ("\n  …(truncated)" if len(text) > 2000 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
