"""Append-only writer for ``world.jsonl``."""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4


class WorldLedger:
    """Write monotonic events to the Luna world ledger."""

    def __init__(self, path: Path, lock_path: Path | None = None) -> None:
        self.path = Path(path)
        self.lock_path = Path(lock_path or self.path.with_suffix(".lock"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.lock_path.touch(exist_ok=True)

    def append(
        self,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append one event and return the exact stored object."""

        if not event_type.strip():
            raise ValueError("event_type must not be empty")
        if not actor.strip():
            raise ValueError("actor must not be empty")

        with self._exclusive_lock():
            event = {
                "event_id": str(uuid4()),
                "seq": self._last_sequence() + 1,
                "timestamp": datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                "type": event_type,
                "actor": actor,
                "payload": dict(payload),
            }

            encoded = json.dumps(
                event,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            return event

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the last ``limit`` valid events."""

        if limit < 1:
            return []

        events: list[dict[str, Any]] = []

        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    events.append(value)

        return events[-limit:]

    def _last_sequence(self) -> int:
        last = 0

        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                    seq = int(value.get("seq", 0))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                last = max(last, seq)

        return last

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
