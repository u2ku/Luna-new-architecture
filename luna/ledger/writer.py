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
    """Write monotonic events to the Luna world ledger.

    Events follow the multi-channel ``world_event`` schema. The
    ``actor``, ``source``, and ``destination`` fields are objects;
    ``stream_id`` is required so per-channel/per-thread context can
    be isolated when reading the ledger back.
    """

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
        actor: Mapping[str, Any] | str,
        payload: Mapping[str, Any],
        *,
        source: Mapping[str, Any] | None = None,
        destination: Mapping[str, Any] | None = None,
        stream_id: str | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Append one event and return the exact stored object.

        Required: ``event_type``, ``actor``, ``payload``,
        ``stream_id``. The writer adds ``event_id``, ``seq``, and
        ``timestamp`` so the on-disk shape is consistent.
        """
        if not event_type.strip():
            raise ValueError("event_type must not be empty")
        if not stream_id or not stream_id.strip():
            raise ValueError("stream_id is required (events must be scoped to a stream)")

        # Coerce a string actor to the object shape for backward
        # compatibility with older callers; the canonical form is a
        # dict with at least {id, type}.
        if isinstance(actor, str):
            actor_obj: dict[str, Any] = {"id": actor, "type": "system"}
        else:
            actor_obj = dict(actor)
            if "id" not in actor_obj or not actor_obj["id"]:
                raise ValueError("actor must include 'id'")

        with self._exclusive_lock():
            event: dict[str, Any] = {
                "event_id": str(uuid4()),
                "seq": self._last_sequence() + 1,
                "timestamp": datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                "type": event_type,
                "actor": actor_obj,
                "source": dict(source) if source else {"platform": "luna-runtime"},
                "destination": (
                    dict(destination) if destination else {"platform": "luna-runtime"}
                ),
                "stream_id": stream_id,
                "payload": dict(payload),
            }
            if turn_id is not None:
                event["turn_id"] = turn_id

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
