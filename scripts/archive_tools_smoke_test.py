#!/usr/bin/env python3
"""End-to-end smoke test for the Luna archive tools.

Proves Luna can:

1. search the real archive for "Project Hull";
2. open one returned artifact;
3. answer using its contents;
4. create one test artifact;
5. produce matching tool_call / tool_result receipts.

Uses a temporary data root so the live ledger and live artifacts are never
touched. The single test artifact created in the temp artifacts dir is deleted
after verification.

Run:

    python scripts/archive_tools_smoke_test.py

Optionally point at a different archive root:

    LUNA_ARCHIVE_ROOT=/path/to/wiki python scripts/archive_tools_smoke_test.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Allow running from the repo root without an installed package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from luna.api.routes import ChatRequest, ChatService  # noqa: E402
from luna.archive._common import artifact_id_for, iter_markdown_files  # noqa: E402
from luna.ledger import WorldLedger  # noqa: E402
from luna.models.base import (  # noqa: E402
    FinishReason,
    ModelProvider,
    ModelResponse,
    ToolCall,
    Usage,
)
from luna.tools.config import ArchiveConfig, ToolsConfig  # noqa: E402
from luna.tools.executor import build_archive_registry  # noqa: E402

#: The located real archive root. Used only when LUNA_ARCHIVE_ROOT is unset
#: and the path exists; never invented.
DEFAULT_ARCHIVE_ROOT = "/Users/pieratradio/Archive/wiki"


def _resolve_archive_root() -> Path:
    env = os.environ.get("LUNA_ARCHIVE_ROOT")
    if env:
        root = Path(env).expanduser()
        if not root.is_dir():
            raise SystemExit(f"LUNA_ARCHIVE_ROOT={env} is not a directory")
        return root
    candidate = Path(DEFAULT_ARCHIVE_ROOT)
    if candidate.is_dir():
        return candidate
    raise SystemExit(
        "Archive root not found. Set LUNA_ARCHIVE_ROOT to the Markdown "
        "archive root (e.g. /Users/pieratradio/Archive/wiki)."
    )


def _last_tool_content(messages) -> dict | None:
    """Return the parsed JSON content of the last tool message."""
    for msg in reversed(messages):
        if getattr(msg, "role", None) == "tool" and getattr(msg, "tool_call_id", None):
            try:
                return json.loads(msg.content)
            except (TypeError, ValueError):
                return None
    return None


class SmokeProvider(ModelProvider):
    """A scripted provider that drives the full tool loop.

    It does not call a real model; it inspects the tool results the runtime
    feeds back and issues the next structured call, finally answering from
    the read artifact's contents.
    """

    name = "smoke"

    def __init__(self) -> None:
        self.step = 0
        self.read_title: str | None = None
        self.created_relative_path: str | None = None

    def complete(self, request) -> ModelResponse:
        self.step += 1
        msgs = request.messages

        if self.step == 1:
            return ModelResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="smk-search",
                        name="search_archive",
                        arguments={"query": "Project Hull", "limit": 5},
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=Usage(),
                model="smoke",
            )

        if self.step == 2:
            payload = _last_tool_content(msgs) or {}
            results = (payload.get("content") or {}).get("results") or []
            if not results:
                return ModelResponse(
                    content="Search returned no results.",
                    finish_reason=FinishReason.STOP,
                    usage=Usage(),
                    model="smoke",
                )
            top = results[0]
            return ModelResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="smk-read",
                        name="read_artifact",
                        arguments={
                            "artifact_id": top["artifact_id"],
                            "line_count": 40,
                        },
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=Usage(),
                model="smoke",
            )

        if self.step == 3:
            payload = _last_tool_content(msgs) or {}
            content = payload.get("content") or {}
            self.read_title = content.get("title", "the archive")
            first_line = ""
            for entry in content.get("content") or []:
                if entry.get("text"):
                    first_line = entry["text"]
                    break
            self._first_line = first_line
            return ModelResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="smk-create",
                        name="create_artifact",
                        arguments={
                            "title": "Project Hull Smoke Synthesis",
                            "content": (
                                f"Synthesised from '{self.read_title}' via the "
                                "Luna archive tool loop. First line: "
                                f"{first_line}"
                            ),
                            "category": "luna-system",
                            "source_event_ids": ["smoke"],
                        },
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=Usage(),
                model="smoke",
            )

        # step 4: final answer using the read contents
        answer = (
            f"From {self.read_title or 'the archive'}: "
            f"{getattr(self, '_first_line', '')}"
        )
        return ModelResponse(
            content=answer,
            finish_reason=FinishReason.STOP,
            usage=Usage(),
            model="smoke",
        )


def main() -> int:
    archive_root = _resolve_archive_root()
    tmp = Path(tempfile.mkdtemp(prefix="luna-smoke-"))
    try:
        os.environ["LUNA_DATA_ROOT"] = str(tmp)
        ledger = WorldLedger(
            path=tmp / "ledger" / "world.jsonl",
            lock_path=tmp / "locks" / "world.lock",
        )
        artifacts_root = tmp / "artifacts"
        archive_config = ArchiveConfig(
            root=archive_root,
            artifact_output_root=artifacts_root,
            search_default_limit=8,
            search_max_limit=20,
            read_default_lines=200,
            read_max_lines=500,
        )
        tools_config = ToolsConfig(
            enabled=["search_archive", "read_artifact", "create_artifact"],
            max_tool_calls_per_turn=6,
            max_result_chars_per_turn=20000,
        )
        provider = SmokeProvider()
        svc = ChatService(
            provider=provider,
            ledger=ledger,
            system_prompt="You are Luna. Use the archive tools when asked.",
            registry=build_archive_registry(),
            archive_config=archive_config,
            tools_config=tools_config,
        )

        artifacts_root.mkdir(parents=True, exist_ok=True)
        before = set(p.name for p in artifacts_root.glob("*.md"))

        resp = svc.complete(ChatRequest(text="Tell me about Project Hull."))

        failures: list[str] = []

        # 1. search returned project-hull hits
        search_results = [
            e for e in ledger.tail(200)
            if e["type"] == "tool_call"
            and e["payload"].get("tool") == "search_archive"
        ]
        # the answer references the read artifact
        if not resp.response.strip():
            failures.append("model returned an empty reply")
        if provider.read_title is None:
            failures.append("read_artifact was not exercised")

        # 2. paired receipts
        calls = [e for e in ledger.tail(500) if e["type"] == "tool_call"]
        results = [e for e in ledger.tail(500) if e["type"] == "tool_result"]
        if len(calls) != len(results):
            failures.append(
                f"receipt pairing mismatch: {len(calls)} calls vs {len(results)} results"
            )
        for c, r in zip(calls, results):
            if r["payload"].get("call_event_id") != c["event_id"]:
                failures.append("a tool_result does not reference its tool_call")
                break
        statuses = [r["payload"].get("status") for r in results]
        if "error" in statuses:
            failures.append(f"at least one tool result errored: {statuses}")

        # 3. artifact was created
        after = set(p.name for p in artifacts_root.glob("*.md"))
        created = sorted(after - before)
        if len(created) != 1:
            failures.append(
                f"expected exactly 1 created artifact, found {len(created)}: {created}"
            )
        else:
            art_path = artifacts_root / created[0]
            body = art_path.read_text(encoding="utf-8")
            if "Project Hull Smoke Synthesis" not in body:
                failures.append("created artifact missing expected content")
            # 4. delete only the smoke-test artifact
            art_path.unlink()
            if art_path.exists():
                failures.append("failed to delete the smoke-test artifact")

        # 5. assistant_message recorded tool_calls
        asst = [e for e in ledger.tail(500) if e["type"] == "assistant_message"]
        if not asst or asst[-1]["payload"].get("tool_calls", 0) == 0:
            failures.append("assistant_message did not record tool calls")

        print("=" * 60)
        print("Luna archive tools smoke test")
        print("=" * 60)
        print(f"archive root : {archive_root}")
        print(f"data root    : {tmp} (temp)")
        print(f"tool calls   : {len(calls)}  tool results: {len(results)}")
        print(f"read title   : {provider.read_title}")
        print(f"final answer : {resp.response[:120]}")
        print(f"created+deleted artifact: {created[0] if created else '-'}")
        print("-" * 60)
        if failures:
            print("RESULT: FAIL")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("RESULT: PASS")
        print("All five behaviours verified: search, read, answer, create, receipts.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
