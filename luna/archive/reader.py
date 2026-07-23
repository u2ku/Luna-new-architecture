"""read_artifact — bounded reading of one known archive result.

Resolves an ``artifact_id`` (returned by ``search_archive``) back to a
file by walking the archive — it never trusts a caller-supplied path.
The model cannot read an arbitrary file: the only ids that resolve are
ones the archive actually contains, and every resolved file is
verified to stay inside ``archive_root`` after symlink resolution.

Safety guarantees:

* **no traversal** — ids resolve only to files inside the archive root;
* **no symlink escape** — realpath containment is checked;
* **no secrets** — paths under a ``secrets`` directory are rejected;
* **no binaries** — non-UTF-8 / NUL-containing ``.md`` files are rejected;
* **bounded output** — at most ``read_max_lines`` lines per call.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..tools.protocol import ToolContext, ToolError, ToolRequest, ToolResult, ToolSpec
from ..tools.registry import ToolResultException
from ._common import (
    BinaryFileError,
    artifact_id_for,
    path_is_forbidden,
    resolve_artifact_id,
    title_of,
    within_root,
    archive_root_realpath,
)

#: Cap on bytes read for a single read call. Archive content files are
#: far smaller; this bounds memory against a future large file.
_MAX_READ_BYTES = 8 * 1024 * 1024

READ_ARTIFACT_SPEC = ToolSpec(
    name="read_artifact",
    description=(
        "Open one archive artifact by its stable artifact_id (from "
        "search_archive) and read a bounded window of lines. Returns "
        "line-numbered content plus metadata. Cannot read arbitrary paths."
    ),
    input_schema={
        "type": "object",
        "required": ["artifact_id"],
        "additionalProperties": False,
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "Stable id returned by search_archive (archive:…).",
                "minLength": 1,
            },
            "start_line": {
                "type": "integer",
                "description": "1-indexed first line to read (default 1).",
                "minimum": 1,
            },
            "line_count": {
                "type": "integer",
                "description": "Lines to read (default 200, max 500).",
                "minimum": 1,
            },
        },
    },
    access="read",
    enabled=True,
)


def _read_text(path: Path) -> str:
    """Read a Markdown file as UTF-8 text, rejecting binaries."""
    with path.open("rb") as fh:
        raw = fh.read(_MAX_READ_BYTES)
    if b"\x00" in raw:
        raise BinaryFileError(str(path))
    return raw.decode("utf-8-sig", errors="strict")


def read_artifact(
    artifact_id: str,
    root: Path | None,
    *,
    start_line: int = 1,
    line_count: int | None = None,
    default_lines: int = 200,
    max_lines: int = 500,
) -> dict[str, Any]:
    """Pure read function (no receipts). Raises ToolError on bad input."""
    if root is None or not root.exists() or not root.is_dir():
        raise ToolResultException(
            ToolError(
                code="archive_unavailable",
                message="archive root not configured or missing",
            )
        )
    resolved = resolve_artifact_id(root, artifact_id)
    if resolved is None:
        raise ToolResultException(
            ToolError(
                code="artifact_not_found",
                message=f"no archive artifact with id {artifact_id!r}",
                details={"artifact_id": artifact_id},
            )
        )
    full, rel = resolved

    # Defence in depth: re-check containment and forbidden names even
    # though resolve_artifact_id already enforces them. A future change
    # to resolution must not weaken these guarantees.
    root_real = archive_root_realpath(root)
    if not within_root(full, root_real):
        raise ToolResultException(
            ToolError(
                code="path_escape",
                message="resolved path escapes the archive root",
            )
        )
    if path_is_forbidden(rel):
        raise ToolResultException(
            ToolError(
                code="forbidden_path",
                message="reading from a secrets directory is not permitted",
            )
        )

    try:
        text = _read_text(full)
    except BinaryFileError as exc:
        raise ToolResultException(
            ToolError(code="binary_file", message=f"not a text file: {exc}")
        ) from exc

    lines = text.splitlines()
    total_lines = len(lines)

    start = max(1, int(start_line) if start_line else 1)
    if start > total_lines and total_lines > 0:
        start = total_lines
    requested = (
        int(line_count) if isinstance(line_count, int) else default_lines
    )
    requested = max(1, min(requested, max_lines))
    end = min(start + requested - 1, total_lines)

    window = [
        {"line": n, "text": lines[n - 1]}
        for n in range(start, end + 1)
    ]
    truncated = (start + requested - 1) > total_lines

    try:
        mtime_ts = full.stat().st_mtime
        modified_at = (
            datetime.fromtimestamp(mtime_ts, tz=timezone.utc)
            .date()
            .isoformat()
            if mtime_ts
            else None
        )
    except OSError:
        modified_at = None

    return {
        "artifact_id": artifact_id_for(rel),
        "title": title_of(text, rel),
        "relative_path": rel.as_posix(),
        "start_line": start,
        "end_line": end,
        "total_lines": total_lines,
        "truncated": truncated,
        "content": window,
        "modified_at": modified_at,
    }


def handle_read_artifact(
    request: ToolRequest, context: ToolContext
) -> ToolResult:
    args = request.arguments
    artifact_id = str(args.get("artifact_id") or "")
    if not artifact_id.strip():
        raise ToolResultException(
            ToolError(code="empty_artifact_id", message="artifact_id must not be empty")
        )
    content = read_artifact(
        artifact_id=artifact_id,
        root=context.archive_root,
        start_line=int(args["start_line"]) if args.get("start_line") is not None else 1,
        line_count=args.get("line_count"),
        default_lines=context.read_default_lines,
        max_lines=context.read_max_lines,
    )
    # Build the receipt digest: enough to prove what was read (a hash of
    # the returned text) and locate it, without storing the content.
    lines_read = content.get("content", [])
    returned_text = "\n".join(entry.get("text", "") for entry in lines_read)
    characters_returned = sum(len(entry.get("text", "")) for entry in lines_read)
    content_hash = hashlib.sha256(returned_text.encode("utf-8")).hexdigest()
    return ToolResult(
        call_id=request.call_id,
        name="read_artifact",
        ok=True,
        content=content,
        artifact_ids=(content["artifact_id"],),
        receipt={
            "result_summary": (
                f"read {content['artifact_id']} lines "
                f"{content['start_line']}-{content['end_line']} "
                f"of {content['total_lines']}"
            ),
            "affected_resources": [content["artifact_id"]],
            "artifact_id": content["artifact_id"],
            "relative_path": content["relative_path"],
            "start_line": content["start_line"],
            "end_line": content["end_line"],
            "characters_returned": characters_returned,
            "truncated": content["truncated"],
            "content_hash": content_hash,
        },
    )
