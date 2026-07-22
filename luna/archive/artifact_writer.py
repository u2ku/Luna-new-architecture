"""create_artifact — durable Markdown creation under LunaData/artifacts.

Writes only under the configured artifact output root. The filename is
generated inside the tool (a date-prefixed slug); callers cannot supply
a filesystem path. Creation is atomic and never overwrites: a duplicate
filename fails rather than clobbering existing work.

Intended for durable decisions, synthesis documents, specifications,
and milestones — not transient conversation. The handler rejects
secret/token-shaped content rather than silently persisting it.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..tools.protocol import ToolContext, ToolError, ToolRequest, ToolResult, ToolSpec
from ..tools.registry import ToolResultException

#: Regex for the slug. Keep alphanumerics and hyphens; collapse the
#: rest. Matches the archive's own kebab-case filenames.
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SLUG_MAX = 60

# Patterns that mark secret / token-shaped content. If any match, the
# artifact is rejected outright (explicit rejection, never redaction
# into storage — the model is told to remove the secret and retry).
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|bearer)\s*[:=]"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36}"),  # GitHub token
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack token
    # A long opaque bearer-style token on its own.
    re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._\-]{20,}"),
)


CREATE_ARTIFACT_SPEC = ToolSpec(
    name="create_artifact",
    description=(
        "Persist a durable Markdown artifact (a decision, spec, or "
        "synthesis) under LunaData/artifacts. The filename is generated "
        "for you; do not pass a path. Rejects secrets in content."
    ),
    input_schema={
        "type": "object",
        "required": ["title", "content"],
        "additionalProperties": False,
        "properties": {
            "title": {
                "type": "string",
                "description": "Human-readable title; becomes the filename slug and H1.",
                "minLength": 1,
            },
            "content": {
                "type": "string",
                "description": "Markdown body (no frontmatter; the tool adds provenance).",
                "minLength": 1,
            },
            "category": {
                "type": "string",
                "description": "Logical category stored in the provenance header (e.g. luna-system).",
            },
            "source_event_ids": {
                "type": "array",
                "description": "Ledger event ids this artifact synthesises.",
                "items": {"type": "string"},
            },
        },
    },
    access="write",
    enabled=True,
)


class DuplicateArtifactError(Exception):
    """Raised when an artifact filename already exists."""


class SecretContentError(Exception):
    """Raised when content contains a secret/token-shaped value."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def detect_secret_content(content: str) -> list[str]:
    """Return a list of reasons (empty = content is clean)."""
    reasons: list[str] = []
    for pattern in _SECRET_PATTERNS:
        match = pattern.search(content)
        if match:
            reasons.append(f"matched secret pattern: {pattern.pattern[:40]!r}")
    return reasons


def _slugify(title: str) -> str:
    slug = _SLUG_RE.sub("-", title.lower().strip()).strip("-")
    slug = slug[:_SLUG_MAX].strip("-")
    return slug or "untitled"


def _filename(title: str, when: datetime) -> str:
    date = when.strftime("%Y-%m-%d")
    slug = _slugify(title)
    # A short hash of the title disambiguates same-day same-slug titles
    # without relying on a counter (which would race under concurrency).
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:6]
    return f"{date}-{slug}-{digest}.md"


def _provenance_header(
    *,
    artifact_id: str,
    title: str,
    when: datetime,
    category: str,
    source_event_ids: Iterable[str],
) -> str:
    import json

    lines = [
        "---",
        f"artifact_id: {json.dumps(artifact_id)}",
        f"title: {json.dumps(title)}",
        f"created_at: {when.isoformat(timespec='seconds')}",
        f"category: {json.dumps(category)}",
        "generator: luna.create_artifact",
        f"source_event_ids: {json.dumps(list(source_event_ids))}",
        "---",
        "",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class CreatedArtifact:
    artifact_id: str
    relative_path: str
    title: str
    category: str
    created_at: str
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "relative_path": self.relative_path,
            "title": self.title,
            "category": self.category,
            "created_at": self.created_at,
        }


def create_artifact(
    *,
    title: str,
    content: str,
    output_root: Path,
    category: str = "uncategorised",
    source_event_ids: Iterable[str] = (),
    now: datetime | None = None,
) -> CreatedArtifact:
    """Create one Markdown artifact atomically. Never overwrites.

    Raises :class:`SecretContentError` if content carries a
    secret/token-shaped value, and :class:`DuplicateArtifactError` if
    the generated filename already exists.
    """
    if not title.strip():
        raise ValueError("title must not be empty")
    if not content.strip():
        raise ValueError("content must not be empty")

    reasons = detect_secret_content(content)
    if reasons:
        raise SecretContentError(reasons)

    when = now or datetime.now(timezone.utc)
    filename = _filename(title, when)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    final_path = output_root / filename
    rel_path = Path(filename)

    artifact_id = "artifact:" + hashlib.sha1(
        rel_path.as_posix().encode("utf-8")
    ).hexdigest()[:16]

    header = _provenance_header(
        artifact_id=artifact_id,
        title=title,
        when=when,
        category=category,
        source_event_ids=source_event_ids,
    )
    body = f"{header}# {title}\n\n{content.rstrip()}\n"

    # Atomic publish: write a temp file in the same directory, fsync,
    # then hard-link it into place. os.link fails if the target exists,
    # which is the no-overwrite guarantee.
    fd, tmp_name = tempfile.mkstemp(
        prefix=".luna-artifact-", suffix=".tmp", dir=str(output_root)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp_path, final_path)
        except FileExistsError as exc:
            raise DuplicateArtifactError(
                f"artifact already exists: {final_path.name}"
            ) from exc
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    return CreatedArtifact(
        artifact_id=artifact_id,
        relative_path=rel_path.as_posix(),
        title=title,
        category=category,
        created_at=when.isoformat(timespec="seconds"),
        path=final_path,
    )


def handle_create_artifact(
    request: ToolRequest, context: ToolContext
) -> ToolResult:
    args = request.arguments
    title = str(args.get("title") or "")
    content = str(args.get("content") or "")
    category = str(args.get("category") or "uncategorised")
    source_event_ids = args.get("source_event_ids") or []

    # The handler always writes under the configured output root from
    # context — it never accepts a path from the caller. (The schema's
    # additionalProperties:false makes a path field impossible anyway.)
    try:
        created = create_artifact(
            title=title,
            content=content,
            output_root=context.artifact_output_root,
            category=category,
            source_event_ids=source_event_ids,
        )
    except SecretContentError as exc:
        raise ToolResultException(
            ToolError(
                code="content_contains_secret",
                message=(
                    "content contains a secret or token-shaped value; "
                    f"remove it and retry ({exc})"
                ),
                details={"reasons": exc.reasons},
            )
        ) from exc
    except DuplicateArtifactError as exc:
        raise ToolResultException(
            ToolError(
                code="duplicate_artifact",
                message=str(exc),
            )
        ) from exc
    except ValueError as exc:
        raise ToolResultException(
            ToolError(code="invalid_arguments", message=str(exc))
        ) from exc

    return ToolResult(
        call_id=request.call_id,
        name="create_artifact",
        ok=True,
        content=created.to_dict(),
        artifact_ids=(created.artifact_id,),
    )
