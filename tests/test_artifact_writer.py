"""Tests for create_artifact: atomic writes, duplicates, secrets, enforcement."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from luna.archive.artifact_writer import (
    DuplicateArtifactError,
    SecretContentError,
    create_artifact,
    handle_create_artifact,
)
from luna.tools.protocol import ToolContext, ToolRequest


def _ctx(output_root: Path) -> ToolContext:
    return ToolContext(
        archive_root=Path("/nonexistent-archive"),
        artifact_output_root=output_root,
        search_default_limit=8,
        search_max_limit=20,
        read_default_lines=200,
        read_max_lines=500,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
        stream_id="web::test:test",
        turn_id="turn-1",
    )


def test_atomic_creation(tmp_path):
    out = tmp_path / "artifacts"
    created = create_artifact(
        title="Archive Tool Implementation Decision",
        content="We chose a retrieval-based archive.\n\nBounded reads.",
        output_root=out,
        category="luna-system",
        source_event_ids=["event-1", "event-2"],
        now=datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc),
    )
    assert created.path.exists()
    assert created.path.parent == out
    assert created.relative_path == created.path.name
    assert created.artifact_id.startswith("artifact:")
    body = created.path.read_text(encoding="utf-8")
    assert "artifact_id" in body
    assert "Archive Tool Implementation Decision" in body
    assert "luna-system" in body
    assert "event-1" in body
    assert "# Archive Tool Implementation Decision" in body


def test_duplicate_filename_rejected(tmp_path):
    out = tmp_path / "artifacts"
    when = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    first = create_artifact(
        title="Same Title", content="body", output_root=out, now=when
    )
    with pytest.raises(DuplicateArtifactError):
        create_artifact(
            title="Same Title", content="body two", output_root=out, now=when
        )
    # original file untouched
    assert first.path.read_text(encoding="utf-8").endswith("body\n")


def test_duplicate_does_not_overwrite_content(tmp_path):
    out = tmp_path / "artifacts"
    when = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    first = create_artifact(
        title="Dup Check", content="ORIGINAL", output_root=out, now=when
    )
    try:
        create_artifact(
            title="Dup Check", content="OVERWRITE", output_root=out, now=when
        )
    except DuplicateArtifactError:
        pass
    assert "ORIGINAL" in first.path.read_text(encoding="utf-8")
    assert "OVERWRITE" not in first.path.read_text(encoding="utf-8")


def test_output_root_enforcement(tmp_path):
    out = tmp_path / "artifacts"
    created = create_artifact(
        title="Enforced", content="body", output_root=out,
        now=datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc),
    )
    # The created file must live strictly under the configured output root.
    assert out in created.path.parents or created.path == out
    # relative_path must not carry directory traversal components
    assert "/" not in created.relative_path
    assert not created.relative_path.startswith(".")


def test_handler_ignores_caller_supplied_path(tmp_path):
    out = tmp_path / "artifacts"
    ctx = _ctx(out)
    # A caller tries to smuggle a path/filename field. The handler must
    # ignore it (write under the configured root) — and the schema's
    # additionalProperties:false would already reject it at the registry.
    req = ToolRequest(
        name="create_artifact",
        arguments={
            "title": "Smuggled",
            "content": "body",
            "category": "luna-system",
            "path": "/tmp/evil.md",  # must be ignored
        },
        call_id="c1",
    )
    # Bypass registry validation (which would reject the extra field) to
    # prove the handler itself never uses a caller path.
    result = handle_create_artifact(req, ctx)
    assert result.ok is True
    assert "/tmp/evil.md" not in result.content["relative_path"]
    assert (out / result.content["relative_path"]).exists()


def test_secret_content_rejected(tmp_path):
    out = tmp_path / "artifacts"
    with pytest.raises(SecretContentError):
        create_artifact(
            title="Leak",
            content="the api_key=sk-abc123 and password=hunter2 here",
            output_root=out,
        )
    assert not out.exists() or not any(out.iterdir())


def test_secret_rejection_handler(tmp_path):
    ctx = _ctx(tmp_path / "artifacts")
    req = ToolRequest(
        name="create_artifact",
        arguments={
            "title": "Leak",
            "content": "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIx"
            + "a" * 60,
        },
        call_id="c1",
    )
    from luna.tools.registry import ToolResultException

    with pytest.raises(ToolResultException) as exc:
        handle_create_artifact(req, ctx)
    assert exc.value.error.code == "content_contains_secret"


def test_slug_is_date_prefixed(tmp_path):
    out = tmp_path / "artifacts"
    created = create_artifact(
        title="Milestone Report",
        content="body",
        output_root=out,
        now=datetime(2026, 7, 22, 9, 30, tzinfo=timezone.utc),
    )
    assert created.path.name.startswith("2026-07-22-")
    assert "milestone-report" in created.path.name
