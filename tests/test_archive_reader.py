"""Tests for read_artifact: resolution, safety, bounded reading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from luna.archive._common import artifact_id_for
from luna.archive.reader import handle_read_artifact, read_artifact
from luna.archive.search import search_archive
from luna.tools.protocol import ToolContext, ToolRequest
from luna.tools.registry import ToolResultException


def _ctx(root: Path) -> ToolContext:
    return ToolContext(
        archive_root=root,
        artifact_output_root=Path("/tmp/luna-artifacts-test"),
        search_default_limit=8,
        search_max_limit=20,
        read_default_lines=200,
        read_max_lines=500,
        actor={"id": "agent:luna", "type": "agent"},
        source={"platform": "luna-runtime"},
        stream_id="web::test:test",
        turn_id="turn-1",
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_read_returns_line_numbers_and_metadata(tmp_path):
    root = tmp_path / "archive"
    _write(root / "doc.md", "# Title\n\nline one\nline two\nline three\n")
    res = search_archive("line", root, limit=1)
    aid = res["results"][0]["artifact_id"]
    out = read_artifact(aid, root, line_count=2)
    assert out["artifact_id"] == aid
    assert out["title"] == "Title"
    assert out["relative_path"] == "doc.md"
    assert out["total_lines"] == 5
    assert out["start_line"] == 1
    assert out["end_line"] == 2
    assert out["truncated"] is False
    assert out["content"][0]["line"] == 1
    assert out["content"][0]["text"] == "# Title"
    assert out["content"][1]["line"] == 2


def test_bounded_line_reading_and_max_clamp(tmp_path):
    root = tmp_path / "archive"
    body = "\n".join(f"line {i}" for i in range(1, 601))  # 600 lines
    _write(root / "big.md", body)
    aid = artifact_id_for(Path("big.md"))
    # default 200 lines
    out = read_artifact(aid, root)
    assert out["end_line"] - out["start_line"] + 1 == 200
    # max clamp to 500 even if more requested
    out2 = read_artifact(aid, root, line_count=10000)
    assert out2["end_line"] - out2["start_line"] + 1 == 500


def test_read_truncation(tmp_path):
    root = tmp_path / "archive"
    _write(root / "short.md", "\n".join(f"l{i}" for i in range(50)))
    aid = artifact_id_for(Path("short.md"))
    out = read_artifact(aid, root, line_count=200)
    assert out["end_line"] == 50
    assert out["truncated"] is True  # asked for 200, only 50 remained
    assert out["total_lines"] == 50


def test_read_start_line_offset(tmp_path):
    root = tmp_path / "archive"
    _write(root / "doc.md", "\n".join(f"line {i}" for i in range(1, 21)))
    aid = artifact_id_for(Path("doc.md"))
    out = read_artifact(aid, root, start_line=15, line_count=10)
    assert out["start_line"] == 15
    assert out["end_line"] == 20
    assert out["content"][0]["line"] == 15
    assert out["content"][0]["text"] == "line 15"


def test_path_traversal_rejected(tmp_path):
    root = tmp_path / "archive"
    _write(root / "real.md", "# Real\ncontent")
    # A forged id derived from a traversal path never resolves, because
    # resolution walks only real files inside the root.
    forged = artifact_id_for(Path("../../etc/passwd"))
    with pytest.raises(ToolResultException) as exc:
        read_artifact(forged, root)
    assert exc.value.error.code == "artifact_not_found"


def test_symlink_escape_rejected(tmp_path):
    root = tmp_path / "archive"
    _write(root / "real.md", "# Real\nhull content")
    # secret file OUTSIDE the archive
    outside = tmp_path / "outside-secret.md"
    _write(outside, "# Secret\nshould not be readable")
    # symlink inside the archive pointing outside
    link = root / "escaped.md"
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("cannot create symlink on this platform")
    # search must not return the symlinked escapee
    res = search_archive("secret", root, limit=10)
    assert "escaped.md" not in [r["relative_path"] for r in res["results"]]
    # and its id (if forged) does not resolve either
    forged = artifact_id_for(Path("escaped.md"))
    with pytest.raises(ToolResultException) as exc:
        read_artifact(forged, root)
    assert exc.value.error.code == "artifact_not_found"


def test_secret_root_rejected(tmp_path):
    # Archive root that contains a secrets/ directory.
    root = tmp_path / "archive"
    _write(root / "secrets" / "creds.md", "# Credentials\npassword=hunter2")
    _write(root / "notes.md", "# Notes\nproject hull notes")
    # secrets never appear in search
    res = search_archive("password", root, limit=10)
    assert all("secrets/" not in r["relative_path"] for r in res["results"])
    # and a forged id for the secrets file cannot be read
    forged = artifact_id_for(Path("secrets/creds.md"))
    with pytest.raises(ToolResultException) as exc:
        read_artifact(forged, root)
    assert exc.value.error.code == "artifact_not_found"


def test_binary_file_rejected(tmp_path):
    root = tmp_path / "archive"
    # A .md file that is actually binary (NUL bytes).
    (root).mkdir(parents=True, exist_ok=True)
    (root / "bin.md").write_bytes(b"# bin\n\x00\x00\xff binary garbage")
    aid = artifact_id_for(Path("bin.md"))
    with pytest.raises(ToolResultException) as exc:
        read_artifact(aid, root)
    assert exc.value.error.code == "binary_file"


def test_read_artifact_missing_root(tmp_path):
    ctx = _ctx(None)
    with pytest.raises(ToolResultException) as exc:
        handle_read_artifact(
            ToolRequest(
                name="read_artifact",
                arguments={"artifact_id": "archive:whatever"},
                call_id="c1",
            ),
            ctx,
        )
    assert exc.value.error.code == "archive_unavailable"


def test_empty_artifact_id_rejected(tmp_path):
    ctx = _ctx(tmp_path / "archive")
    with pytest.raises(ToolResultException) as exc:
        handle_read_artifact(
            ToolRequest(
                name="read_artifact",
                arguments={"artifact_id": ""},
                call_id="c1",
            ),
            ctx,
        )
    assert exc.value.error.code == "empty_artifact_id"
