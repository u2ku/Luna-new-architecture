"""Tests for search_archive: relevance, ranking, suppression, limits, validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from luna.archive.search import handle_search_archive, search_archive
from luna.archive._common import artifact_id_for, iter_markdown_files
from luna.tools.protocol import ToolContext, ToolRequest


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


def _make_archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    _write(
        root / "project-hull" / "hull-sensors.md",
        "# Hull Sensor Stack\n\nThe hull sensor stack is the biometric input layer "
        "for project hull. Certification requires each sensor validated.",
    )
    _write(
        root / "project-hull" / "certification.md",
        "# Certification Plan\n\nProject hull certification steps for the hull "
        "sensor stack.",
    )
    _write(
        root / "ocean" / "boat.md",
        "# Boat\n\nA boat has a hull. Hull certification for maritime use.",
    )
    # A large generated index that mentions the terms many times — must
    # never appear in results.
    _write(
        root / "_master-index.md",
        "# Index\n\n" + ("hull sensor certification project " * 2000),
    )
    _write(root / "relational" / "_index.md", "hull hull hull hull hull\n")
    _write(root / ".obsidian" / "app.json", "{}")
    _write(root / "__pycache__" / "x.pyc", "hull")
    return root


def test_search_relevance_returns_project_hull(tmp_path):
    root = _make_archive(tmp_path)
    res = search_archive("hull sensor certification", root, limit=8)
    assert res["available"] is True
    assert res["count"] >= 2
    paths = [r["relative_path"] for r in res["results"]]
    assert "project-hull/hull-sensors.md" in paths
    assert "project-hull/certification.md" in paths


def test_title_boosting(tmp_path):
    root = tmp_path / "archive"
    # title file: term only in title
    _write(
        root / "a.md",
        "# Hull Certification\n\nUnrelated body text about apples and oranges.",
    )
    # body file: term only in body, not title, but more occurrences
    _write(
        root / "b.md",
        "# Other Title\n\nhull hull hull hull hull certification certification.",
    )
    res = search_archive("hull certification", root, limit=2)
    assert res["results"][0]["relative_path"] == "a.md"


def test_term_coverage_beats_density(tmp_path):
    root = tmp_path / "archive"
    # matches all 3 distinct terms once
    _write(root / "coverage.md", "# Doc\n\nalpha beta gamma.")
    # matches only ONE term, but many times
    _write(root / "density.md", "# Doc2\n\n" + ("alpha " * 50))
    res = search_archive("alpha beta gamma", root, limit=2)
    assert res["results"][0]["relative_path"] == "coverage.md"


def test_large_index_file_suppressed(tmp_path):
    root = _make_archive(tmp_path)
    res = search_archive("hull sensor certification project", root, limit=20)
    paths = [r["relative_path"] for r in res["results"]]
    assert "_master-index.md" not in paths
    assert not any(p.startswith("_") for p in paths)


def test_limit_enforcement(tmp_path):
    root = tmp_path / "archive"
    for i in range(12):
        _write(root / f"f{i}.md", f"# Doc {i}\n\ncommonterm content {i}")
    res = search_archive("commonterm", root, limit=5)
    assert len(res["results"]) == 5
    # default limit
    res_default = search_archive("commonterm", root)
    assert len(res_default["results"]) == 8
    # over max clamps to 20
    res_over = search_archive("commonterm", root, limit=100)
    assert len(res_over["results"]) == 12  # only 12 exist, all <=20


def test_empty_query_rejected(tmp_path):
    root = _make_archive(tmp_path)
    ctx = _ctx(root)
    with pytest.raises(Exception) as exc:
        handle_search_archive(ToolRequest(name="search_archive", arguments={"query": "   "}, call_id="c1"), ctx)
    assert exc.value.error.code == "empty_query"


def test_missing_archive_root_returns_unavailable(tmp_path):
    res = search_archive("anything", None, limit=8)
    assert res["available"] is False
    assert res["results"] == []
    # also for a non-existent path
    res2 = search_archive("anything", tmp_path / "nope", limit=8)
    assert res2["available"] is False


def test_stable_artifact_ids(tmp_path):
    root = _make_archive(tmp_path)
    res1 = search_archive("hull sensor certification", root, limit=8)
    res2 = search_archive("hull sensor certification", root, limit=8)
    ids1 = [r["artifact_id"] for r in res1["results"]]
    ids2 = [r["artifact_id"] for r in res2["results"]]
    assert ids1 == ids2
    # ids are opaque hashes, not paths
    assert all(i.startswith("archive:") for i in ids1)
    assert all("/" not in i for i in ids1)


def test_path_prefix_filter(tmp_path):
    root = _make_archive(tmp_path)
    res = search_archive(
        "hull", root, limit=8, path_prefix="project-hull/"
    )
    assert res["count"] >= 1
    assert all(
        r["relative_path"].startswith("project-hull/")
        for r in res["results"]
    )


def test_results_have_provenance_for_read(tmp_path):
    root = _make_archive(tmp_path)
    res = search_archive("hull sensor certification", root, limit=3)
    for r in res["results"]:
        assert r["artifact_id"]
        assert r["title"]
        assert r["relative_path"]
        assert isinstance(r["score"], float)
        assert r["matched_terms"]
        assert r["excerpt"]
        # artifact_id must resolve to a real file in the archive
        assert any(
            artifact_id_for(rel) == r["artifact_id"]
            for _, rel in iter_markdown_files(root)
        )
