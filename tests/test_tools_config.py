"""Tests for config loading and explicit ${VAR} expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from luna.tools.config import expand_env, load_tools_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_expand_env_uses_environment(monkeypatch):
    monkeypatch.setenv("LUNA_ARCHIVE_ROOT", "/explicit/root")
    assert expand_env("${LUNA_ARCHIVE_ROOT}", {}) == "/explicit/root"


def test_expand_env_uses_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("LUNA_DATA_ROOT", raising=False)
    assert expand_env("${LUNA_DATA_ROOT}/artifacts", {"LUNA_DATA_ROOT": "/dr"}) == "/dr/artifacts"


def test_expand_env_unset_no_default_is_empty(monkeypatch):
    monkeypatch.delenv("LUNA_NEVER_SET", raising=False)
    assert expand_env("${LUNA_NEVER_SET}", {}) == ""


def test_expand_env_non_string():
    assert expand_env(123, {}) == "123"


def test_load_config_archive_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LUNA_ARCHIVE_ROOT", str(tmp_path / "wiki"))
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path / "data"))
    ac, tc = load_tools_config(REPO_ROOT, data_root=tmp_path / "data")
    assert ac.root == (tmp_path / "wiki").resolve()
    assert ac.artifact_output_root == (tmp_path / "data" / "artifacts").resolve()
    assert ac.search_default_limit == 8
    assert ac.search_max_limit == 20
    assert ac.read_default_lines == 200
    assert ac.read_max_lines == 500
    assert tc.enabled == ["search_archive", "read_artifact", "create_artifact"]
    assert tc.max_tool_calls_per_turn == 6
    assert tc.max_result_chars_per_turn == 20000


def test_load_config_falls_back_to_existing_default(monkeypatch, tmp_path):
    # No LUNA_ARCHIVE_ROOT: must fall back to <data_root>/<paths.archive>,
    # the existing configured default — never an invented location.
    monkeypatch.delenv("LUNA_ARCHIVE_ROOT", raising=False)
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path / "data"))
    ac, _ = load_tools_config(REPO_ROOT, data_root=tmp_path / "data")
    assert ac.root == (tmp_path / "data" / "archive" / "wiki").resolve()
