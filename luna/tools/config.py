"""Configuration loading for the archive tools.

Reads ``config/tools.yaml`` (the ``archive`` and ``tools`` blocks) and
expands ``${VAR}`` references against the environment. Expansion is
explicit and tested: a literal ``${LUNA_ARCHIVE_ROOT}`` is resolved from
``os.environ`` first, then from a small ``defaults`` map (carrying the
resolved data root), and finally from the existing configured default
in ``config/paths.yaml`` — never invented.

The archive root is resolved in this priority order:

1. ``$LUNA_ARCHIVE_ROOT`` (explicit override — what the smoke test sets);
2. the expanded ``archive.root`` value from ``config/tools.yaml``;
3. the existing configured default ``$LUNA_DATA_ROOT / paths.archive``
   (``LunaData/archive/wiki``) — only used when it already exists in
   the repository.

A root that resolves to a non-existent directory is kept as-is;
``search_archive`` then returns a structured ``available: False`` result
rather than raising.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

_ENV_REF = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def expand_env(value: Any, defaults: Mapping[str, str] | None = None) -> str:
    """Expand ``${VAR}`` references against the environment.

    Environment wins over ``defaults``; an unset variable with no
    default expands to the empty string (never left as a literal
    ``${...}``). Non-string values are returned as their string form.
    """
    if not isinstance(value, str):
        return str(value)
    defaults = defaults or {}

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in os.environ:
            return os.environ[name]
        return defaults.get(name, "")

    return _ENV_REF.sub(repl, value)


@dataclass(frozen=True)
class ArchiveConfig:
    """Resolved archive configuration.

    ``root`` is the directory ``search_archive``/``read_artifact`` read
    from; it may not exist on disk (see :attr:`available`).
    """

    root: Path | None
    artifact_output_root: Path
    search_default_limit: int
    search_max_limit: int
    read_default_lines: int
    read_max_lines: int

    @property
    def available(self) -> bool:
        return self.root is not None and self.root.is_dir()


@dataclass(frozen=True)
class ToolsConfig:
    """Per-turn tool budget and the enabled tool list."""

    enabled: list[str]
    max_tool_calls_per_turn: int
    max_result_chars_per_turn: int


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    return value if isinstance(value, dict) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def resolve_data_root(repo_root: Path) -> Path:
    """Resolve ``$LUNA_DATA_ROOT`` with the runtime fallback."""
    runtime = _mapping(_load_yaml(repo_root / "config/runtime.yaml").get("runtime"))
    raw = os.environ.get(
        "LUNA_DATA_ROOT",
        str(runtime.get("data_root", repo_root.parent / "LunaData")),
    )
    return Path(raw).expanduser()


def load_tools_config(
    repo_root: Path, *, data_root: Path | None = None
) -> tuple[ArchiveConfig, ToolsConfig]:
    """Load and resolve the archive + tools config from ``config/tools.yaml``."""
    tools_yaml = _mapping(_load_yaml(repo_root / "config/tools.yaml").get("tools"))
    archive_yaml = _mapping(_load_yaml(repo_root / "config/tools.yaml").get("archive"))
    paths_yaml = _mapping(_load_yaml(repo_root / "config/paths.yaml").get("paths"))

    dr = (data_root or resolve_data_root(repo_root)).expanduser()
    # Defaults passed to expansion: the data root is the one value a
    # ${LUNA_DATA_ROOT} reference should fall back to when the env var
    # is unset.
    defaults = {"LUNA_DATA_ROOT": str(dr)}

    # --- archive root ---------------------------------------------------
    root_str = os.environ.get("LUNA_ARCHIVE_ROOT") or ""
    if not root_str:
        expanded = expand_env(archive_yaml.get("root", "${LUNA_ARCHIVE_ROOT}"), defaults)
        root_str = expanded
    if not root_str:
        # Fall back to the existing configured default in paths.yaml —
        # never invent a new location.
        arch_rel = paths_yaml.get("archive", "archive/wiki")
        root_str = str(dr / arch_rel)
    root = Path(root_str).expanduser()

    # --- artifact output root ------------------------------------------
    out_raw = archive_yaml.get("artifact_output_root", "${LUNA_DATA_ROOT}/artifacts")
    out_str = expand_env(out_raw, defaults)
    if not out_str:
        out_str = str(dr / "artifacts")
    artifact_output_root = Path(out_str).expanduser()

    archive_config = ArchiveConfig(
        root=root,
        artifact_output_root=artifact_output_root,
        search_default_limit=int(archive_yaml.get("search_default_limit", 8)),
        search_max_limit=int(archive_yaml.get("search_max_limit", 20)),
        read_default_lines=int(archive_yaml.get("read_default_lines", 200)),
        read_max_lines=int(archive_yaml.get("read_max_lines", 500)),
    )

    per_turn = _mapping(tools_yaml.get("per_turn"))
    tools_config = ToolsConfig(
        enabled=list(tools_yaml.get("enabled", [])),
        max_tool_calls_per_turn=int(per_turn.get("max_tool_calls", 6)),
        max_result_chars_per_turn=int(per_turn.get("max_result_chars", 20000)),
    )
    return archive_config, tools_config
