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

_ENV_REF = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any, defaults: Mapping[str, str] | None = None) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` references against the env.

    Environment wins over ``defaults``; an unset variable with no
    ``:-default`` expands to the empty string (never left as a literal
    ``${...}``). A ``${VAR:-default}`` form uses ``default`` when the
    variable is unset. Non-string values are returned as their string
    form.
    """
    if not isinstance(value, str):
        return str(value)
    defaults = defaults or {}

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        fallback = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if name in defaults:
            return defaults[name]
        return fallback if fallback is not None else ""

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


# ---------------------------------------------------------------------------
# Web research tools configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebSearchConfig:
    """Search provider selection and limits (provider-neutral data only).

    ``provider`` is the *built* :class:`~luna.web.providers.WebSearchProvider`
    instance, wired at startup (None when no provider is configured so
    ``search_web`` surfaces ``available: False`` rather than crash).
    """

    provider_name: str  # "none" | "searxng" | "brave"
    default_limit: int
    max_limit: int
    timeout_seconds: float
    searxng_url: str
    brave_api_key: str
    provider: object | None = None  # WebSearchProvider | None


@dataclass(frozen=True)
class WebFetchConfig:
    """Fetch limits + collaborators. ``transport``/``resolver`` are
    injectable so tests use fakes; the live transport is wired at startup.
    """

    connect_timeout_seconds: float
    read_timeout_seconds: float
    total_timeout_seconds: float
    max_redirects: int
    max_response_bytes: int
    default_text_chars: int
    max_text_chars: int
    allowed_ports: tuple[int, ...]
    user_agent: str
    max_links: int = 50
    transport: object | None = None  # HttpTransport | None
    resolver: object | None = None  # Resolver callable | None


@dataclass(frozen=True)
class WebTurnLimits:
    """Per-turn ceilings on web-tool calls. Enforced by the turn loop."""

    max_search_calls: int
    max_fetch_calls: int
    max_combined_web_calls: int
    max_combined_webpage_text: int


@dataclass(frozen=True)
class WebConfig:
    """Resolved web research configuration (search + fetch + turn limits)."""

    search: WebSearchConfig
    fetch: WebFetchConfig
    turn_limits: WebTurnLimits


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


def load_web_config(
    repo_root: Path, *, data_root: Path | None = None
) -> tuple[WebSearchConfig, WebFetchConfig, WebTurnLimits]:
    """Load and resolve the web search + fetch config from ``config/tools.yaml``.

    Data-only: the returned configs carry no live provider/transport —
    those are wired by :func:`build_web_config` at startup. Missing
    provider configuration expands to ``provider_name="none"`` so Luna
    still starts and ``search_web`` marks itself unavailable.
    """
    tools_yaml = _mapping(_load_yaml(repo_root / "config/tools.yaml").get("web"))
    search_yaml = _mapping(tools_yaml.get("search"))
    fetch_yaml = _mapping(tools_yaml.get("fetch"))
    per_turn_yaml = _mapping(tools_yaml.get("per_turn"))

    defaults = {"LUNA_DATA_ROOT": str((data_root or resolve_data_root(repo_root)).expanduser())}

    search_config = WebSearchConfig(
        provider_name=(
            expand_env(
                search_yaml.get("provider", "${LUNA_WEB_SEARCH_PROVIDER:-none}"),
                defaults,
            ).lower()
            or "none"
        ),
        default_limit=int(search_yaml.get("default_limit", 5)),
        max_limit=int(search_yaml.get("max_limit", 10)),
        timeout_seconds=float(search_yaml.get("timeout_seconds", 15)),
        searxng_url=expand_env(
            search_yaml.get("searxng_url", "${LUNA_SEARXNG_URL}"), defaults
        ),
        brave_api_key=expand_env(
            search_yaml.get("brave_api_key", "${LUNA_BRAVE_SEARCH_API_KEY}"), defaults
        ),
    )

    ports_raw = fetch_yaml.get("allowed_ports", [80, 443])
    allowed_ports = tuple(int(p) for p in ports_raw) if ports_raw else (80, 443)

    fetch_config = WebFetchConfig(
        connect_timeout_seconds=float(fetch_yaml.get("connect_timeout_seconds", 5)),
        read_timeout_seconds=float(fetch_yaml.get("read_timeout_seconds", 15)),
        total_timeout_seconds=float(fetch_yaml.get("total_timeout_seconds", 20)),
        max_redirects=int(fetch_yaml.get("max_redirects", 5)),
        max_response_bytes=int(fetch_yaml.get("max_response_bytes", 2_000_000)),
        default_text_chars=int(fetch_yaml.get("default_text_chars", 20_000)),
        max_text_chars=int(fetch_yaml.get("max_text_chars", 50_000)),
        allowed_ports=allowed_ports,
        user_agent=str(fetch_yaml.get("user_agent", "LunaRuntime/0.1")),
        max_links=int(fetch_yaml.get("max_links", 50)),
    )

    turn_limits = WebTurnLimits(
        max_search_calls=int(per_turn_yaml.get("max_search_calls", 3)),
        max_fetch_calls=int(per_turn_yaml.get("max_fetch_calls", 5)),
        max_combined_web_calls=int(per_turn_yaml.get("max_combined_web_calls", 8)),
        max_combined_webpage_text=int(
            per_turn_yaml.get("max_combined_webpage_text", 50_000)
        ),
    )
    return search_config, fetch_config, turn_limits


def build_search_provider(search_config: WebSearchConfig) -> object | None:
    """Build the active search provider from config, or None if unconfigured.

    Importing the providers here (not at module top) keeps
    :mod:`luna.tools.config` free of a hard dependency on the network
    library until a provider is actually needed.
    """
    name = (search_config.provider_name or "none").lower()
    if name in ("", "none"):
        return None
    if name == "searxng":
        if not search_config.searxng_url:
            return None
        from luna.web.providers.searxng import SearxngSearchProvider

        return SearxngSearchProvider(
            search_config.searxng_url, timeout=search_config.timeout_seconds
        )
    if name == "brave":
        if not search_config.brave_api_key:
            return None
        from luna.web.providers.brave import BraveSearchProvider

        return BraveSearchProvider(
            search_config.brave_api_key, timeout=search_config.timeout_seconds
        )
    return None


def build_web_config(repo_root: Path, *, data_root: Path | None = None) -> WebConfig:
    """Load web config and wire the live provider + transport.

    Used by the server at startup. Tests and the smoke script construct
    :class:`WebConfig` directly with fake collaborators.
    """
    search_config, fetch_config, turn_limits = load_web_config(
        repo_root, data_root=data_root
    )
    provider = build_search_provider(search_config)

    # Build the search config with the live provider attached without
    # mutating the frozen data-only config.
    search_wired = WebSearchConfig(
        provider_name=search_config.provider_name,
        default_limit=search_config.default_limit,
        max_limit=search_config.max_limit,
        timeout_seconds=search_config.timeout_seconds,
        searxng_url=search_config.searxng_url,
        brave_api_key=search_config.brave_api_key,
        provider=provider,
    )

    from luna.web.fetch import RequestsHttpTransport, default_resolver

    fetch_wired = WebFetchConfig(
        connect_timeout_seconds=fetch_config.connect_timeout_seconds,
        read_timeout_seconds=fetch_config.read_timeout_seconds,
        total_timeout_seconds=fetch_config.total_timeout_seconds,
        max_redirects=fetch_config.max_redirects,
        max_response_bytes=fetch_config.max_response_bytes,
        default_text_chars=fetch_config.default_text_chars,
        max_text_chars=fetch_config.max_text_chars,
        allowed_ports=fetch_config.allowed_ports,
        user_agent=fetch_config.user_agent,
        max_links=fetch_config.max_links,
        transport=RequestsHttpTransport(),
        resolver=default_resolver,
    )
    return WebConfig(search=search_wired, fetch=fetch_wired, turn_limits=turn_limits)


def load_tools_config(
    repo_root: Path, *, data_root: Path | None = None
) -> tuple[ArchiveConfig, ToolsConfig]:
    """Load and resolve the archive + tools config from ``config/tools.yaml``.

    Returns only the archive + per-turn budget; web config is loaded
    separately via :func:`load_web_config` / :func:`build_web_config` so
    Luna starts even when no web provider is configured.
    """
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
