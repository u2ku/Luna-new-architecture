"""Luna HTTP server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse

from luna.api.routes import ChatService, create_api_router
from luna.ledger import WorldLedger
from luna.models import OpenAIProvider, WhooshdProvider
from luna.tools.config import build_web_config, load_tools_config
from luna.tools.executor import build_registry


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}

    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a mapping in {path}")

    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_service(repo_root: Path = REPO_ROOT) -> ChatService:
    runtime_config = _mapping(_load_yaml(repo_root / "config/runtime.yaml").get("runtime"))
    paths_config = _mapping(_load_yaml(repo_root / "config/paths.yaml").get("paths"))
    models_config = _mapping(_load_yaml(repo_root / "config/models.yaml").get("models"))
    primary = _mapping(models_config.get("primary"))

    data_root = Path(
        os.environ.get(
            "LUNA_DATA_ROOT",
            str(runtime_config.get("data_root", repo_root.parent / "LunaData")),
        )
    ).expanduser()

    # Propagate the resolved data root to the environment so env-based
    # resolvers (notably recent_events.ledger_path, used by the context
    # builder) resolve the same absolute ledger path the writer uses —
    # regardless of the server's working directory. An explicit env value
    # already set by the caller wins.
    os.environ.setdefault("LUNA_DATA_ROOT", str(data_root))

    ledger_dir = data_root / str(paths_config.get("ledger", "ledger"))
    ledger = WorldLedger(
        path=ledger_dir / "world.jsonl",
        lock_path=data_root / "locks/world.lock",
    )

    provider_name = os.environ.get(
        "LUNA_MODEL_PROVIDER",
        str(primary.get("provider", "whooshd")),
    ).lower()

    model_name = os.environ.get("LUNA_MODEL_NAME") or primary.get("name") or None
    base_url = os.environ.get("LUNA_MODEL_BASE_URL") or primary.get("base_url")
    timeout = float(primary.get("timeout_seconds", 120))

    if provider_name == "whooshd":
        kwargs: dict[str, Any] = {"timeout": timeout}
        if model_name:
            kwargs["default_model"] = model_name
        if base_url:
            kwargs["base_url"] = str(base_url)
        provider = WhooshdProvider(**kwargs)
    elif provider_name == "openai":
        kwargs = {"timeout": timeout}
        if model_name:
            kwargs["default_model"] = model_name
        if base_url:
            kwargs["base_url"] = str(base_url)
        provider = OpenAIProvider(**kwargs)
    else:
        raise RuntimeError(f"Unsupported model provider: {provider_name}")

    archive_config, tools_config = load_tools_config(repo_root, data_root=data_root)
    registry = build_registry()
    web_config = build_web_config(repo_root, data_root=data_root)

    return ChatService(
        provider=provider,
        ledger=ledger,
        system_prompt=str(
            runtime_config.get(
                "system_prompt",
                "You are Luna. This is the new runtime test. Reply directly.",
            )
        ),
        model_name=model_name,
        temperature=float(primary.get("temperature", 0.3)),
        max_tokens=int(primary.get("max_tokens", 800)),
        registry=registry,
        archive_config=archive_config,
        tools_config=tools_config,
        web_search_config=web_config.search,
        web_fetch_config=web_config.fetch,
        web_turn_limits=web_config.turn_limits,
    )


def create_app(repo_root: Path = REPO_ROOT) -> FastAPI:
    service = build_service(repo_root)
    app = FastAPI(title="Luna Runtime", version="0.1.0")
    app.include_router(create_api_router(service))

    web_root = repo_root / "web"
    index_path = web_root / "index.html"

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        if not index_path.exists():
            raise RuntimeError(f"Missing web interface: {index_path}")
        return FileResponse(index_path)

    return app


app = create_app()


def main() -> None:
    server_config = _mapping(_load_yaml(REPO_ROOT / "config/server.yaml").get("server"))
    host = os.environ.get("LUNA_SERVER_HOST", str(server_config.get("host", "0.0.0.0")))
    port = int(os.environ.get("LUNA_SERVER_PORT", server_config.get("port", 7777)))
    uvicorn.run("luna.api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
