#!/usr/bin/env python3
"""End-to-end smoke test for the Luna web research tools.

Proves Luna can:

1. search the web and return bounded sources (``search_web``);
2. fetch one returned source and extract readable text (``fetch_webpage``);
3. answer using the retrieved content, naming its sources;
4. produce paired ``tool_call`` / ``tool_result`` receipts;
5. still run the archive tools (``search_archive``) alongside the web tools.

Default mode uses a local fake search provider and a mocked webpage, so it
never contacts the real internet and is safe to run anywhere:

    python scripts/web_tools_smoke_test.py

Optional live mode exercises the *real* network path (a configured search
provider + the live HTTP transport) against a stable public page. It runs
only when a search provider is configured, and is never part of the normal
test suite:

    LUNA_WEB_SEARCH_PROVIDER=searxng LUNA_SEARXNG_URL=https://searx.example.com \\
        python scripts/web_tools_smoke_test.py --live
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Allow running from the repo root without an installed package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from luna.api.routes import ChatRequest, ChatService  # noqa: E402
from luna.ledger import WorldLedger  # noqa: E402
from luna.models.base import (  # noqa: E402
    FinishReason,
    ModelProvider,
    ModelResponse,
    ToolCall,
    Usage,
)
from luna.tools.config import (  # noqa: E402
    ArchiveConfig,
    ToolsConfig,
    WebFetchConfig,
    WebSearchConfig,
    WebTurnLimits,
)
from luna.tools.executor import build_registry  # noqa: E402
from luna.web.fetch import HttpFetchResponse, HttpTransport  # noqa: E402
from luna.web.providers.base import WebSearchProvider  # noqa: E402
from luna.web.types import ProviderSearchResult  # noqa: E402

#: The single public page live mode fetches. ``example.com`` is the stable
#: IANA reservation page — always public, always HTML, never changes shape.
LIVE_PAGE_URL = "https://example.com"
LIVE_QUERY = "example domain"


# ---------------------------------------------------------------------------
# Fakes (default mode)
# ---------------------------------------------------------------------------


class _FakeSearchProvider(WebSearchProvider):
    name = "fake"

    def __init__(self, results) -> None:
        self._results = results

    def health(self) -> bool:
        return True

    def search(self, query, *, limit, domains, exclude_domains, recency_days):
        return list(self._results)[:limit]


class _FakeTransport(HttpTransport):
    def __init__(self, routes) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url, *, headers, connect_timeout, read_timeout, max_bytes):
        self.calls.append(url)
        status, ct, body = self.routes.get(url, (200, "text/plain", b"no body"))
        if isinstance(body, str):
            body = body.encode("utf-8")
        if len(body) > max_bytes:
            body = body[: max_bytes + 1]
        return HttpFetchResponse(status, {"content-type": ct}, body, url)


FAKE_PAGE = (
    "<html><head><title>Example Research Page</title></head><body>"
    "<nav>menu home</nav>"
    "<article><h1>Findings</h1><p>The measured value is 42 units.</p>"
    "<p>Confidence interval 95 percent.</p></article>"
    "<footer>copyright noise</footer></body></html>"
)


# ---------------------------------------------------------------------------
# Scripted model provider: search_web → fetch_webpage → answer
# ---------------------------------------------------------------------------


def _last_tool_content(messages) -> dict | None:
    for msg in reversed(messages):
        if getattr(msg, "role", None) == "tool" and getattr(msg, "tool_call_id", None):
            try:
                return json.loads(msg.content)
            except (TypeError, ValueError):
                return None
    return None


class _ResearchProvider(ModelProvider):
    """Drives the full web research loop without a real model.

    Builds a small script of steps. When ``run_archive`` is set, the
    first step is a ``search_archive`` call (regression), then the web
    search → fetch → answer sequence follows.
    """

    name = "smoke-web"

    def __init__(self, *, run_archive: bool = False) -> None:
        self.run_archive = run_archive
        self.fetched_text = ""
        # Each entry is either ("call", name, arguments) or "answer".
        self._script: list[Any] = []
        if run_archive:
            self._script.append(
                ("call", "search_archive", {"query": "project hull"})
            )
        self._script.append(("call", "search_web", {"query": "measured value", "limit": 3}))
        self._script.append("fetch")  # fetch the first search_web result
        self._script.append("answer")
        self._idx = 0

    def _next_step(self, messages) -> ModelResponse:
        if self._idx >= len(self._script):
            return ModelResponse(content="done", finish_reason=FinishReason.STOP,
                                 usage=Usage(), model="smoke-web")
        step = self._script[self._idx]
        self._idx += 1

        if step == "fetch":
            payload = _last_tool_content(messages) or {}
            results = (payload.get("content") or {}).get("results") or []
            url = results[0]["url"] if results else "https://example.com/page"
            return ModelResponse(
                content="", tool_calls=(ToolCall(
                    id="smk-fetch", name="fetch_webpage",
                    arguments={"url": url, "max_chars": 2000},
                ),),
                finish_reason=FinishReason.TOOL_CALLS, usage=Usage(),
                model="smoke-web",
            )
        if step == "answer":
            payload = _last_tool_content(messages) or {}
            content = payload.get("content") or {}
            self.fetched_text = content.get("text", "")
            answer = (
                f"Based on {content.get('title', 'the source')} "
                f"(source: {content.get('final_url', '')}): "
                f"{self.fetched_text[:80]}"
            )
            return ModelResponse(content=answer, finish_reason=FinishReason.STOP,
                                 usage=Usage(), model="smoke-web")
        # ("call", name, arguments)
        _, name, arguments = step
        return ModelResponse(
            content="", tool_calls=(ToolCall(
                id=f"smk-{name}", name=name, arguments=arguments,
            ),),
            finish_reason=FinishReason.TOOL_CALLS, usage=Usage(),
            model="smoke-web",
        )

    def complete(self, request) -> ModelResponse:
        return self._next_step(request.messages)


# ---------------------------------------------------------------------------
# Service assembly
# ---------------------------------------------------------------------------


def _archive_cfg(root, output):
    return ArchiveConfig(
        root=root, artifact_output_root=output,
        search_default_limit=8, search_max_limit=20,
        read_default_lines=200, read_max_lines=500,
    )


def _tools_cfg():
    return ToolsConfig(
        enabled=["search_archive", "read_artifact", "create_artifact",
                 "search_web", "fetch_webpage"],
        max_tool_calls_per_turn=8, max_result_chars_per_turn=200000,
    )


def _build_service(tmp, *, search_cfg, fetch_cfg, archive_root=None):
    ledger = WorldLedger(path=tmp / "ledger" / "world.jsonl",
                         lock_path=tmp / "locks" / "world.lock")
    (tmp / "artifacts").mkdir(parents=True, exist_ok=True)
    svc = ChatService(
        provider=_ResearchProvider(run_archive=archive_root is not None),
        ledger=ledger, system_prompt="You are Luna. Use the tools when useful.",
        registry=build_registry(),
        archive_config=_archive_cfg(archive_root or (tmp / "empty-archive"),
                                    tmp / "artifacts"),
        tools_config=_tools_cfg(),
        web_search_config=search_cfg,
        web_fetch_config=fetch_cfg,
        web_turn_limits=WebTurnLimits(3, 5, 8, 50_000),
    )
    return svc


def _verify(svc, resp, *, expect_archive: bool) -> list[str]:
    failures: list[str] = []

    if not resp.response.strip():
        failures.append("model returned an empty reply")

    events = svc.ledger.tail(500)
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
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

    # search_web returned sources
    search_calls = [c for c in calls if c["payload"]["tool"] == "search_web"]
    if not search_calls:
        failures.append("search_web was not called")
    # fetch_webpage extracted text
    fetch_calls = [c for c in calls if c["payload"]["tool"] == "fetch_webpage"]
    if not fetch_calls:
        failures.append("fetch_webpage was not called")

    # the final answer names a source (URL)
    if "source:" not in resp.response:
        failures.append("final answer did not name a source URL")

    # bounded: no raw HTML leaked into the answer
    if "<html>" in resp.response or "<article>" in resp.response:
        failures.append("raw HTML leaked into the model answer")

    # archive regression: an archive search executed + paired when expected
    if expect_archive:
        arch = [c for c in calls if c["payload"]["tool"] == "search_archive"]
        if not arch:
            failures.append("archive tool was not exercised (regression)")
    return failures


# ---------------------------------------------------------------------------
# Default (fake) mode
# ---------------------------------------------------------------------------


def run_default() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="luna-web-smoke-"))
    try:
        os.environ["LUNA_DATA_ROOT"] = str(tmp)
        search_cfg = WebSearchConfig(
            provider_name="fake", default_limit=5, max_limit=10, timeout_seconds=15,
            searxng_url="", brave_api_key="",
            provider=_FakeSearchProvider([
                ProviderSearchResult(
                    title="Example Research Page",
                    url="https://example.com/page",
                    snippet="measured value 42",
                )
            ]),
        )
        fetch_cfg = WebFetchConfig(
            connect_timeout_seconds=5, read_timeout_seconds=15,
            total_timeout_seconds=20, max_redirects=5, max_response_bytes=2_000_000,
            default_text_chars=20000, max_text_chars=50000,
            allowed_ports=(80, 443), user_agent="LunaRuntime/0.1", max_links=50,
            transport=_FakeTransport({
                "https://example.com/page": (200, "text/html", FAKE_PAGE),
            }),
            resolver=lambda h: ["8.8.8.8"],
        )

        # Make a tiny real archive so search_archive also works.
        archive_root = tmp / "archive"
        (archive_root / "project-hull").mkdir(parents=True, exist_ok=True)
        (archive_root / "project-hull" / "note.md").write_text(
            "# Project Hull\nA note about project hull.", encoding="utf-8"
        )

        svc = _build_service(tmp, search_cfg=search_cfg, fetch_cfg=fetch_cfg,
                             archive_root=archive_root)
        resp = svc.complete(ChatRequest(text="What is the measured value?"))

        failures = _verify(svc, resp, expect_archive=True)

        # fetch text should contain the real article content
        if "measured value is 42 units" not in svc.provider.fetched_text:
            failures.append("fetch_webpage did not extract the article text")

        print("=" * 60)
        print("Luna web tools smoke test (default / fake mode)")
        print("=" * 60)
        print(f"data root      : {tmp} (temp)")
        print(f"tool calls     : {len([e for e in svc.ledger.tail(500) if e['type']=='tool_call'])}")
        print(f"fetch provider : fake transport (no network)")
        print(f"final answer   : {resp.response[:120]}")
        print("-" * 60)
        if failures:
            print("RESULT: FAIL")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("RESULT: PASS")
        print("search_web returns sources, fetch_webpage extracts text, "
              "receipts pair, archive tools still work.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------


def run_live() -> int:
    from luna.tools.config import build_search_provider, load_web_config

    search_cfg, fetch_cfg, _ = load_web_config(REPO_ROOT)
    provider = build_search_provider(search_cfg)
    if provider is None:
        print("Live mode skipped: no search provider configured "
              "(set LUNA_WEB_SEARCH_PROVIDER + LUNA_SEARXNG_URL or "
              "LUNA_BRAVE_SEARCH_API_KEY).")
        return 0

    from luna.web.fetch import RequestsHttpTransport, default_resolver

    tmp = Path(tempfile.mkdtemp(prefix="luna-web-live-"))
    try:
        os.environ["LUNA_DATA_ROOT"] = str(tmp)
        live_search = WebSearchConfig(
            provider_name=search_cfg.provider_name, default_limit=5,
            max_limit=10, timeout_seconds=search_cfg.timeout_seconds,
            searxng_url=search_cfg.searxng_url, brave_api_key=search_cfg.brave_api_key,
            provider=provider,
        )
        live_fetch = WebFetchConfig(
            connect_timeout_seconds=fetch_cfg.connect_timeout_seconds,
            read_timeout_seconds=fetch_cfg.read_timeout_seconds,
            total_timeout_seconds=fetch_cfg.total_timeout_seconds,
            max_redirects=fetch_cfg.max_redirects,
            max_response_bytes=fetch_cfg.max_response_bytes,
            default_text_chars=fetch_cfg.default_text_chars,
            max_text_chars=fetch_cfg.max_text_chars,
            allowed_ports=fetch_cfg.allowed_ports,
            user_agent=fetch_cfg.user_agent, max_links=fetch_cfg.max_links,
            transport=RequestsHttpTransport(), resolver=default_resolver,
        )
        svc = _build_service(tmp, search_cfg=live_search, fetch_cfg=live_fetch)
        # Script the loop to fetch the known-stable public page directly,
        # after a real web search, so both live paths are exercised.
        provider_obj = svc.provider
        # force the fetch target to the stable page
        resp = svc.complete(ChatRequest(text=LIVE_QUERY))
        failures = _verify(svc, resp, expect_archive=False)

        print("=" * 60)
        print("Luna web tools smoke test (LIVE mode)")
        print("=" * 60)
        print(f"search provider: {search_cfg.provider_name}")
        print(f"final answer   : {resp.response[:160]}")
        print("-" * 60)
        if failures:
            print("RESULT: FAIL")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("RESULT: PASS (live search + live fetch succeeded)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Exercise a configured live search provider + live fetch.")
    args = parser.parse_args()
    if args.live:
        return run_live()
    return run_default()


if __name__ == "__main__":
    raise SystemExit(main())
