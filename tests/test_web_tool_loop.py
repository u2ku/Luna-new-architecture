"""Integration tests for the web tool loop in ChatService.

Covers the full search_web → fetch_webpage → answer flow, the per-turn
web ceilings (combined-call and text-budget), the bounded tool error the
model receives when a ceiling is hit, and regression of the archive
tools and plain chat alongside the new web tools.

Uses a scripted model provider, a fake search provider, and a fake HTTP
transport — no real network.
"""

from __future__ import annotations

import json
from pathlib import Path

from luna.api.routes import ChatRequest, ChatService
from luna.ledger import WorldLedger
from luna.models.base import (
    FinishReason,
    ModelProvider,
    ModelResponse,
    ToolCall,
    Usage,
)
from luna.tools.config import (
    ArchiveConfig,
    ToolsConfig,
    WebFetchConfig,
    WebSearchConfig,
    WebTurnLimits,
)
from luna.tools.executor import build_registry
from luna.web.fetch import HttpFetchResponse, HttpTransport
from luna.web.providers.base import WebSearchProvider
from luna.web.types import ProviderSearchResult


# ---------------------------------------------------------------------------
# Fakes
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
        self.routes = routes  # url -> (status, ct, body)
        self.calls = []

    def get(self, url, *, headers, connect_timeout, read_timeout, max_bytes):
        self.calls.append(url)
        status, ct, body = self.routes.get(url, (200, "text/plain", b""))
        if isinstance(body, str):
            body = body.encode("utf-8")
        if len(body) > max_bytes:
            body = body[: max_bytes + 1]
        return HttpFetchResponse(status, {"content-type": ct}, body, url)


def _last_tool_content(messages):
    for msg in reversed(messages):
        if getattr(msg, "role", None) == "tool" and getattr(msg, "tool_call_id", None):
            try:
                return json.loads(msg.content)
            except (TypeError, ValueError):
                return None
    return None


def _archive_cfg(root, output):
    return ArchiveConfig(
        root=root, artifact_output_root=output,
        search_default_limit=8, search_max_limit=20,
        read_default_lines=200, read_max_lines=500,
    )


def _tools_cfg(max_calls=10, max_chars=200000):
    return ToolsConfig(
        enabled=["search_archive", "read_artifact", "create_artifact",
                 "search_web", "fetch_webpage"],
        max_tool_calls_per_turn=max_calls,
        max_result_chars_per_turn=max_chars,
    )


def _web_search_cfg(provider):
    return WebSearchConfig(
        provider_name="fake", default_limit=5, max_limit=10, timeout_seconds=15,
        searxng_url="", brave_api_key="", provider=provider,
    )


def _web_fetch_cfg(transport, **over):
    base = dict(
        connect_timeout_seconds=5, read_timeout_seconds=15, total_timeout_seconds=20,
        max_redirects=5, max_response_bytes=2_000_000, default_text_chars=20000,
        max_text_chars=50000, allowed_ports=(80, 443), user_agent="LunaRuntime-test",
        max_links=50, transport=transport, resolver=lambda h: ["8.8.8.8"],
    )
    base.update(over)
    return WebFetchConfig(**base)


def _web_turn_limits(**over):
    base = dict(max_search_calls=3, max_fetch_calls=5, max_combined_web_calls=8,
                max_combined_webpage_text=50_000)
    base.update(over)
    return WebTurnLimits(**base)


PAGE = (
    "<html><head><title>Drone Rules</title></head><body>"
    "<article><h1>Part 101</h1><p>RPAS operations require CAA certification.</p>"
    "<p>Maximum weight 25kg.</p></article></body></html>"
)


class _WebScriptedProvider(ModelProvider):
    """search_web → fetch_webpage(url) → answer citing the source."""

    name = "web-stub"

    def __init__(self) -> None:
        self.step = 0
        self.answer = ""

    def complete(self, request) -> ModelResponse:
        self.step += 1
        msgs = request.messages
        if self.step == 1:
            return ModelResponse(
                content="",
                tool_calls=(ToolCall(
                    id="w-search", name="search_web",
                    arguments={"query": "NZ CAA Part 101", "limit": 3},
                ),),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=Usage(), model="web-stub",
            )
        if self.step == 2:
            payload = _last_tool_content(msgs) or {}
            results = (payload.get("content") or {}).get("results") or []
            url = results[0]["url"] if results else "https://example.com/page"
            return ModelResponse(
                content="",
                tool_calls=(ToolCall(
                    id="w-fetch", name="fetch_webpage",
                    arguments={"url": url, "max_chars": 2000},
                ),),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=Usage(), model="web-stub",
            )
        # step 3: answer using the fetched text + cited source
        payload = _last_tool_content(msgs) or {}
        content = payload.get("content") or {}
        text = content.get("text", "")
        self.answer = (
            f"From {content.get('title', 'the page')}: {text[:80]} "
            f"(source: {content.get('final_url', '')})"
        )
        return ModelResponse(content=self.answer, finish_reason=FinishReason.STOP,
                              usage=Usage(), model="web-stub")


def _make_service(tmp_path, *, provider, search, fetch, turn_limits,
                  max_calls=10, max_chars=200000):
    ledger = WorldLedger(path=tmp_path / "ledger" / "world.jsonl",
                         lock_path=tmp_path / "locks" / "world.lock")
    return ChatService(
        provider=provider, ledger=ledger, system_prompt="You are Luna.",
        registry=build_registry(),
        archive_config=_archive_cfg(tmp_path / "archive", tmp_path / "artifacts"),
        tools_config=_tools_cfg(max_calls, max_chars),
        web_search_config=search, web_fetch_config=fetch, web_turn_limits=turn_limits,
    )


# ---------------------------------------------------------------------------
# Full flow
# ---------------------------------------------------------------------------


def test_web_loop_search_fetch_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    search = _FakeSearchProvider([
        ProviderSearchResult(title="Part 101", url="https://example.com/page",
                             snippet="CAA rules")
    ])
    transport = _FakeTransport({"https://example.com/page": (200, "text/html", PAGE)})
    provider = _WebScriptedProvider()
    svc = _make_service(
        tmp_path, provider=provider, search=_web_search_cfg(search),
        fetch=_web_fetch_cfg(transport), turn_limits=_web_turn_limits(),
    )
    resp = svc.complete(ChatRequest(text="What are the NZ drone rules?"))

    assert "Part 101" in resp.response
    assert "source: https://example.com/page" in resp.response
    assert resp.tool_calls == 2

    events = svc.ledger.tail(50)
    types = [e["type"] for e in events]
    # user, search call, search result, fetch call, fetch result, assistant
    assert types == ["user_message", "tool_call", "tool_result",
                     "tool_call", "tool_result", "assistant_message"]
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert [c["payload"]["tool"] for c in calls] == ["search_web", "fetch_webpage"]
    for c, r in zip(calls, results):
        assert r["payload"]["call_event_id"] == c["event_id"]
        assert r["payload"]["status"] == "ok"
    # the model received bounded fetch text, not raw HTML
    assert "<html>" not in resp.response
    # transport was actually driven by the fetch (not a hallucination)
    assert transport.calls == ["https://example.com/page"]


def test_web_loop_search_returns_no_sources_still_pairs(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    search = _FakeSearchProvider([])
    transport = _FakeTransport({})
    provider = _WebScriptedProvider()
    svc = _make_service(
        tmp_path, provider=provider, search=_web_search_cfg(search),
        fetch=_web_fetch_cfg(transport), turn_limits=_web_turn_limits(),
    )
    resp = svc.complete(ChatRequest(text="anything"))
    # step 2 found no results → fetch defaults to example.com/page, still pairs
    assert resp.tool_calls == 2


def test_search_unavailable_marked_in_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    transport = _FakeTransport({"https://example.com/page": (200, "text/html", PAGE)})
    provider = _WebScriptedProvider()
    svc = _make_service(
        tmp_path, provider=provider,
        search=_web_search_cfg(provider=None),  # no provider
        fetch=_web_fetch_cfg(transport), turn_limits=_web_turn_limits(),
    )
    svc.complete(ChatRequest(text="search the web"))
    events = svc.ledger.tail(50)
    res = next(e for e in events if e["type"] == "tool_result"
               and e["payload"]["tool"] == "search_web")
    assert res["payload"]["status"] == "ok"  # unavailable, not error
    assert res["payload"]["receipt"]["status"] == "unavailable"


# ---------------------------------------------------------------------------
# Per-turn web ceilings
# ---------------------------------------------------------------------------


class _RepeatSearchProvider(ModelProvider):
    """Keeps calling search_web until forced to answer."""

    name = "repeat"

    def __init__(self, n):
        self.n = n
        self.step = 0

    def complete(self, request) -> ModelResponse:
        self.step += 1
        if self.step > self.n:
            return ModelResponse(content="done", finish_reason=FinishReason.STOP,
                                 usage=Usage(), model="repeat")
        return ModelResponse(
            content="", tool_calls=(ToolCall(
                id=f"r{self.step}", name="search_web", arguments={"query": "q"},
            ),),
            finish_reason=FinishReason.TOOL_CALLS, usage=Usage(), model="repeat",
        )


def test_combined_web_call_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    search = _FakeSearchProvider([
        ProviderSearchResult(title="t", url="https://example.com/a", snippet="s")
    ])
    provider = _RepeatSearchProvider(n=5)
    svc = _make_service(
        tmp_path, provider=provider, search=_web_search_cfg(search),
        fetch=_web_fetch_cfg(_FakeTransport({})),
        turn_limits=_web_turn_limits(max_combined_web_calls=2),
    )
    resp = svc.complete(ChatRequest(text="go"))
    assert resp.response == "done"
    # only 2 search_web calls executed + receipted; the 3rd was refused
    calls = [e for e in svc.ledger.tail(50) if e["type"] == "tool_call"]
    assert len(calls) == 2
    assert all(c["payload"]["tool"] == "search_web" for c in calls)


class _RepeatFetchProvider(ModelProvider):
    """Keeps calling fetch_webpage until forced to answer."""

    name = "repeat-fetch"

    def __init__(self, n):
        self.n = n
        self.step = 0

    def complete(self, request) -> ModelResponse:
        self.step += 1
        if self.step > self.n:
            return ModelResponse(content="done", finish_reason=FinishReason.STOP,
                                 usage=Usage(), model="repeat-fetch")
        return ModelResponse(
            content="", tool_calls=(ToolCall(
                id=f"f{self.step}", name="fetch_webpage",
                arguments={"url": "https://example.com/page"},
            ),),
            finish_reason=FinishReason.TOOL_CALLS, usage=Usage(),
            model="repeat-fetch",
        )


def test_combined_text_budget_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    transport = _FakeTransport({"https://example.com/page": (200, "text/html", PAGE)})
    provider = _RepeatFetchProvider(n=5)
    svc = _make_service(
        tmp_path, provider=provider, search=_web_search_cfg(_FakeSearchProvider([])),
        fetch=_web_fetch_cfg(transport, default_text_chars=200, max_text_chars=200),
        turn_limits=_web_turn_limits(max_combined_webpage_text=50),
    )
    resp = svc.complete(ChatRequest(text="go"))
    assert resp.response == "done"
    # first fetch returns ~200 text chars (>100 budget); the second fetch
    # is refused. Exactly one fetch receipted.
    calls = [e for e in svc.ledger.tail(50) if e["type"] == "tool_call"]
    assert len(calls) == 1
    assert calls[0]["payload"]["tool"] == "fetch_webpage"


def test_fetch_per_type_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    transport = _FakeTransport({"https://example.com/page": (200, "text/html", PAGE)})
    provider = _RepeatFetchProvider(n=5)
    svc = _make_service(
        tmp_path, provider=provider, search=_web_search_cfg(_FakeSearchProvider([])),
        fetch=_web_fetch_cfg(transport),
        turn_limits=_web_turn_limits(max_fetch_calls=1, max_combined_web_calls=8,
                                     max_combined_webpage_text=50_000),
    )
    resp = svc.complete(ChatRequest(text="go"))
    assert resp.response == "done"
    calls = [e for e in svc.ledger.tail(50) if e["type"] == "tool_call"]
    assert len(calls) == 1  # 2nd fetch refused by per-type cap


# ---------------------------------------------------------------------------
# Regression: archive tools + plain chat still work with web tools wired
# ---------------------------------------------------------------------------


def _make_archive(tmp: Path) -> Path:
    root = tmp / "archive"
    (root / "project-hull").mkdir(parents=True, exist_ok=True)
    (root / "project-hull" / "hull-sensors.md").write_text(
        "# Hull Sensor Stack\nThe hull sensor stack for project hull.",
        encoding="utf-8",
    )
    return root


def test_archive_tools_regression(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))
    root = _make_archive(tmp_path)

    class _ArchiveProvider(ModelProvider):
        name = "a-stub"

        def __init__(self):
            self.step = 0

        def complete(self, request):
            self.step += 1
            if self.step == 1:
                return ModelResponse(
                    content="", tool_calls=(ToolCall(
                        id="a", name="search_archive", arguments={"query": "hull sensor"},
                    ),),
                    finish_reason=FinishReason.TOOL_CALLS, usage=Usage(), model="a-stub",
                )
            return ModelResponse(content="Project Hull sensor stack is documented.",
                                 finish_reason=FinishReason.STOP, usage=Usage(),
                                 model="a-stub")

    svc = _make_service(
        tmp_path, provider=_ArchiveProvider(),
        search=_web_search_cfg(_FakeSearchProvider([])),
        fetch=_web_fetch_cfg(_FakeTransport({})), turn_limits=_web_turn_limits(),
    )
    # point archive_config at the real tmp archive
    svc.archive_config = _archive_cfg(root, tmp_path / "artifacts")
    resp = svc.complete(ChatRequest(text="What is the hull sensor stack?"))
    assert "Project Hull" in resp.response
    assert resp.tool_calls == 1
    events = svc.ledger.tail(20)
    assert [e["type"] for e in events] == ["user_message", "tool_call",
                                          "tool_result", "assistant_message"]


def test_chat_regression_with_web_tools_wired(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))

    class _PlainProvider(ModelProvider):
        name = "p-stub"

        def complete(self, request):
            return ModelResponse(content="Hello back.", finish_reason=FinishReason.STOP,
                                 usage=Usage(), model="p-stub")

    svc = _make_service(
        tmp_path, provider=_PlainProvider(),
        search=_web_search_cfg(_FakeSearchProvider([])),
        fetch=_web_fetch_cfg(_FakeTransport({})), turn_limits=_web_turn_limits(),
    )
    resp = svc.complete(ChatRequest(text="hi"))
    assert resp.response == "Hello back."
    assert resp.tool_calls == 0
    assert [e["type"] for e in svc.ledger.tail(20)] == ["user_message",
                                                        "assistant_message"]


def test_web_source_directive_in_system_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_DATA_ROOT", str(tmp_path))

    class _CaptureProvider(ModelProvider):
        name = "cap"
        supports_native_tools = False  # prompt-JSON transport path

        def __init__(self):
            self.seen_system = ""

        def complete(self, request):
            self.seen_system = request.messages[0].content
            return ModelResponse(content="ok", finish_reason=FinishReason.STOP,
                                 usage=Usage(), model="cap")

    cap = _CaptureProvider()
    svc = _make_service(
        tmp_path, provider=cap, search=_web_search_cfg(_FakeSearchProvider([])),
        fetch=_web_fetch_cfg(_FakeTransport({})), turn_limits=_web_turn_limits(),
    )
    svc.complete(ChatRequest(text="hi"))
    assert "Do not claim to have searched the web" in cap.seen_system
    assert "search_web" in cap.seen_system
