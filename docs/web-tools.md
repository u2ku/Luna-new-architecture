# Luna Web Research Tools

Luna's second family of model-callable tools: two **read-only** public-web
research tools that extend the existing tool framework (registry, executor,
receipts, per-turn budget) — not a second framework.

* `search_web` — search the public web and return bounded source records.
* `fetch_webpage` — fetch one public webpage and extract readable text.

Both produce the same paired `tool_call` / `tool_result` receipts in
`world.jsonl` as the archive tools. Neither executes JavaScript, submits
forms, authenticates, downloads binaries, or takes screenshots.

## Tool flow

```
model calls search_web
  → Gate validates request (registry schema + handler)
  → tool_call receipt appended
  → search executes (provider-neutral, normalised, de-duplicated)
  → tool_result receipt appended
  → bounded search results return to model
model calls fetch_webpage on a useful source
  → URL validated (and re-validated after every redirect)
  → bounded page text returns to model
model answers, naming the sources it relied upon
```

Tools are only ever invoked through structured tool calls (native
function-calling or the prompt-JSON ` ```tool_call ` sentinel). Calls are
never parsed from prose or ordinary Markdown.

## `search_web`

**Input**

| field            | type             | required | notes                                  |
|------------------|------------------|----------|----------------------------------------|
| `query`          | string           | yes      | 1–500 characters                        |
| `limit`          | integer          | no       | default 5, max 10                       |
| `domains`        | array of strings | no       | include filter, max 20, bare hostnames |
| `exclude_domains`| array of strings | no       | exclude filter, max 20                  |
| `recency_days`   | integer or null  | no       | results newer than N days              |

Domain filters must be bare hostnames (`aviation.govt.nz`). A filter with a
scheme, path, port, or credentials is rejected.

**Output**

```json
{
  "available": true,
  "query": "...",
  "provider": "searxng",
  "result_count": 2,
  "retrieved_at": "2026-07-22T22:00:00+00:00",
  "results": [
    {
      "result_id": "web:1a2b3c4d5e6f7a8b",
      "rank": 1,
      "title": "Page title",
      "url": "https://example.org/page",
      "display_domain": "example.org",
      "snippet": "Bounded search excerpt",
      "published_at": null
    }
  ]
}
```

Rules:

* URLs are normalised: host lower-cased, well-known tracking parameters
  (`utm_*`, `fbclid`, `gclid`, `msclkid`, …) stripped.
* Results are de-duplicated by canonical URL (scheme + host + path +
  remaining sorted query).
* Stable `result_id` = `web:` + `sha1(canonical)[:16]`.
* Snippets are bounded. The tool never invents a title, URL, date, or
  snippet.
* Provider ranking is preserved unless de-duplication removes an entry.
* An empty result set is a **success** with `result_count: 0`.
* When no provider is configured, the result is `available: false`
  (`reason: "no_provider"`) — not a failure, and Luna still starts.
* When a configured provider fails, the result is a `failed` tool error
  (`error.code: "provider_failed"`).
* Provider-specific payloads never escape into the model-facing result.

### Search providers

`WebSearchProvider` (in `luna/web/providers/base.py`) is the interface every
provider implements:

```python
class WebSearchProvider(Protocol):
    name: str
    def search(self, query, *, limit, domains, exclude_domains, recency_days) -> list[ProviderSearchResult]
    def health(self) -> bool
```

Two providers ship:

* **SearXNG** (`searxng`) — self-hostable, JSON API, no key. Recommended.
  Endpoint: `GET <LUNA_SEARXNG_URL>/search?format=json`. Domain filters
  are applied as `site:`/`-site:` operators.
* **Brave** (`brave`) — `GET https://api.search.brave.com/res/v1/web/search`,
  authenticated with `X-Subscription-Token: <LUNA_BRAVE_SEARCH_API_KEY>`.

The active provider is chosen by config, never hard-coded.

## `fetch_webpage`

**Input**

| field          | type    | required | notes                                  |
|----------------|---------|----------|----------------------------------------|
| `url`          | string  | yes      | public http(s) URL                     |
| `max_chars`    | integer | no       | extracted text ceiling (default 20000, max 50000) |
| `include_links`| boolean | no       | default false                          |

**Output**

```json
{
  "requested_url": "https://example.org/page",
  "final_url": "https://example.org/page",
  "title": "Page title",
  "content_type": "text/html",
  "status_code": 200,
  "retrieved_at": "2026-07-22T22:00:00+00:00",
  "text": "Extracted readable text",
  "text_chars": 12345,
  "content_hash": "sha256...",
  "truncated": false,
  "links": []
}
```

Supported content types: `text/html`, `text/plain`, `application/json`
(JSON is pretty-printed into bounded readable text). PDFs, images, audio,
video, archives, and binaries return a structured `unsupported_content`
error. PDF parsing is intentionally not supported in this task.

### HTML extraction

The extractor (stdlib `html.parser`, no extra dependency) drops `script`,
`style`, `noscript`, `svg`, `canvas`, `template`, `form` and identifiable
navigation/ad containers (`<nav>`, `<aside>`, `<footer>`, and class/id
substrings like `nav`, `menu`, `sidebar`, `ad`, `promo`, `cookie`, …). It
keeps headings and paragraph boundaries and collapses repeated whitespace.
It prefers the main article/document content and falls back to cleaned body
text. Raw HTML is never returned to the model. Charset is taken from the
`Content-Type` header, an HTML `<meta charset>`, or UTF-8 with replacement.

### Network safety

Every URL — initial and after each redirect — is validated before a
connection opens:

* only `http` / `https` (rejects `file`, `data`, `javascript`, `ftp`,
  `gopher`, `smb`, `mailto`);
* no usernames or passwords in the URL;
* no `localhost` or `.local` hostnames;
* the resolved destination must not be loopback, private, link-local,
  multicast, reserved, unspecified, or an IPv4-mapped private IPv6;
* explicit ports must be in the allowlist (80/443 by default; extra ports
  configurable but denied by default);
* DNS is the one network touch-point needed before connect; it is injected
  so tests mock it without contacting the real internet.

Redirects are followed manually (max 5, configurable) so each `Location`
is re-validated — a redirect into a private address is caught at the
boundary. The response body is streamed and the read stops at the
configured byte ceiling (default 2 MB). Luna sends only a fixed
`User-Agent` and `Accept`/`Accept-Encoding: identity` — never its cookies,
OAuth credentials, email tokens, browser state, or environment variables.
JavaScript is never executed.

## Configuration

`config/tools.yaml` (the `web` block):

```yaml
web:
  search:
    provider: ${LUNA_WEB_SEARCH_PROVIDER:-none}
    default_limit: 5
    max_limit: 10
    timeout_seconds: 15
    searxng_url: ${LUNA_SEARXNG_URL}
    brave_api_key: ${LUNA_BRAVE_SEARCH_API_KEY}

  fetch:
    connect_timeout_seconds: 5
    read_timeout_seconds: 15
    total_timeout_seconds: 20
    max_redirects: 5
    max_response_bytes: 2000000
    default_text_chars: 20000
    max_text_chars: 50000
    allowed_ports: [80, 443]
    user_agent: LunaRuntime/0.1
    max_links: 50

  per_turn:
    max_search_calls: 3
    max_fetch_calls: 5
    max_combined_web_calls: 8
    max_combined_webpage_text: 50000

tools:
  enabled:
    - search_archive
    - read_artifact
    - create_artifact
    - search_web
    - fetch_webpage
```

Environment expansion reuses the archive config loader's `${VAR}` /
`${VAR:-default}` mechanism (no second config loader). Suggested
environment variables:

| variable                          | purpose                                     |
|-----------------------------------|---------------------------------------------|
| `LUNA_WEB_SEARCH_PROVIDER`        | `none` (default) / `searxng` / `brave`      |
| `LUNA_SEARXNG_URL`                 | base URL of the SearXNG instance            |
| `LUNA_BRAVE_SEARCH_API_KEY`        | Brave Search API key                        |

Missing provider configuration does **not** prevent Luna from starting;
`search_web` reports `available: false` and the fetch tool still works.

### Per-turn limits

The general budget (`tools.per_turn.max_tool_calls`, default 6) bounds all
tool calls. On top of it, web tools have their own ceilings
(`web.per_turn`):

* max `search_web` calls per turn: 3
* max `fetch_webpage` calls per turn: 5
* max combined web calls per turn: 8
* max combined webpage text per turn: 50 000 characters

When a ceiling is reached, the runtime returns a bounded
`web_limit_exceeded` tool error to the model (not a receipted execution) and
stops the loop cleanly so the model answers. The model cannot raise any
limit beyond the configured maximum.

## Receipts

Web tools attach a bounded receipt summary (via `ToolResult.receipt`) so
the `tool_result` event records generic execution facts without persisting
full snippets or page text:

* `search_web`: `query`, `provider`, `result_count`, `source_domains`,
  `result_domains`, `retrieved_at`, `duration_ms`, `status`.
* `fetch_webpage`: `requested_url`, `final_url`, `status_code`,
  `content_type`, `bytes_received`, `text_length`, `content_hash`,
  `retrieved_at`, `duration_ms`, `status`.

API keys, request headers, cookies, and provider payloads never reach the
ledger. The event spine stays `tool_call` / `tool_result`; no web-specific
event types were introduced.

## Source handling

* The final model answer should name the source URLs it relied upon.
* The model is instructed (system prompt) not to claim it searched the web
  unless a successful `tool_result` was returned.
* Web results are treated as unverified external sources, not trusted
  internal state.
* Web results are not automatically written into the archive. Luna may use
  `create_artifact` to preserve a deliberate research synthesis.

## Files

| path                                | role                                              |
|-------------------------------------|---------------------------------------------------|
| `luna/web/__init__.py`              | package surface                                   |
| `luna/web/types.py`                | provider-neutral result types                     |
| `luna/web/security.py`             | URL validation, address classification, DNS hook  |
| `luna/web/fetch.py`                 | HTTP transport, redirect loop, HTML extraction    |
| `luna/web/providers/base.py`       | `WebSearchProvider` interface                     |
| `luna/web/providers/searxng.py`     | SearXNG provider                                  |
| `luna/web/providers/brave.py`       | Brave provider                                    |
| `luna/tools/web_tools.py`          | `search_web` + `fetch_webpage` specs + handlers  |
| `luna/tools/executor.py`           | `build_registry()`, web-tool registration         |
| `luna/tools/config.py`             | `WebConfig`, provider/transport wiring            |
| `config/tools.yaml`                | `web` block + enabled list                        |
| `scripts/web_tools_smoke_test.py`  | default (fake) + `--live` smoke test              |

## Tests

`pytest -q` covers: URL validation, scheme/userinfo/localhost/private/
link-local/IPv6 rejection, redirect re-validation and counting, byte/timeout
limits, unsupported content, HTML/plain/JSON extraction, truncation,
malformed HTML, charset handling, provider unavailable/failed, result
normalisation, de-duplication, stable ids, receipt pairing and sanitisation,
the model tool loop, combined web-call and text-budget limits, and
archive-tool + chat regression. All tests use fake providers and mocked
HTTP/DNS — none contact the real internet.

## Smoke test

```sh
# default (fake provider + mocked webpage, no network)
python scripts/web_tools_smoke_test.py

# live (only when a provider is configured; never part of the test suite)
LUNA_WEB_SEARCH_PROVIDER=searxng LUNA_SEARXNG_URL=https://searx.example.com \
    python scripts/web_tools_smoke_test.py --live
```
