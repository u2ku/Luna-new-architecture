# Luna Archive Tools

Luna's first model-callable tools: a retrieval-based archive so the model
can pull historical context on demand instead of having the whole archive
injected into every prompt.

The archive is **retrieval-based**. The runtime never loads the archive into
the prompt. When the model needs history, it calls `search_archive`, opens a
hit with `read_artifact`, and synthesises a durable note with
`create_artifact`.

| tool | access | purpose |
| --- | --- | --- |
| `search_archive` | read | ranked excerpts over Markdown in the archive root |
| `read_artifact` | read | bounded, line-numbered window of one known artifact |
| `create_artifact` | write | durable Markdown note under `LunaData/artifacts` |

## The archive root

The root is **configurable** through `LUNA_ARCHIVE_ROOT`. It is resolved in
this order:

1. `$LUNA_ARCHIVE_ROOT` (explicit override);
2. the expanded `archive.root` value in `config/tools.yaml`
   (`${LUNA_ARCHIVE_ROOT}`);
3. the **existing configured default** from `config/paths.yaml`
   (`$LUNA_DATA_ROOT/archive/wiki` → `LunaData/archive/wiki`).

A new archive location is never invented. If the resolved root does not exist,
`search_archive` returns a structured `available: false` result rather than
raising. Set the root for a real run:

```sh
export LUNA_ARCHIVE_ROOT=/Users/pieratradio/Archive/wiki
```

(The live Markdown archive — including the `project-hull/` tree — lives at
`/Users/pieratradio/Archive/wiki`. It is **not** under the obsolete
`/Users/pieratradio/Luna/` repository.)

## Shared tool protocol

`luna/tools/protocol.py` defines the minimal types every tool agrees on:

- `ToolSpec` — `name`, `description`, `input_schema`, `access`
  (`read`/`write`), `enabled`.
- `ToolContext` — archive root, artifact output root, limits, and the
  `actor` / `source` / `stream_id` / `turn_id` provenance receipts need.
- `ToolRequest` — one invocation: name, arguments, `call_id`.
- `ToolResult` — bounded outcome: `ok`, structured `content`,
  `artifact_ids`, optional `ToolError`, `duration_ms`.
- `ToolError` — stable `code` + `message` (never prose to parse).

## Registry

`luna/tools/registry.py` exposes `register` / `get` / `list` / `execute`.
`execute` validates the request against the tool's `input_schema`
(required fields + declared types; undeclared fields rejected so a caller
cannot smuggle, for example, a path into `create_artifact`), then dispatches.
A disabled or unknown tool, bad arguments, or a handler failure all return a
structured `ToolError` — `execute` never raises on bad input.

`luna/tools/executor.py`'s `build_archive_registry()` registers the three
archive tools at startup.

## search_archive

Read-only search over Markdown files **inside the archive root only**.

Input:

```json
{ "query": "Project Hull certification", "limit": 8 }
```

Optional filters: `path_prefix`, `date_from`, `date_until`.

Behaviour:

- skips `.git`, `.obsidian`, `node_modules`, `__pycache__`, manifests, and
  generated indexes (the `_`-prefixed files: `_master-index.md`,
  `_index.md`, `_compile-log.md`, …);
- searches `.md` files only;
- bounded memory — a streaming walk keeps only the top-`limit` in a heap;
- `limit` default 8, max 20 (clamped);
- returns **excerpts**, never full documents;
- never fabricates — a file is returned only if it matches a query term.

Each result carries: `artifact_id`, `title`, `relative_path`, `score`,
`matched_terms`, `excerpt`, `modified_at`.

Ranking weights, in rough order: distinct query-term coverage, title matches,
exact phrase matches, and a **capped** occurrence-density tie-breaker (per
term counts capped at 8 so a large file cannot dominate by repetition).
Large generated indexes are skipped entirely, so a 44 KB master index never
drowns authored content.

An empty query fails validation (`empty_query`). A missing root returns a
structured `available: false` result.

## read_artifact

Bounded reading of one **known** artifact.

```json
{ "artifact_id": "archive:…", "start_line": 1, "line_count": 200 }
```

`artifact_id` is an opaque hash (`archive:` + sha1 of the archive-relative
path). The model cannot supply an arbitrary path: the only way to turn an id
back into a file is `resolve_artifact_id`, which **walks the archive** and
matches — so the id must correspond to a real file inside the root.

Guarantees:

- no traversal — ids resolve only to files inside the root;
- no symlink escape — realpath containment is checked after resolution;
- no secrets — paths under a `secrets` directory are rejected;
- no binaries — non-UTF-8 / NUL-containing `.md` files are rejected;
- bounded output — `line_count` default 200, max 500.

Output: `artifact_id`, `title`, `relative_path`, `start_line`, `end_line`,
`total_lines`, `truncated`, `content` (a list of `{line, text}`), `modified_at`.

## create_artifact

Durable Markdown creation, written **only** under the configured
`artifact_output_root` (`$LUNA_DATA_ROOT/artifacts` → `LunaData/artifacts`).

```json
{
  "title": "Archive Tool Implementation Decision",
  "content": "Markdown content",
  "category": "luna-system",
  "source_event_ids": ["event-id"]
}
```

The filename is generated inside the tool — a date-prefixed slug
(`YYYY-MM-DD-<slug>-<hash6>.md`). Caller-supplied paths are rejected (and the
schema's `additionalProperties: false` makes a path field impossible).
Creation is **atomic** (write temp → `os.link`) and **never overwrites** — a
duplicate filename fails with `duplicate_artifact` rather than clobbering.

Content is stored UTF-8 with a small provenance frontmatter
(`artifact_id`, `title`, `created_at`, `category`, `source_event_ids`).

Secret/token-shaped content is **rejected** (`content_contains_secret`), not
silently persisted.

Intended for durable decisions, synthesis documents, specifications, and
milestones — not transient conversation.

## Receipts

Every tool execution appends a **paired** event to `world.jsonl`:

1. `tool_call` — written **before** dispatch (intent);
2. `tool_result` — written **after**, referencing its call by
   `call_event_id` (proof).

A tool call is intent; a result is proof. Receipts never carry complete
artifact contents: `luna/tools/receipts.py` bounds arguments to declared
fields, truncates long strings, caps list sizes, and redacts token-shaped
values. The `tool_result` payload carries an envelope on **every** call plus
a bounded per-tool digest — never `content`:

```
tool_result:
  tool, call_event_id, call_id, status,         # identity + pairing
  started_at, finished_at, duration_ms,         # when it ran
  error_code, error_message,                     # always present; None on success
  result_summary,                               # one-line human description
  affected_resources,                            # stable ids/paths touched
  artifact_ids,                                  # legacy alias of affected_resources
  receipt:                                       # per-tool digest (sanitised)
    search_archive:  query, result_count, top_results[]
                      (artifact_id, title, relative_path, score)
    read_artifact:   artifact_id, relative_path, start_line, end_line,
                      characters_returned, truncated, content_hash
    create_artifact: artifact_id, relative_path, title, category,
                      content_hash, bytes_written
```

`content_hash` is a sha256 of the bytes the model received (read) or that
were written to disk (create) — enough to prove what happened and locate the
durable result without storing it. `top_results` carries no excerpts. The
ledger holds only what's needed for proof and lookup.

## Model integration

`luna/api/routes.py`'s `ChatService` exposes the three tool schemas to the
provider (`ModelRequest.tools`) and runs a structured tool loop:

```
model requests archive tool
  → runtime validates request (registry)
  → runtime appends tool_call receipt
  → runtime executes tool (executor)
  → runtime appends tool_result receipt
  → bounded result returns to model
  → model produces final response
```

Per-turn budget (from `config/tools.yaml` → `tools.per_turn`):

- max archive tool calls: **6**
- max archive result characters: **20,000**

Only **structured** model tool calls are executed. Tools parsed from ordinary
prose or Markdown code blocks are ignored — text that merely mentions a tool
name is content, not an invocation. Once the call budget is spent, tools are
dropped from the next request so the model must answer.

## Tool transports

The tool framework is provider-agnostic; only the **transport** — how a call is
read out of a model response and how a result is fed back — differs by
provider. `luna/tools/transport.py` selects one via the provider's
`supports_native_tools` flag (`luna/models/base.py`):

| provider | flag | transport | how a call is carried |
| --- | --- | --- | --- |
| OpenAI-compatible (native function-calling) | `True` | `NativeTransport` | `tools` on the wire; model returns `tool_calls` |
| whooshd (local, no native tool calling) | `False` | `PromptJsonTransport` | tool schemas in the system prompt; model emits a ```` ```tool_call ```` sentinel block in its text |

### Prompt-JSON transport (whooshd)

whooshd accepts `tools` on the wire but does not emit/consume `tool_calls`, and
it 502s when the model returns `content: null` (the shape a tool-call attempt
produces). The prompt transport therefore sends `tools=()` — removing that
trigger — and carries the protocol in the system prompt instead.

To call a tool, the model emits a fenced block:

~~~
```tool_call
{"tool": "search_archive", "arguments": {"query": "Project Hull"}}
```
~~~

The runtime parses the model's `content` for **exactly** that sentinel —
generic ```` ```json ```` blocks and inline JSON are ordinary content, never
executed — validates the call against the registry, runs it through the same
executor (same paired receipts, same 6-call / 20k-char budget), and feeds the
bounded result back as a ```` ```tool_result ```` block in a `user` message:

~~~
```tool_result
{"ok": true, "content": {"results": [...]}}
```
~~~

One tool call per turn (first block only). If the JSON inside a block is
malformed or the shape is wrong, the runtime sends a repair message and
re-prompts (counted against the 6-call budget, so a model that keeps emitting
bad blocks cannot loop forever). If no sentinel is present, the content is the
final answer (any stray ```` ```tool_result ```` echoes are stripped).

The native transport is unchanged (`tool_calls` ↔ `tool` messages). When
whooshd gains native function-calling upstream, flipping `supports_native_tools`
to `True` switches it to the native transport with no other changes.

## Configuration

`config/tools.yaml`:

```yaml
tools:
  default_policy: deny
  enabled: [search_archive, read_artifact, create_artifact]
  per_turn:
    max_tool_calls: 6
    max_result_chars: 20000

archive:
  root: ${LUNA_ARCHIVE_ROOT}
  artifact_output_root: ${LUNA_DATA_ROOT}/artifacts
  search_default_limit: 8
  search_max_limit: 20
  read_default_lines: 200
  read_max_lines: 500
```

`${VAR}` expansion is explicit and tested (`luna/tools/config.py`):
environment wins, then a small defaults map (the resolved data root), then
the existing configured default. An unset variable expands to empty, never a
literal `${...}`.

## Running

```sh
cd /Users/pieratradio/luna-new-architecture/luna-runtime
source ../.venv/bin/activate
export LUNA_ARCHIVE_ROOT=/Users/pieratradio/Archive/wiki
luna-server            # or: python -m luna.api.server
```

Smoke test:

```sh
python scripts/archive_tools_smoke_test.py
```

It searches the real archive for *Project Hull*, opens a returned artifact,
answers using its contents, creates one test artifact, verifies the paired
receipts, and deletes only the test artifact.

## Limitations

- No write back into the archive — `create_artifact` writes to
  `LunaData/artifacts`, a separate durable-notes tree, never into the
  searchable archive root.
- Search is term-based (tokenised, case-insensitive, stopword-stripped); it
  is not semantic or embedding-based.
- The whooshd local provider does not support native function-calling; tools
  work over the **prompt-JSON transport** (a ```` ```tool_call ```` sentinel
  block in the model's text). Reliability depends on the model following the
  sentinel protocol; a repair loop handles malformed JSON / bad tool / bad
  args by feeding the error back and re-prompting. The OpenAI-compatible path
  uses native `tool_calls`.
- whooshd supports `response_format` (json_object / json_schema) — a future
  reliability lever (force valid JSON on repair turns); not used in v1.
- If a model returns `content: null` on a plain (non-tool) answer, whooshd
  502s — a whooshd/model issue outside the transport. Not sending `tools`
  removes the main trigger but cannot guarantee it.
- `read_artifact` reads at most 500 lines per call; long documents are
  paged via `start_line`.
- The secret rejecter is pattern-based; it will not catch every conceivable
  secret shape.
