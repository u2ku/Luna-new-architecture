"""search_archive — read-only search over the Markdown archive.

Searches within ``LUNA_ARCHIVE_ROOT`` only. Walks the archive as a
stream (bounded memory), scores each Markdown file against the query,
and returns the top ``limit`` excerpts with enough provenance to call
``read_artifact``.

Ranking prioritises, in rough order of weight:

* **distinct query-term coverage** — a document matching more distinct
  query terms outranks one that repeats a single term;
* **title matches** — terms appearing in the document's H1/stem;
* **phrase matches** — the exact query string appearing verbatim;
* **occurrence density** — a capped tie-breaker (per-term counts are
  capped so a large file cannot dominate by repetition).

Large generated indexes are skipped entirely (see
:mod:`luna.archive._common`), so a 44 KB master index never drowns out
authored content. The tool never fabricates results: a file is returned
only if it matches at least one query term.
"""

from __future__ import annotations

import heapq
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..tools.protocol import (
    ToolContext,
    ToolError,
    ToolRequest,
    ToolResult,
    ToolSpec,
)
from ..tools.registry import ToolResultException
from ._common import (
    BinaryFileError,
    iter_markdown_files,
    read_bounded,
    title_of,
)

#: Stopwords removed from query term lists so "the project hull" does
#: not score every document containing "the".
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "for", "on",
        "with", "is", "are", "be", "by", "at", "from", "this", "that",
        "as", "it", "its",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+")

#: Per-term occurrence cap. A file that repeats a term a thousand times
#: scores no more than one that repeats it ``_OCCURRENCE_CAP`` times,
#: which is what keeps large files from dominating density.
_OCCURRENCE_CAP = 8

#: Characters of excerpt returned around the first matched term.
_EXCERPT_CHARS = 320

SEARCH_ARCHIVE_SPEC = ToolSpec(
    name="search_archive",
    description=(
        "Search Luna's Markdown archive for historical context. Returns "
        "ranked excerpts with stable artifact_ids; call read_artifact to "
        "open one. Use this instead of guessing past decisions."
    ),
    input_schema={
        "type": "object",
        "required": ["query"],
        "additionalProperties": False,
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query.",
                "minLength": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 8, max 20).",
                "minimum": 1,
                "maximum": 20,
            },
            "path_prefix": {
                "type": "string",
                "description": "Restrict to files whose archive-relative path starts with this prefix.",
            },
            "date_from": {
                "type": "string",
                "description": "ISO date; only files modified on/after this date.",
            },
            "date_until": {
                "type": "string",
                "description": "ISO date; only files modified on/before this date.",
            },
        },
    },
    access="read",
    enabled=True,
)


def _tokenize(query: str) -> list[str]:
    return [
        w for w in _WORD_RE.findall(query.lower())
        if len(w) >= 2 and w not in _STOPWORDS
    ]


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(value, fmt)
        except ValueError:
            continue
        if fmt.endswith("Z"):
            return dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _excerpt(text: str, terms: list[str]) -> str:
    """A window of text around the first matched term, single-line."""
    low = text.lower()
    pos = len(text)
    for term in terms:
        idx = low.find(term)
        if idx != -1 and idx < pos:
            pos = idx
    if pos >= len(text):
        pos = 0
    start = max(0, pos - _EXCERPT_CHARS // 4)
    end = min(len(text), start + _EXCERPT_CHARS)
    snippet = text[start:end]
    snippet = snippet.replace("\n", " ").strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _score(
    content: str,
    title: str,
    query: str,
    terms: list[str],
) -> tuple[float, list[str]]:
    """Return ``(score, matched_terms)`` for one document.

    Never invents matches: ``matched_terms`` contains only terms that
    actually appear (case-insensitive) in ``content`` or ``title``.
    """
    low_body = content.lower()
    low_title = title.lower()
    phrase_present = bool(query) and query.lower() in low_body

    matched: list[str] = []
    distinct = 0
    density_sum = 0.0
    title_terms = 0
    for term in terms:
        count = low_body.count(term) + low_title.count(term)
        if count == 0:
            continue
        distinct += 1
        matched.append(term)
        density_sum += min(count, _OCCURRENCE_CAP)
        if term in low_title:
            title_terms += 1

    if distinct == 0:
        return 0.0, []

    coverage = distinct / len(terms) if terms else 0.0
    title_coverage = title_terms / len(terms) if terms else 0.0

    score = (
        100.0 * coverage
        + 40.0 * title_coverage
        + (25.0 if phrase_present else 0.0)
        + min(density_sum, 30.0)
    )
    return score, matched


def search_archive(
    query: str,
    root: Path | None,
    *,
    limit: int = 8,
    default_limit: int = 8,
    max_limit: int = 20,
    path_prefix: str | None = None,
    date_from: str | None = None,
    date_until: str | None = None,
) -> dict[str, Any]:
    """Pure search function (no receipts). Returns a structured result.

    A missing/unconfigured archive root yields a structured
    ``available: False`` result rather than raising.
    """
    if root is None or not root.exists() or not root.is_dir():
        return {
            "available": False,
            "reason": "archive root not configured or missing",
            "query": query,
            "results": [],
            "count": 0,
        }

    if not query or not query.strip():
        # The handler converts this to a structured error; keep the
        # pure function honest by signalling it.
        return {"available": False, "reason": "empty_query", "query": query, "results": [], "count": 0}

    effective_limit = max(1, min(limit if limit else default_limit, max_limit))
    terms = _tokenize(query)
    if not terms:
        return {
            "available": True,
            "reason": "no searchable terms in query",
            "query": query,
            "results": [],
            "count": 0,
        }

    from ._common import artifact_id_for

    dfrom = _parse_date(date_from)
    duntil = _parse_date(date_until)
    prefix = path_prefix.strip() if path_prefix else ""

    # A heap of size `effective_limit` keeps memory bounded regardless of
    # archive size. We store (-score, seq, result_dict) so the smallest
    # score is popped first.
    heap: list[tuple[float, int, dict[str, Any]]] = []
    seq = 0
    for full, rel in iter_markdown_files(root):
        rel_posix = rel.as_posix()
        if prefix and not rel_posix.startswith(prefix):
            continue
        try:
            text = read_bounded(full)
        except BinaryFileError:
            continue
        except OSError:
            continue
        try:
            mtime_ts = full.stat().st_mtime
        except OSError:
            mtime_ts = 0.0
        modified_at = (
            datetime.fromtimestamp(mtime_ts, tz=timezone.utc)
            .date()
            .isoformat()
            if mtime_ts
            else None
        )
        if dfrom or duntil:
            mt = datetime.fromtimestamp(mtime_ts, tz=timezone.utc) if mtime_ts else None
            if dfrom and (mt is None or mt < dfrom):
                continue
            if duntil and (mt is None or mt > duntil):
                continue
        title = title_of(text, rel)
        score, matched = _score(text, title, query, terms)
        if score <= 0.0:
            continue
        artifact_id = artifact_id_for(rel)
        result = {
            "artifact_id": artifact_id,
            "title": title,
            "relative_path": rel_posix,
            "score": round(score, 2),
            "matched_terms": matched,
            "excerpt": _excerpt(text, matched),
            "modified_at": modified_at,
        }
        seq += 1
        # Min-heap top-K of size `effective_limit` keeping the K highest
        # scores. heap[0] is the smallest score among those kept; a new
        # higher score replaces it. seq precedes the dict so heapq never
        # has to compare dicts (which would raise TypeError on ties).
        item = (score, seq, result)
        if len(heap) < effective_limit:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)

    # Sort by score desc; seq keeps ties deterministic across runs.
    results = [
        entry[2] for entry in sorted(heap, key=lambda e: (-e[0], e[1]))
    ]
    return {
        "available": True,
        "query": query,
        "results": results,
        "count": len(results),
    }


def handle_search_archive(
    request: ToolRequest, context: ToolContext
) -> ToolResult:
    args = request.arguments
    query = str(args.get("query") or "")
    if not query.strip():
        raise ToolResultException(
            ToolError(code="empty_query", message="query must not be empty")
        )
    limit = args.get("limit")
    content = search_archive(
        query=query,
        root=context.archive_root,
        limit=int(limit) if isinstance(limit, int) else None,
        default_limit=context.search_default_limit,
        max_limit=context.search_max_limit,
        path_prefix=args.get("path_prefix"),
        date_from=args.get("date_from"),
        date_until=args.get("date_until"),
    )
    artifact_ids = tuple(r["artifact_id"] for r in content.get("results", []))
    results = content.get("results", [])
    result_count = content.get("count", len(results))
    available = content.get("available", True)
    # Top results are bounded and carry no excerpt/matched_terms — only
    # enough to prove ranking and to let read_artifact locate the file.
    top_results = [
        {
            "artifact_id": r["artifact_id"],
            "title": r["title"],
            "relative_path": r["relative_path"],
            "score": r["score"],
        }
        for r in results[:5]
    ]
    summary = (
        "archive unavailable"
        if not available
        else f"search {query!r} → {result_count} result(s)"
    )
    # An unavailable root is not a failure — surface it as a successful
    # but empty result so the model can react gracefully.
    return ToolResult(
        call_id=request.call_id,
        name="search_archive",
        ok=True,
        content=content,
        artifact_ids=artifact_ids,
        receipt={
            "result_summary": summary,
            "affected_resources": list(artifact_ids),
            "query": query,
            "result_count": result_count,
            "top_results": top_results,
        },
    )
