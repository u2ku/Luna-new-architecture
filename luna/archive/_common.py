"""Shared helpers for the archive tools.

Everything the three archive tools agree on lives here so the rules are
stated once:

* what counts as a searchable Markdown file (and what is skipped);
* how a stable ``artifact_id`` is derived from a file's archive-relative
  path, and how it is resolved back to a file (by walking the archive —
  never by trusting a caller-supplied path);
* the path-containment and symlink-escape checks both ``search_archive``
  and ``read_artifact`` rely on.

No tool reads or writes through a caller-supplied path. Identifiers are
opaque hashes; the only way to turn one back into a file is to walk the
archive root and match, which guarantees the file is real and inside the
root.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator

# Directories that are never descended into. These are build/IDE/tooling
# noise, never archive content.
SKIP_DIRS: frozenset[str] = frozenset(
    {".git", ".obsidian", "node_modules", "__pycache__", ".pytest_cache", ".venv"}
)

# File-name prefixes that mark generated indexes / manifests / compile
# logs rather than authored content. In the Luna archive these are all
# underscore-prefixed (``_master-index.md``, ``_index.md``, ``_compile-log.md``,
# ``_tag-taxonomy.md``). A 44KB master index would otherwise dominate
# every search; skipping it keeps results authored.
_GENERATED_PREFIXES: tuple[str, ...] = ("_",)

#: Max bytes read from any one file when scoring or excerpting. The
#: archive's largest content file is well under this; the bound is a
#: defence against a future huge file blowing memory.
MAX_FILE_BYTES = 256 * 1024

# Subdirectories whose contents must never be returned, even if they
# happened to live under the archive root. ``secrets`` is the hard one.
FORBIDDEN_NAME_PARTS: frozenset[str] = frozenset({"secrets"})


def _is_generated(name: str) -> bool:
    return any(name.startswith(p) for p in _GENERATED_PREFIXES)


def is_markdown_file(path: Path) -> bool:
    """True for ``.md`` files (case-insensitive)."""
    return path.suffix.lower() == ".md"


def should_skip_name(name: str) -> bool:
    """True for generated-index / manifest / log basenames."""
    if _is_generated(name):
        return True
    low = name.lower()
    return low.startswith("manifest") or low in {"_index.md"}


def archive_root_realpath(root: Path) -> Path:
    """Resolve the archive root, following symlinks at the root itself."""
    return root.resolve(strict=False)


def within_root(child: Path, root_real: Path) -> bool:
    """True only if ``child``'s real path is inside ``root_real``.

    Uses realpath on both sides so a symlink that escapes the archive is
    caught regardless of how its string path was constructed.
    """
    try:
        child_real = child.resolve(strict=False)
        root_real_abs = root_real.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    try:
        child_real.relative_to(root_real_abs)
        return True
    except ValueError:
        return False


def path_is_forbidden(rel_path: Path) -> bool:
    """True if any path component is a forbidden name (e.g. ``secrets``)."""
    return any(part in FORBIDDEN_NAME_PARTS for part in rel_path.parts)


def artifact_id_for(relative_path: Path) -> str:
    """Derive a stable opaque id from an archive-relative path.

    The id is ``"archive:" + sha1(relative_path)[:16]``. It is stable
    across runs and processes (same path → same id) and gives the model
    no way to forge a path: the only resolution path is
    :func:`resolve_artifact_id`, which walks the real archive.
    """
    digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()
    return "archive:" + digest[:16]


def iter_markdown_files(root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield ``(absolute_path, relative_path)`` for each kept Markdown file.

    Streaming generator: the whole archive is never held in memory.
    Skips :data:`SKIP_DIRS`, generated indexes/manifests, non-Markdown
    files, and anything whose real path escapes ``root`` (symlink
    escape). Follows directory symlinks only when they stay inside the
    root.
    """
    if not root.exists() or not root.is_dir():
        return
    root_real = archive_root_realpath(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        # Prune skip dirs in place so os.walk does not descend.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        current = Path(dirpath)
        # Defensive: a symlinked dir that escaped the root would let
        # os.walk follow it; prune any real-resolved-escape dir.
        try:
            current.resolve(strict=False).relative_to(root_real)
        except ValueError:
            dirnames[:] = []
            continue
        for fname in sorted(filenames):
            if not is_markdown_file(Path(fname)):
                continue
            if should_skip_name(fname):
                continue
            full = current / fname
            if not within_root(full, root_real):
                continue
            rel = full.resolve(strict=False).relative_to(root_real)
            if path_is_forbidden(rel):
                continue
            yield full, rel


def resolve_artifact_id(
    root: Path, artifact_id: str
) -> tuple[Path, Path] | None:
    """Resolve an ``artifact_id`` to ``(absolute_path, relative_path)``.

    Walks the archive and matches :func:`artifact_id_for` against each
    kept Markdown file. Returns ``None`` if no file matches. Raises
    :class:`AmbiguousArtifactId` if two files hash to the same id (a
    path collision — should not happen for real content).
    """
    if not artifact_id or not artifact_id.startswith("archive:"):
        return None
    found: tuple[Path, Path] | None = None
    for full, rel in iter_markdown_files(root):
        if artifact_id_for(rel) == artifact_id:
            if found is not None:
                raise AmbiguousArtifactId(artifact_id)
            found = (full, rel)
    return found


class AmbiguousArtifactId(Exception):
    """Two archive files resolved to the same artifact_id."""


def title_of(content: str, relative_path: Path) -> str:
    """First H1 heading in ``content``, else the file stem."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return relative_path.stem


def read_bounded(path: Path, max_bytes: int = MAX_FILE_BYTES) -> str:
    """Read up to ``max_bytes`` of a file as UTF-8.

    Used for scoring/excerpting so a giant file is never fully loaded.
    A leading byte-order mark is stripped. If the file is not valid
    UTF-8 (binary masquerading as ``.md``) raises :class:`BinaryFileError`.
    """
    with path.open("rb") as fh:
        raw = fh.read(max_bytes)
    # NUL bytes are a strong binary signal.
    if b"\x00" in raw:
        raise BinaryFileError(str(path))
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise BinaryFileError(str(path)) from exc
    return text


class BinaryFileError(Exception):
    """Raised when a ``.md`` file is not decodable as UTF-8 text."""
