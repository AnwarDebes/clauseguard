"""Canonical JSON serialisation, hashing, JSONL IO.

Every audit-relevant object in ClauseGuard is hashed via ``sha256_of``
operating on the canonical JSON dump from ``canonical_json``. This
guarantees that two semantically-identical inputs (e.g., the same
set of triples regardless of insertion order) hash to the same value
and produce the same SAT-verifiable receipt.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def _default(obj: Any) -> Any:
    """JSON encoder fallback that handles dataclasses, sets, and bytes."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, frozenset):
        return sorted(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, deterministic."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def sha256_of(obj: Any) -> str:
    """SHA-256 hex digest of the canonical JSON form of ``obj``."""
    s = canonical_json(obj) if not isinstance(obj, str) else obj
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[Any], *, gzip_output: bool = False) -> None:
    """Write rows to a JSONL file. Creates parent directories.

    If ``gzip_output`` is True or the path ends with ``.gz``, the file
    is gzip-compressed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    use_gzip = gzip_output or str(p).endswith(".gz")
    opener = gzip.open if use_gzip else open
    mode = "wt" if use_gzip else "w"
    with opener(p, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(canonical_json(row))
            f.write("\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Iterate rows from a JSONL file. Transparent gzip if .gz suffix."""
    p = Path(path)
    use_gzip = str(p).endswith(".gz")
    opener = gzip.open if use_gzip else open
    mode = "rt" if use_gzip else "r"
    with opener(p, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def cache_path(*parts: str, base: str | None = None) -> Path:
    """Build a path under the CLAUSEGUARD_CACHE_DIR (defaults to ~/.cache/clauseguard).

    Used by data loaders to memoise downloads and triple extractions.
    """
    if base is None:
        base = os.environ.get("CLAUSEGUARD_CACHE_DIR", "~/.cache/clauseguard")
    p = Path(os.path.expanduser(base), *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
