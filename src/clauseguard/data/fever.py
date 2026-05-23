"""FEVER (Fact Extraction and Verification) loader.

Loads the canonical FEVER train/dev/test splits via HuggingFace
``datasets``. The dev split has gold evidence; the blind test split
requires submission to the FEVER leaderboard. We support a held-out
"shared task" dev split (a 10% slice) for in-house evaluation when the
blind set is not appropriate.

Reference: Thorne et al., NAACL 2018, "FEVER: a large-scale dataset
for fact extraction and verification" (https://fever.ai/).

Labels:
    SUPPORTS    -> 0
    REFUTES     -> 1
    NOT ENOUGH INFO -> 2

Evidence is a list of (Wikipedia page, sentence index) tuples in the
original dataset. We resolve to plain sentences via the
``fever_evidence_sentences`` helper from the FEVER repo when possible;
when unavailable (test split, network constraints, etc.) we leave
``evidence_sentences`` empty and downstream code falls back to the
"no evidence" path (which then surfaces as NotEnoughInfo).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..utils.io import cache_path, read_jsonl, write_jsonl

log = logging.getLogger(__name__)


FEVER_LABEL_MAP = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
    "NOT_ENOUGH_INFO": 2,
    "NEI": 2,
    "Supports": 0,
    "Refutes": 1,
}


def load_fever(
    split: str = "train",
    *,
    max_samples: int | None = None,
    cache_dir: str | Path | None = None,
    resolve_evidence: bool = True,
) -> list[dict]:
    """Load FEVER. Cached as JSONL under ``$CLAUSEGUARD_CACHE_DIR/fever/``."""
    p = cache_path("fever", f"{split}.jsonl", base=cache_dir)
    if p.exists():
        rows = list(read_jsonl(p))
        log.info("Loaded %d rows from cache %s", len(rows), p)
        if max_samples is not None:
            rows = rows[:max_samples]
        return rows

    rows = _download_fever(split=split, resolve_evidence=resolve_evidence)
    write_jsonl(p, rows)
    log.info("Wrote %d FEVER rows to %s", len(rows), p)
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def _download_fever(split: str, resolve_evidence: bool) -> list[dict]:
    """Download the FEVER split from HuggingFace and normalise into our schema."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The HuggingFace `datasets` library is required for FEVER loading. "
            "Run `pip install datasets`."
        ) from exc

    log.info("Downloading FEVER %s from HuggingFace...", split)
    # The 'fever' dataset on HF is split as: train / labelled_dev / paper_dev /
    # paper_test / etc. We map our 'train' / 'dev' / 'test' onto canonical splits.
    hf_split_map = {
        "train": ("fever/v1.0", "train"),
        "dev": ("fever/v1.0", "labelled_dev"),
        "paper_dev": ("fever/v1.0", "paper_dev"),
        "test": ("fever/v1.0", "paper_test"),
    }
    if split not in hf_split_map:
        raise ValueError(f"unknown FEVER split {split!r}")

    config, hf_split = hf_split_map[split]
    try:
        ds = load_dataset(config.split("/", 1)[0], config.split("/", 1)[1], split=hf_split)
    except Exception as exc:  # pragma: no cover, network / availability fallback
        log.warning(
            "Failed to load FEVER from HF (%s); falling back to local FEVER repo path",
            exc,
        )
        return _load_fever_from_local(split, resolve_evidence=resolve_evidence)

    out: list[dict] = []
    for i, ex in enumerate(ds):
        label_raw = ex.get("label") or ex.get("verifiable") or ""
        label = FEVER_LABEL_MAP.get(str(label_raw).upper(), 2)
        evidence_pairs = ex.get("evidence") or []
        # Normalise evidence to a list of (page, sentence_id) tuples.
        flattened_pairs = []
        for ev_group in evidence_pairs:
            for ev in ev_group:
                if isinstance(ev, (list, tuple)) and len(ev) >= 4:
                    page, sent_id = ev[2], ev[3]
                    if page and sent_id is not None:
                        flattened_pairs.append((page, int(sent_id)))
        evidence_sentences: list[str] = []
        if resolve_evidence and flattened_pairs:
            evidence_sentences = _resolve_evidence_sentences(flattened_pairs)
        out.append(
            {
                "claim_id": str(ex.get("id", i)),
                "claim": ex.get("claim", ""),
                "label": int(label),
                "evidence_sentences": evidence_sentences,
                "evidence_pairs": flattened_pairs,
                "claim_triples": None,
                "evidence_triples": None,
                "source": "fever",
            }
        )
    return out


def _load_fever_from_local(split: str, resolve_evidence: bool) -> list[dict]:
    """Fallback: load FEVER JSONL files from a local path.

    Looks under ``$FEVER_DATA_DIR/<split>.jsonl`` (typical layout when
    using the official FEVER scripts).
    """
    import json
    import os

    base = os.environ.get("FEVER_DATA_DIR")
    if not base:
        raise RuntimeError(
            "FEVER data not available via HuggingFace and FEVER_DATA_DIR is unset. "
            "Set FEVER_DATA_DIR to a directory containing train.jsonl / dev.jsonl."
        )
    p = Path(base, f"{split}.jsonl")
    if not p.exists():
        raise FileNotFoundError(p)
    out: list[dict] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        label = FEVER_LABEL_MAP.get(str(ex.get("label", "")).upper(), 2)
        out.append(
            {
                "claim_id": str(ex.get("id", "")),
                "claim": ex.get("claim", ""),
                "label": int(label),
                "evidence_sentences": [],
                "evidence_pairs": ex.get("evidence", []),
                "claim_triples": None,
                "evidence_triples": None,
                "source": "fever",
            }
        )
    return out


def _resolve_evidence_sentences(pairs: list[tuple[str, int]]) -> list[str]:
    """Resolve (page, sentence_id) pairs to plain sentences.

    The canonical resolution uses the FEVER Wikipedia dump (which is
    ~ 11 GB). To avoid forcing that download for everyone, we expose a
    fallback environment variable ``FEVER_WIKI_DIR`` pointing at the
    extracted JSONL pages directory. If unset, we return an empty list
    and downstream code surfaces NotEnoughInfo for those samples.
    """
    import json
    import os

    base = os.environ.get("FEVER_WIKI_DIR")
    if not base:
        return []
    cache: dict[str, list[str]] = {}
    out: list[str] = []
    for page, sid in pairs:
        if page not in cache:
            p = Path(base, f"{page}.jsonl")
            if not p.exists():
                cache[page] = []
                continue
            sentences: list[str] = []
            for line in p.read_text().splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sentences.append(d.get("text", ""))
            cache[page] = sentences
        if 0 <= sid < len(cache[page]):
            out.append(cache[page][sid])
    return out
