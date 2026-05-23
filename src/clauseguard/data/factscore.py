"""FActScore (Min et al., EMNLP 2023, arXiv 2305.14251).

Atomic-claim factual precision benchmark. Each generated biography
from GPT-3.5 / GPT-4 / ChatGPT / Vicuna is decomposed into atomic
facts by a teacher LLM, then each fact is judged Supported / Not
Supported against retrieved Wikipedia evidence.

We map FActScore's binary label onto our 3-class scheme:
    Supported     -> 0
    Not Supported -> 1 (Refute)

Atomic facts that are "irrelevant" or "skipped" by FActScore become
class 2 (NEI), they are discarded by FActScore itself but kept here
so the FEVER-trained verifier can be tested on them too.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..utils.io import cache_path, read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def load_factscore(
    *,
    max_samples: int | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict]:
    p = cache_path("factscore", "atomic.jsonl", base=cache_dir)
    if p.exists():
        rows = list(read_jsonl(p))
        if max_samples is not None:
            rows = rows[:max_samples]
        return rows
    rows = _download_factscore()
    write_jsonl(p, rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def _download_factscore() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The HuggingFace `datasets` library is required for FActScore loading."
        ) from exc

    log.info("Downloading FActScore from HuggingFace...")
    # The 'factscore' dataset is mirrored at multiple paths; we try the most
    # common one and degrade gracefully.
    candidates = ["yixinL7/FActScore", "shmsw25/factscore"]
    ds = None
    last_exc = None
    for cand in candidates:
        try:
            ds = load_dataset(cand, split="train")
            break
        except Exception as exc:
            last_exc = exc
            continue
    if ds is None:
        raise RuntimeError(
            f"Could not load FActScore from {candidates}. Last error: {last_exc}"
        )

    out: list[dict] = []
    for i, ex in enumerate(ds):
        topic = ex.get("topic", "")
        atomic_facts = ex.get("atomic_facts", [])
        labels = ex.get("atomic_facts_labels", [])
        evidence = ex.get("retrieved_passages") or ex.get("evidence") or []
        if isinstance(evidence, str):
            evidence_sentences = [evidence]
        elif isinstance(evidence, list):
            evidence_sentences = [str(e) for e in evidence]
        else:
            evidence_sentences = []
        for j, (fact, lab) in enumerate(zip(atomic_facts, labels)):
            label_int = _map_factscore_label(lab)
            out.append(
                {
                    "claim_id": f"factscore-{i}-{j}",
                    "topic": topic,
                    "claim": fact,
                    "label": label_int,
                    "evidence_sentences": evidence_sentences,
                    "evidence_pairs": [],
                    "claim_triples": None,
                    "evidence_triples": None,
                    "source": "factscore",
                }
            )
    return out


def _map_factscore_label(label_raw) -> int:
    if isinstance(label_raw, bool):
        return 0 if label_raw else 1
    s = str(label_raw).strip().lower()
    if s in ("s", "supported", "true", "1", "y", "yes"):
        return 0
    if s in ("ns", "not supported", "not_supported", "false", "0", "n", "no"):
        return 1
    return 2
