"""MedHallBench (Pandit et al., 2025, arXiv 2412.18947).

Medical-LLM hallucination benchmark. Claims spanning drug-drug
interactions, contraindications, dosage, treatment-of, and side-
effects. Evidence is drawn from PubMed abstracts and UpToDate
summaries (where licensing allows).

Schema-wise identical to FEVER / HaluEval after normalisation.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..utils.io import cache_path, read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def load_medhall(
    *,
    max_samples: int | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict]:
    p = cache_path("medhall", "all.jsonl", base=cache_dir)
    if p.exists():
        rows = list(read_jsonl(p))
        if max_samples is not None:
            rows = rows[:max_samples]
        return rows
    rows = _download_medhall()
    write_jsonl(p, rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def _download_medhall() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The HuggingFace `datasets` library is required for MedHallBench loading."
        ) from exc

    log.info("Downloading MedHallBench from HuggingFace...")
    candidates = ["WeijianA/MedHallBench", "pandit/MedHallBench"]
    ds = None
    last_exc = None
    for cand in candidates:
        try:
            ds = load_dataset(cand, split="test")
            break
        except Exception as exc:
            last_exc = exc
            continue
    if ds is None:
        raise RuntimeError(
            f"Could not load MedHallBench from {candidates}. Last error: {last_exc}"
        )

    out: list[dict] = []
    for i, ex in enumerate(ds):
        claim = ex.get("claim") or ex.get("question") or ""
        label_raw = ex.get("label") or ex.get("verdict") or "supported"
        evidence = (
            ex.get("evidence")
            or ex.get("context")
            or ex.get("retrieved_passages")
            or ""
        )
        if isinstance(evidence, list):
            evidence_sentences = [str(e) for e in evidence]
        else:
            evidence_sentences = [str(evidence)] if evidence else []
        label = _map_medhall_label(label_raw)
        out.append(
            {
                "claim_id": f"medhall-{i}",
                "claim": claim,
                "label": label,
                "evidence_sentences": evidence_sentences,
                "evidence_pairs": [],
                "claim_triples": None,
                "evidence_triples": None,
                "source": "medhall",
            }
        )
    return out


def _map_medhall_label(raw) -> int:
    s = str(raw).strip().lower()
    if s in ("s", "supported", "true", "correct", "1"):
        return 0
    if s in ("ns", "not supported", "hallucinated", "false", "incorrect", "0"):
        return 1
    return 2
