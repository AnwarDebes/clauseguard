"""HaluEval (Li et al., EMNLP 2023, arXiv 2305.11747).

35 K LLM-hallucinated samples paired with the correct reference for
the same input. Three subsets:

* QA: SQuAD-style question + correct answer + hallucinated answer
* Dialogue: conversation + correct response + hallucinated response
* Summarization: source document + correct summary + hallucinated summary

For our verifier the schema is:

    {
        "claim_id": str,
        "claim": <hallucinated or correct text>,
        "label": 1 (Refute) if hallucinated else 0 (Support),
        "evidence_sentences": [<reference>],
        ...
    }

Each original sample becomes two rows (one hallucinated, one
non-hallucinated). We always preserve the pairing via ``pair_id`` so
downstream calibration can use it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..utils.io import cache_path, read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def load_halueval(
    subset: str = "qa",
    *,
    max_samples: int | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict]:
    if subset not in ("qa", "dialogue", "summarization"):
        raise ValueError(f"unknown HaluEval subset {subset!r}")
    p = cache_path("halueval", f"{subset}.jsonl", base=cache_dir)
    if p.exists():
        rows = list(read_jsonl(p))
        if max_samples is not None:
            rows = rows[:max_samples]
        return rows
    rows = _download_halueval(subset=subset)
    write_jsonl(p, rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def _download_halueval(subset: str) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The HuggingFace `datasets` library is required for HaluEval loading."
        ) from exc

    log.info("Downloading HaluEval %s from HuggingFace...", subset)
    ds = load_dataset("pminervini/HaluEval", subset, split="data")
    out: list[dict] = []
    for i, ex in enumerate(ds):
        ref_key, halu_key, ctx_key = _halu_keys(subset)
        reference = ex.get(ref_key) or ""
        hallucinated = ex.get(halu_key) or ""
        context = ex.get(ctx_key) or ""
        pair_id = f"halueval-{subset}-{i}"
        # Hallucinated row -> label 1 (Refute / hallucinated).
        out.append(
            {
                "claim_id": f"{pair_id}-halu",
                "pair_id": pair_id,
                "claim": hallucinated,
                "label": 1,
                "evidence_sentences": [context] if context else [],
                "evidence_pairs": [],
                "claim_triples": None,
                "evidence_triples": None,
                "source": f"halueval-{subset}",
            }
        )
        # Reference row -> label 0 (Support).
        out.append(
            {
                "claim_id": f"{pair_id}-ref",
                "pair_id": pair_id,
                "claim": reference,
                "label": 0,
                "evidence_sentences": [context] if context else [],
                "evidence_pairs": [],
                "claim_triples": None,
                "evidence_triples": None,
                "source": f"halueval-{subset}",
            }
        )
    return out


def _halu_keys(subset: str) -> tuple[str, str, str]:
    if subset == "qa":
        return "right_answer", "hallucinated_answer", "question"
    if subset == "dialogue":
        return "right_response", "hallucinated_response", "dialogue_history"
    if subset == "summarization":
        return "right_summary", "hallucinated_summary", "document"
    raise ValueError(subset)
