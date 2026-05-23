"""Per-sample provenance logger.

Mirrors ``paper-c-tm-robustness/src/eval/logger.py``: every prediction
is appended to a JSONL file with seed, model version, claim id, label,
predicted label, class scores, firing clause ids, and the path to the
SAT receipt. Anyone reproducing the paper can subset the JSONL on any
field and recompute the aggregate metric.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class ExperimentLogger:
    """Append-only JSONL logger.

    Args:
        path: target file path. Parent directories are created.
        run_meta: dict of metadata written as a one-shot header row.
    """

    def __init__(self, path: str | Path, run_meta: dict | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if run_meta is not None:
            self.log({"_kind": "run_meta", **run_meta})

    def log(self, record: Any) -> None:
        if is_dataclass(record):
            record = asdict(record)
        if not isinstance(record, dict):
            record = {"value": record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_default))
            f.write("\n")

    def log_prediction(
        self,
        *,
        claim_id: str,
        true_label: int,
        pred_label: int,
        class_scores: list[float] | None = None,
        firing_clause_ids: list[int] | None = None,
        receipt_path: str | None = None,
        extra: dict | None = None,
    ) -> None:
        rec = {
            "_kind": "prediction",
            "claim_id": claim_id,
            "true_label": int(true_label),
            "pred_label": int(pred_label),
            "class_scores": class_scores,
            "firing_clause_ids": firing_clause_ids or [],
            "receipt_path": receipt_path,
        }
        if extra:
            rec.update(extra)
        self.log(rec)


def _default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
