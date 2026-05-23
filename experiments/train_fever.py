"""Train ClauseGuard's dual-graph GraphTM on FEVER train, evaluate on FEVER dev.

Pipeline:
    1. Load FEVER train + dev.
    2. Extract claim triples and evidence triples (regex fallback if no GPU).
    3. Build dual graphs.
    4. Fit GraphTMVerifier.
    5. Evaluate on dev (label accuracy, macro-F1).
    6. Save per-sample JSONL + summary JSON + verifier artefact id.

Usage:
    python experiments/train_fever.py --seed 42
    python experiments/train_fever.py --seed 42 --max-train 5000 --epochs 10
    python experiments/train_fever.py --seed 42 --extractor qwen
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

# Make the package importable when running ``python experiments/...``.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clauseguard.data.fever import load_fever
from clauseguard.eval.logger import ExperimentLogger
from clauseguard.eval.metrics import label_accuracy, macro_f1
from clauseguard.extraction.triple_extractor import (
    Claim,
    RegexTripleExtractor,
    SimpleClaimDecomposer,
)
from clauseguard.graphs.dual_graph import DualGraphBuilder
from clauseguard.tm.graphtm_verifier import GraphTMConfig, GraphTMVerifier
from clauseguard.utils.seeding import seed_all


log = logging.getLogger("train_fever")


def build_extractor(name: str):
    """Choose between the dependency-free regex extractor and the Qwen LLM extractor."""
    if name == "regex":
        return SimpleClaimDecomposer(RegexTripleExtractor()), RegexTripleExtractor()
    if name == "qwen":
        from clauseguard.extraction.qwen_extractor import make_qwen_extractor

        return make_qwen_extractor()
    raise ValueError(f"unknown extractor {name!r}")


def extract_triples_for_split(rows: list[dict], decomposer, triple_extractor) -> None:
    """In-place: populate ``claim_triples`` and ``evidence_triples`` on each row.

    A row whose triple extraction yields nothing keeps the field as ``[]``;
    the dual-graph builder gracefully degrades to a 1-node, no-edge graph.
    """
    for row in rows:
        if row.get("claim_triples") is None:
            claim_text = row.get("claim", "")
            row["claim_triples"] = list(triple_extractor.extract(claim_text))
        if row.get("evidence_triples") is None:
            evidence_triples: list = []
            for sent in row.get("evidence_sentences") or []:
                evidence_triples.extend(triple_extractor.extract(sent))
            row["evidence_triples"] = evidence_triples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ClauseGuard on FEVER.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None,
                        help="Cap on FEVER training samples; useful for fast iteration.")
    parser.add_argument("--max-dev", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--clauses", type=int, default=1000)
    parser.add_argument("--T", type=int, default=1500)
    parser.add_argument("--s", type=float, default=10.0)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--extractor", choices=("regex", "qwen"), default="regex")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Defaults to results/train_fever/seed_<seed>/.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    seed_all(args.seed)

    results_dir = Path(
        args.results_dir
        or Path(__file__).resolve().parents[1] / "results" / "train_fever" / f"seed_{args.seed}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(
        results_dir / "predictions.jsonl",
        run_meta={
            "experiment": "train_fever",
            "seed": args.seed,
            "max_train": args.max_train,
            "max_dev": args.max_dev,
            "epochs": args.epochs,
            "clauses": args.clauses,
            "T": args.T,
            "s": args.s,
            "depth": args.depth,
            "extractor": args.extractor,
        },
    )

    # 1. Load FEVER.
    log.info("Loading FEVER train+dev")
    train_rows = load_fever(split="train", max_samples=args.max_train)
    dev_rows = load_fever(split="dev", max_samples=args.max_dev)
    log.info("train=%d dev=%d", len(train_rows), len(dev_rows))

    # 2. Extract triples.
    decomposer, triple_extractor = build_extractor(args.extractor)
    log.info("Extracting triples with %s", decomposer.id)
    extract_triples_for_split(train_rows, decomposer, triple_extractor)
    extract_triples_for_split(dev_rows, decomposer, triple_extractor)

    # 3. Build dual graphs.
    builder = DualGraphBuilder()
    train_graphs = [
        builder.build(r["claim_triples"], r["evidence_triples"]) for r in train_rows
    ]
    dev_graphs = [
        builder.build(r["claim_triples"], r["evidence_triples"]) for r in dev_rows
    ]
    train_labels = np.array([r["label"] for r in train_rows], dtype=np.int64)
    dev_labels = np.array([r["label"] for r in dev_rows], dtype=np.int64)

    # 4. Fit verifier.
    config = GraphTMConfig(
        number_of_clauses=args.clauses,
        T=args.T,
        s=args.s,
        depth=args.depth,
        epochs=args.epochs,
        seed=args.seed,
    )
    verifier = GraphTMVerifier(config)
    log.info("Fitting GraphTMVerifier with %s", config)
    history = verifier.fit_from_typed_graphs(
        train_graphs,
        train_labels,
        dev_graphs=dev_graphs,
        dev_labels=dev_labels,
    )

    # 5. Evaluate on dev with per-sample logging.
    dev_pred = verifier.predict_from_typed_graphs(dev_graphs)
    dev_scores = verifier.class_sums_from_typed_graphs(dev_graphs)
    acc = label_accuracy(dev_labels, dev_pred)
    f1 = macro_f1(dev_labels, dev_pred, n_classes=3)
    for row, p, sc in zip(dev_rows, dev_pred, dev_scores):
        logger.log_prediction(
            claim_id=row.get("claim_id", ""),
            true_label=int(row["label"]),
            pred_label=int(p),
            class_scores=[float(x) for x in (sc if hasattr(sc, "__iter__") else [sc])],
        )

    summary = {
        "experiment": "train_fever",
        "seed": args.seed,
        "n_train": len(train_rows),
        "n_dev": len(dev_rows),
        "label_accuracy": float(acc),
        "macro_f1": float(f1),
        "best_dev_acc": history.get("best_dev_acc"),
        "epochs_run": len(history.get("history", [])),
        "verifier_version": verifier.version_id(),
        "extractor": args.extractor,
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary -> %s", results_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
