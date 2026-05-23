"""Train a verifier on FEVER and pickle it to disk for downstream eval scripts.

This is the bridge between ``experiments/train_fever.py`` and the
zero-shot eval scripts (``eval_halueval``, ``eval_factscore``,
``eval_medhall``, ``recourse_eval``, ``adversarial_eval``,
``sat_receipt_demo``).

Usage:
    python scripts/save_verifier.py --seed 42 --output results/verifiers/verifier_seed_42.pkl
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clauseguard.data.fever import load_fever
from clauseguard.extraction.triple_extractor import (
    RegexTripleExtractor,
    SimpleClaimDecomposer,
)
from clauseguard.graphs.dual_graph import DualGraphBuilder
from clauseguard.tm.graphtm_verifier import GraphTMConfig, GraphTMVerifier
from clauseguard.utils.seeding import seed_all


log = logging.getLogger("save_verifier")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and pickle a ClauseGuard verifier.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--max-dev", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--clauses", type=int, default=1000)
    parser.add_argument("--T", type=int, default=1500)
    parser.add_argument("--s", type=float, default=10.0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--extractor", choices=("regex", "qwen"), default="regex")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    seed_all(args.seed)

    train = load_fever(split="train", max_samples=args.max_train)
    dev = load_fever(split="dev", max_samples=args.max_dev)
    log.info("train=%d dev=%d", len(train), len(dev))

    if args.extractor == "regex":
        ext = RegexTripleExtractor()
        dec = SimpleClaimDecomposer(ext)
    else:
        from clauseguard.extraction.qwen_extractor import make_qwen_extractor

        dec, ext = make_qwen_extractor()
    for row in train + dev:
        row["claim_triples"] = list(ext.extract(row["claim"]))
        row["evidence_triples"] = [
            t for sent in (row.get("evidence_sentences") or []) for t in ext.extract(sent)
        ]
    builder = DualGraphBuilder()
    train_graphs = [builder.build(r["claim_triples"], r["evidence_triples"]) for r in train]
    dev_graphs = [builder.build(r["claim_triples"], r["evidence_triples"]) for r in dev]
    train_labels = np.array([r["label"] for r in train], dtype=np.int64)
    dev_labels = np.array([r["label"] for r in dev], dtype=np.int64)

    verifier = GraphTMVerifier(
        GraphTMConfig(
            number_of_clauses=args.clauses,
            T=args.T,
            s=args.s,
            epochs=args.epochs,
            seed=args.seed,
        )
    )
    history = verifier.fit_from_typed_graphs(
        train_graphs,
        train_labels,
        dev_graphs=dev_graphs,
        dev_labels=dev_labels,
    )
    log.info("best dev acc: %s", history.get("best_dev_acc"))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(verifier, f)
    log.info("Wrote pickled verifier to %s", out_path)


if __name__ == "__main__":
    main()
