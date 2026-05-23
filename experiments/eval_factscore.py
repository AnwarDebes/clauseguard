"""Zero-shot evaluation of a FEVER-trained ClauseGuard on FActScore atomic claims."""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clauseguard.data.factscore import load_factscore
from clauseguard.eval.logger import ExperimentLogger
from clauseguard.eval.metrics import auroc, label_accuracy, macro_f1
from clauseguard.extraction.triple_extractor import (
    RegexTripleExtractor,
    SimpleClaimDecomposer,
)
from clauseguard.graphs.dual_graph import DualGraphBuilder
from clauseguard.tm.graphtm_verifier import GraphTMVerifier
from clauseguard.utils.seeding import seed_all


log = logging.getLogger("eval_factscore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ClauseGuard on FActScore.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--verifier-pickle", type=str, required=True)
    parser.add_argument("--extractor", choices=("regex", "qwen"), default="regex")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    seed_all(args.seed)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "eval_factscore"
        / f"seed_{args.seed}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(
        results_dir / "predictions.jsonl",
        run_meta={"experiment": "eval_factscore", "seed": args.seed},
    )

    rows = load_factscore(max_samples=args.max_samples)
    if args.extractor == "regex":
        ext = RegexTripleExtractor()
        dec = SimpleClaimDecomposer(ext)
    else:
        from clauseguard.extraction.qwen_extractor import make_qwen_extractor

        dec, ext = make_qwen_extractor()
    for row in rows:
        row["claim_triples"] = list(ext.extract(row["claim"]))
        row["evidence_triples"] = [
            t for sent in (row.get("evidence_sentences") or []) for t in ext.extract(sent)
        ]
    builder = DualGraphBuilder()
    graphs = [builder.build(r["claim_triples"], r["evidence_triples"]) for r in rows]

    with open(args.verifier_pickle, "rb") as f:
        verifier: GraphTMVerifier = pickle.load(f)
    preds = verifier.predict_from_typed_graphs(graphs)
    sums = np.asarray(verifier.class_sums_from_typed_graphs(graphs), dtype=float)
    labels = np.array([r["label"] for r in rows], dtype=np.int64)

    # 3-class accuracy + binary precision/recall against {0: supported, 1: not}.
    bin_mask = labels < 2
    bin_pred = preds[bin_mask] == 1
    bin_gold = labels[bin_mask] == 1
    au = auroc(
        bin_gold.astype(int),
        (sums[bin_mask, 1] - sums[bin_mask, 0]) if sums.ndim == 2 else preds[bin_mask],
    )
    acc = label_accuracy(labels, preds)
    f1 = macro_f1(labels, preds, n_classes=3)
    for row, p, sc in zip(rows, preds, sums):
        logger.log_prediction(
            claim_id=row["claim_id"],
            true_label=int(row["label"]),
            pred_label=int(p),
            class_scores=list(map(float, sc)) if sc.ndim else [float(sc)],
        )

    summary = {
        "experiment": "eval_factscore",
        "seed": args.seed,
        "n_samples": len(rows),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "binary_auroc_S_vs_NS": float(au),
        "verifier_version": verifier.version_id(),
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
