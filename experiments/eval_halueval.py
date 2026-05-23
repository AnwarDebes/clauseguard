"""Zero-shot evaluation of a FEVER-trained ClauseGuard on HaluEval.

Loads the verifier trained by ``train_fever.py`` (looked up via
``--seed`` and ``--results-dir``), evaluates on HaluEval QA / Dialogue
/ Summarization, and reports binary AUROC.

Because HaluEval is binary (hallucinated vs reference), we collapse
the 3-class verifier output as:

    P(refute) = vote_share(Refute)
    AUROC computed against (label == 1).

We do NOT retrain on HaluEval, this evaluates zero-shot generalisation
from FEVER to LLM-output verification, which is the deployment story.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clauseguard.data.halueval import load_halueval
from clauseguard.eval.logger import ExperimentLogger
from clauseguard.eval.metrics import auroc, label_accuracy
from clauseguard.extraction.triple_extractor import (
    RegexTripleExtractor,
    SimpleClaimDecomposer,
)
from clauseguard.graphs.dual_graph import DualGraphBuilder
from clauseguard.tm.graphtm_verifier import GraphTMVerifier
from clauseguard.utils.seeding import seed_all


log = logging.getLogger("eval_halueval")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ClauseGuard on HaluEval.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", choices=("qa", "dialogue", "summarization"), default="qa")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument(
        "--verifier-pickle",
        type=str,
        default=None,
        help="Path to a pickled GraphTMVerifier (see scripts/save_verifier.py).",
    )
    parser.add_argument("--extractor", choices=("regex", "qwen"), default="regex")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    seed_all(args.seed)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "results"
        / f"eval_halueval_{args.subset}"
        / f"seed_{args.seed}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(
        results_dir / "predictions.jsonl",
        run_meta={
            "experiment": "eval_halueval",
            "subset": args.subset,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "extractor": args.extractor,
        },
    )

    rows = load_halueval(args.subset, max_samples=args.max_samples)
    log.info("Loaded %d HaluEval rows", len(rows))

    if args.verifier_pickle is None:
        raise SystemExit(
            "Provide --verifier-pickle (pickled GraphTMVerifier produced by "
            "scripts/save_verifier.py). Pickle file holds the FEVER-trained model."
        )
    with open(args.verifier_pickle, "rb") as f:
        verifier: GraphTMVerifier = pickle.load(f)
    log.info("Loaded verifier %s", verifier.version_id())

    # Triple extraction (zero-shot, reuse the same extractor as training).
    if args.extractor == "regex":
        ext = RegexTripleExtractor()
        decomposer = SimpleClaimDecomposer(ext)
    else:
        from clauseguard.extraction.qwen_extractor import make_qwen_extractor

        decomposer, ext = make_qwen_extractor()
    log.info("Extracting triples with %s", decomposer.id)
    for row in rows:
        row["claim_triples"] = list(ext.extract(row["claim"]))
        ev_triples: list = []
        for sent in row.get("evidence_sentences") or []:
            ev_triples.extend(ext.extract(sent))
        row["evidence_triples"] = ev_triples

    builder = DualGraphBuilder()
    graphs = [builder.build(r["claim_triples"], r["evidence_triples"]) for r in rows]
    labels = np.array([r["label"] for r in rows], dtype=np.int64)

    preds = verifier.predict_from_typed_graphs(graphs)
    sums = verifier.class_sums_from_typed_graphs(graphs)
    # P(refute) ~ relative class sum for class 1 vs class 0.
    sums_arr = np.asarray(sums, dtype=float)
    if sums_arr.ndim == 2:
        refute_score = sums_arr[:, 1] - sums_arr[:, 0]
    else:
        refute_score = np.where(preds == 1, 1.0, -1.0)

    # HaluEval is binary: collapse our 3-class label.
    binary_pred = (preds == 1).astype(int)
    # Map gold {0=Support, 1=Refute} to binary directly.
    binary_gold = labels.astype(int)
    acc = label_accuracy(binary_gold, binary_pred)
    au = auroc(binary_gold, refute_score)
    for row, p, score in zip(rows, binary_pred, refute_score):
        logger.log_prediction(
            claim_id=row.get("claim_id", ""),
            true_label=int(row["label"]),
            pred_label=int(p),
            class_scores=[float(score)],
        )

    summary = {
        "experiment": "eval_halueval",
        "subset": args.subset,
        "seed": args.seed,
        "n_samples": len(rows),
        "accuracy": float(acc),
        "auroc": float(au),
        "verifier_version": verifier.version_id(),
        "extractor": args.extractor,
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", results_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
