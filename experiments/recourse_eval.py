"""Recourse evaluation: for Refuted FEVER dev predictions, find minimum
evidence-graph edits that flip the verifier to Supported.

For each Refuted prediction, run the greedy minimum-edit search and
record:

    - whether the flip succeeded within max_edits,
    - the number of edits applied,
    - the per-step latency,
    - the rendered recourse report.

Outputs: ``results/recourse_eval/seed_<seed>/{summary.json,
recourse.jsonl, reports/*.md}``.
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

from clauseguard.data.fever import load_fever
from clauseguard.eval.logger import ExperimentLogger
from clauseguard.extraction.triple_extractor import (
    RegexTripleExtractor,
    SimpleClaimDecomposer,
)
from clauseguard.graphs.dual_graph import DualGraphBuilder
from clauseguard.recourse.output import render_recourse_report
from clauseguard.recourse.search import greedy_minimal_evidence_edit
from clauseguard.tm.graphtm_verifier import GraphTMVerifier, LABEL_NAMES
from clauseguard.utils.seeding import seed_all


log = logging.getLogger("recourse_eval")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recourse evaluation on FEVER dev.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verifier-pickle", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-edits", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--extractor", choices=("regex", "qwen"), default="regex")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    seed_all(args.seed)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "recourse_eval"
        / f"seed_{args.seed}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "reports").mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(
        results_dir / "recourse.jsonl",
        run_meta={
            "experiment": "recourse_eval",
            "seed": args.seed,
            "max_edits": args.max_edits,
            "max_candidates": args.max_candidates,
        },
    )

    rows = load_fever(split="dev", max_samples=args.max_samples)
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

    with open(args.verifier_pickle, "rb") as f:
        verifier: GraphTMVerifier = pickle.load(f)

    builder = DualGraphBuilder()
    # First pass: predict.
    graphs = [builder.build(r["claim_triples"], r["evidence_triples"]) for r in rows]
    preds = verifier.predict_from_typed_graphs(graphs)
    sums = np.asarray(verifier.class_sums_from_typed_graphs(graphs))

    n_attempts = 0
    n_flips = 0
    edits_used: list[int] = []
    latencies: list[float] = []

    for row, p, sc, g in zip(rows, preds, sums, graphs):
        if int(p) == 0:  # already Supported
            continue
        n_attempts += 1
        applied, flipped, trace = greedy_minimal_evidence_edit(
            verifier=verifier,
            builder=builder,
            claim_triples=row["claim_triples"],
            evidence_triples=row["evidence_triples"],
            target_label=0,
            max_edits=args.max_edits,
            max_candidates=args.max_candidates,
        )
        n_flips += int(flipped)
        if flipped:
            edits_used.append(len(applied))
        latencies.append(trace.total_latency_ms)

        # Render and persist a short recourse report.
        original_label_name = LABEL_NAMES[int(p)]
        final_scores = (
            trace.scores_after[-1].tolist()
            if trace.scores_after
            else trace.scores_before[-1].tolist()
        )
        original_scores = trace.scores_before[0].tolist()
        final_label = int(np.argmax(final_scores))
        final_label_name = LABEL_NAMES[final_label]
        report_md = render_recourse_report(
            claim_text=row["claim"],
            original_label_name=original_label_name,
            original_scores=original_scores,
            final_label_name=final_label_name,
            final_scores=final_scores,
            trace=trace,
            case_id=row.get("claim_id", ""),
        )
        with open(results_dir / "reports" / f"{row.get('claim_id', '')}.md", "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.log({
            "_kind": "recourse",
            "claim_id": row.get("claim_id", ""),
            "trace": trace.to_dict(),
            "applied_edits": len(applied),
            "flipped": bool(flipped),
            "final_label": int(final_label),
        })

    summary = {
        "experiment": "recourse_eval",
        "seed": args.seed,
        "n_dev": len(rows),
        "n_recourse_attempts": n_attempts,
        "n_flips": n_flips,
        "flip_rate": n_flips / max(1, n_attempts),
        "mean_edits_to_flip": (float(np.mean(edits_used)) if edits_used else None),
        "median_latency_ms": float(np.median(latencies)) if latencies else None,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "verifier_version": verifier.version_id(),
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
