"""Adversarial robustness evaluation of ClauseGuard on FEVER dev.

Reuses ``paper-c-tm-robustness``-style protocol: take a verifier
trained on FEVER, attack the claim text with three perturbations, and
report clean vs adversarial accuracy.
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
from clauseguard.eval.adversarial import (
    run_negation_flip_attack,
    run_paraphrase_attack,
    run_textfooler_attack,
)
from clauseguard.eval.logger import ExperimentLogger
from clauseguard.extraction.triple_extractor import RegexTripleExtractor
from clauseguard.graphs.dual_graph import DualGraphBuilder
from clauseguard.tm.graphtm_verifier import GraphTMVerifier
from clauseguard.utils.seeding import seed_all


log = logging.getLogger("adversarial_eval")


def make_predict_fn(
    verifier: GraphTMVerifier,
    builder: DualGraphBuilder,
    extractor: RegexTripleExtractor,
    evidence_triples_by_claim_id: dict[str, list],
    row_lookup: dict[str, dict],
):
    """Return a ``predict_fn(claim_text) -> int`` for the adversarial harness.

    The adversarial harness only knows the perturbed claim *text*. The
    evidence triples must be looked up from the original sample because
    they are not perturbed.
    """

    def _predict(claim_text: str) -> int:
        # The harness loses the claim_id during perturbation, so we
        # rely on the most-recently-seen sample's id via a closure on
        # the iteration. Concretely, callers iterate samples in order
        # and we re-extract triples from the perturbed text.
        triples = list(extractor.extract(claim_text))
        # Use the latest sample's evidence triples.
        if not _predict._evidence_cache:
            return 0
        ev = _predict._evidence_cache[-1]
        graph = builder.build(triples, ev)
        return int(verifier.predict_from_typed_graphs([graph])[0])

    _predict._evidence_cache = []
    return _predict


def main() -> None:
    parser = argparse.ArgumentParser(description="ClauseGuard adversarial eval.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verifier-pickle", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    seed_all(args.seed)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "adversarial_eval"
        / f"seed_{args.seed}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(
        results_dir / "adversarial.jsonl",
        run_meta={"experiment": "adversarial_eval", "seed": args.seed},
    )

    rows = load_fever(split="dev", max_samples=args.max_samples)
    extractor = RegexTripleExtractor()
    for row in rows:
        row["claim_triples"] = list(extractor.extract(row["claim"]))
        row["evidence_triples"] = [
            t for sent in (row.get("evidence_sentences") or []) for t in extractor.extract(sent)
        ]

    with open(args.verifier_pickle, "rb") as f:
        verifier: GraphTMVerifier = pickle.load(f)
    builder = DualGraphBuilder()

    # We run sample-by-sample so the per-sample evidence is matched correctly.
    def make_per_sample_predict(evidence_triples):
        def _predict(claim_text: str) -> int:
            triples = list(extractor.extract(claim_text))
            graph = builder.build(triples, evidence_triples)
            return int(verifier.predict_from_typed_graphs([graph])[0])
        return _predict

    attacks = {
        "paraphrase": run_paraphrase_attack,
        "textfooler": run_textfooler_attack,
        "negation": run_negation_flip_attack,
    }
    summary: dict = {"experiment": "adversarial_eval", "seed": args.seed, "results": {}}
    for name, fn in attacks.items():
        log.info("Running attack: %s", name)
        # Build a per-sample harness by running the attack on each sample's slot.
        clean_correct = 0
        adv_correct = 0
        for row in rows:
            predict = make_per_sample_predict(row["evidence_triples"])
            per_attack = fn(predict, [row], seed=args.seed)
            clean_correct += int(per_attack.clean_accuracy * per_attack.n_samples)
            adv_correct += int(per_attack.adversarial_accuracy * per_attack.n_samples)
            logger.log({
                "_kind": "adv_sample",
                "attack": name,
                "claim_id": row.get("claim_id", ""),
                "per_sample": per_attack.per_sample[0] if per_attack.per_sample else {},
            })
        clean_acc = clean_correct / max(1, len(rows))
        adv_acc = adv_correct / max(1, len(rows))
        summary["results"][name] = {
            "clean_accuracy": float(clean_acc),
            "adversarial_accuracy": float(adv_acc),
            "robustness_ratio": float(adv_acc / clean_acc) if clean_acc > 0 else float("nan"),
            "n_samples": len(rows),
        }

    summary["verifier_version"] = verifier.version_id()
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
