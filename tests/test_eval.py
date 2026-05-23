"""Tests for the eval helpers."""
from __future__ import annotations

import numpy as np

from clauseguard.eval.adversarial import (
    AdversarialResult,
    run_negation_flip_attack,
    run_paraphrase_attack,
)
from clauseguard.eval.logger import ExperimentLogger
from clauseguard.eval.metrics import (
    auroc,
    bootstrap_ci,
    label_accuracy,
    macro_f1,
    paired_wilcoxon,
)


def test_label_accuracy_perfect():
    assert label_accuracy([0, 1, 2], [0, 1, 2]) == 1.0


def test_label_accuracy_half():
    assert label_accuracy([0, 1, 2, 0], [0, 1, 2, 1]) == 0.75


def test_macro_f1_perfect():
    assert abs(macro_f1([0, 1, 2], [0, 1, 2], n_classes=3) - 1.0) < 1e-9


def test_auroc_perfect_separation():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auroc_random():
    # Scores anti-correlated with labels -> AUROC near 0
    val = auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert val < 0.5


def test_paired_wilcoxon_greater_significant():
    # Need at least 6 non-zero differences for scipy wilcoxon to return a finite p
    res = paired_wilcoxon([1.1, 1.2, 1.3, 1.4, 1.5, 1.6], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], alternative="greater")
    assert "pvalue" in res
    assert res["pvalue"] < 0.5


def test_bootstrap_ci_returns_three_floats():
    mean, lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_resamples=200, seed=1)
    assert lo <= mean <= hi


def test_paraphrase_attack_runs():
    def predict(text: str) -> int:
        return 0 if "researcher" in text else 1

    samples = [
        {"claim_id": "a", "claim": "Alice is a researcher", "label": 0},
        {"claim_id": "b", "claim": "Bob is a clinician", "label": 1},
    ]
    res = run_paraphrase_attack(predict, samples, seed=1)
    assert isinstance(res, AdversarialResult)
    assert res.n_samples == 2


def test_negation_flip_attack_changes_some_labels():
    def predict(text: str) -> int:
        return 1 if "not" in text else 0

    samples = [{"claim_id": "x", "claim": "Alice is happy", "label": 0}]
    res = run_negation_flip_attack(predict, samples, seed=1)
    # Original predicts 0, after flip predicts 1.
    assert res.clean_accuracy == 1.0
    assert res.adversarial_accuracy == 0.0


def test_experiment_logger_writes_jsonl(tmp_path):
    log_path = tmp_path / "log.jsonl"
    logger = ExperimentLogger(log_path, run_meta={"hello": "world"})
    logger.log_prediction(
        claim_id="c1",
        true_label=0,
        pred_label=0,
        class_scores=[1.0, -1.0, -0.5],
    )
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2  # meta + prediction
