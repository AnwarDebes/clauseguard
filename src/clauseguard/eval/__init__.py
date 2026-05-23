"""Evaluation utilities: metrics, adversarial harness, per-sample logger."""
from .metrics import (
    auroc,
    bootstrap_ci,
    fever_score,
    label_accuracy,
    macro_f1,
    paired_wilcoxon,
)
from .logger import ExperimentLogger
from .adversarial import (
    AdversarialResult,
    run_paraphrase_attack,
    run_textfooler_attack,
)

__all__ = [
    "auroc",
    "bootstrap_ci",
    "fever_score",
    "label_accuracy",
    "macro_f1",
    "paired_wilcoxon",
    "ExperimentLogger",
    "AdversarialResult",
    "run_paraphrase_attack",
    "run_textfooler_attack",
]
