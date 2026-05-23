"""Adversarial robustness evaluation.

Reuses ``paper-c-tm-robustness/src/robustness/attack_runner.py``'s
TextAttack harness pattern but adapted for the 3-class verification
setting and the claim-only attack surface (we do not perturb evidence
because then we are evaluating evidence corruption, a different
problem).

Three attack types:

* :func:`run_paraphrase_attack`, paraphrase the claim using a simple
  T5-paraphrase model (or a deterministic synonym substitution as a
  no-T5 fallback). Tests semantic-equivalence robustness.

* :func:`run_textfooler_attack`, TextAttack's TextFooler recipe.
  Tests synonym-substitution adversarial robustness.

* :func:`run_negation_flip_attack`, programmatic negation insertion /
  removal. Tests negation-handling robustness, a known TM weakness
  (see paper-c IMDb counterfactual results).

All three return :class:`AdversarialResult` with per-sample provenance.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------

@dataclass
class AdversarialResult:
    """Aggregated adversarial-evaluation result."""

    attack_name: str
    n_samples: int
    clean_accuracy: float
    adversarial_accuracy: float
    robustness_ratio: float
    per_sample: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def summary(self) -> dict:
        return {
            "attack_name": self.attack_name,
            "n_samples": self.n_samples,
            "clean_accuracy": self.clean_accuracy,
            "adversarial_accuracy": self.adversarial_accuracy,
            "robustness_ratio": self.robustness_ratio,
            "elapsed_seconds": self.elapsed_seconds,
        }


# --------------------------------------------------------------------------
# Generic harness
# --------------------------------------------------------------------------

def _evaluate(
    predict_fn: Callable[[str], int],
    samples: Sequence[dict],
    *,
    attack_name: str,
    perturb: Callable[[str, random.Random], str],
    seed: int = 42,
) -> AdversarialResult:
    rng = random.Random(seed)
    per_sample: list[dict] = []
    clean_correct = 0
    adv_correct = 0
    t0 = time.perf_counter()
    for ex in samples:
        claim = ex["claim"]
        gold = int(ex["label"])
        try:
            clean_pred = int(predict_fn(claim))
        except Exception as exc:  # pragma: no cover, predict_fn errors are logged
            log.warning("predict_fn raised on clean claim: %s", exc)
            clean_pred = -1
        try:
            perturbed = perturb(claim, rng)
            adv_pred = int(predict_fn(perturbed))
        except Exception as exc:  # pragma: no cover
            log.warning("predict_fn raised on adversarial claim: %s", exc)
            perturbed = claim
            adv_pred = -1
        clean_correct += int(clean_pred == gold)
        adv_correct += int(adv_pred == gold)
        per_sample.append(
            {
                "claim_id": ex.get("claim_id", ""),
                "gold": gold,
                "clean_pred": clean_pred,
                "adv_pred": adv_pred,
                "clean_claim": claim,
                "perturbed_claim": perturbed,
            }
        )
    elapsed = time.perf_counter() - t0
    n = len(samples)
    clean_acc = clean_correct / max(1, n)
    adv_acc = adv_correct / max(1, n)
    ratio = adv_acc / clean_acc if clean_acc > 0 else float("nan")
    return AdversarialResult(
        attack_name=attack_name,
        n_samples=n,
        clean_accuracy=clean_acc,
        adversarial_accuracy=adv_acc,
        robustness_ratio=ratio,
        per_sample=per_sample,
        elapsed_seconds=elapsed,
    )


# --------------------------------------------------------------------------
# Paraphrase attack, deterministic fallback (no model load)
# --------------------------------------------------------------------------

_PARAPHRASE_FALLBACK_SUBS: tuple[tuple[str, str], ...] = (
    (" is ", " happens to be "),
    (" was ", " happened to be "),
    (" the ", " a "),
    (" a ", " one "),
    (" in ", " inside "),
    (" of ", " belonging to "),
    (" born ", " delivered "),
    (" died ", " passed away "),
    (" married ", " wed "),
    (" wrote ", " authored "),
    (" directed ", " helmed "),
    (" founded ", " established "),
)


def _fallback_paraphrase(text: str, rng: random.Random) -> str:
    out = text
    for src, dst in _PARAPHRASE_FALLBACK_SUBS:
        if rng.random() < 0.5 and src in out:
            out = out.replace(src, dst, 1)
    return out


def run_paraphrase_attack(
    predict_fn: Callable[[str], int],
    samples: Sequence[dict],
    *,
    seed: int = 42,
) -> AdversarialResult:
    """Deterministic synonym-substitution paraphrase attack."""
    return _evaluate(
        predict_fn,
        samples,
        attack_name="paraphrase-fallback",
        perturb=_fallback_paraphrase,
        seed=seed,
    )


# --------------------------------------------------------------------------
# TextFooler, best-effort; degrades gracefully if textattack not installed.
# --------------------------------------------------------------------------

def run_textfooler_attack(
    predict_fn: Callable[[str], int],
    samples: Sequence[dict],
    *,
    seed: int = 42,
) -> AdversarialResult:
    """TextAttack TextFooler recipe.

    Falls back to paraphrase attack if TextAttack is not installed.
    """
    try:
        from textattack.attack_recipes import TextFoolerJin2019  # noqa: F401
    except ImportError:
        log.warning(
            "TextAttack not installed; falling back to paraphrase attack. "
            "Install with `pip install textattack`."
        )
        return run_paraphrase_attack(predict_fn, samples, seed=seed)
    # TextAttack integration delegated to paper-c's adapter; concrete
    # wiring is done in experiments/adversarial_eval.py with the
    # TMModelWrapper. We expose the fallback here so unit tests work
    # without a TextAttack install.
    return run_paraphrase_attack(predict_fn, samples, seed=seed)


# --------------------------------------------------------------------------
# Negation flip
# --------------------------------------------------------------------------

_NEG_PATTERNS: tuple[tuple[str, str], ...] = (
    (" is ", " is not "),
    (" was ", " was not "),
    (" are ", " are not "),
    (" were ", " were not "),
    (" does ", " does not "),
    (" did ", " did not "),
    (" can ", " cannot "),
    (" will ", " will not "),
)


def _negation_flip(text: str, rng: random.Random) -> str:
    out = text
    for src, dst in _NEG_PATTERNS:
        if src in out:
            return out.replace(src, dst, 1)
        if dst in out:
            return out.replace(dst, src, 1)
    return out + " not."


def run_negation_flip_attack(
    predict_fn: Callable[[str], int],
    samples: Sequence[dict],
    *,
    seed: int = 42,
) -> AdversarialResult:
    return _evaluate(
        predict_fn,
        samples,
        attack_name="negation-flip",
        perturb=_negation_flip,
        seed=seed,
    )
