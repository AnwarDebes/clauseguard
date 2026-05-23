"""LLM-judge -> GraphTM distillation driver.

Mirrors the GIN -> HGTM distillation pattern from
``graphtm-cbr/graphtm/distill/student.py``. The teacher here is an LLM
judge (e.g., Llama-3.1-8B-Instruct or GPT-4-judge) that scores each
(claim, evidence) pair with a soft probability over
{Support, Refute, NotEnoughInfo}. The student GraphTM is trained on a
mixture of hard ground-truth labels and these soft labels.

The teacher API is abstract, concrete teachers live in
``clauseguard.tm.teachers`` (not in this file because each teacher has
its own dependencies; the LLM judge integration is a follow-up).

The CAIR ``MultiClassGraphTsetlinMachine`` does not natively accept
soft labels. We approximate distillation via a *label-smoothing*
strategy: per-sample, with probability proportional to the teacher's
confidence, the hard label is replaced by the teacher's argmax; the
remainder keep the ground-truth label. This is the same trick used in
graphtm-cbr Phase 2 (paper §4, locked numbers).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .graphtm_verifier import GraphTMConfig, GraphTMVerifier
from ..graphs.dual_graph import TypedGraph

log = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for soft-label distillation.

    Attributes:
        teacher_weight: probability of using the teacher's argmax label
            instead of the ground truth, scaled by the teacher's
            confidence on the sample. ``0.0`` disables distillation.
        confidence_floor: if the teacher's max probability is below this,
            keep the hard label regardless of ``teacher_weight``.
        random_seed: RNG seed for the label-mixing.
    """

    teacher_weight: float = 0.5
    confidence_floor: float = 0.6
    random_seed: int = 42


def mix_labels(
    hard_labels: np.ndarray,
    teacher_probs: np.ndarray,
    config: DistillationConfig,
) -> np.ndarray:
    """Mix hard ground-truth labels with the teacher's argmax labels.

    Arguments:
        hard_labels: shape ``[n,]`` int.
        teacher_probs: shape ``[n, n_classes]`` float, rows sum to 1.
        config: :class:`DistillationConfig`.

    Returns:
        ``[n,]`` int array of distilled labels.
    """
    if config.teacher_weight <= 0.0:
        return hard_labels.astype(np.int64)

    rng = np.random.default_rng(config.random_seed)
    teacher_labels = teacher_probs.argmax(axis=1)
    teacher_conf = teacher_probs.max(axis=1)
    # Use teacher label when teacher_conf >= floor and a Bernoulli draw
    # exceeds (1 - teacher_weight * teacher_conf).
    use_teacher_p = np.clip(config.teacher_weight * teacher_conf, 0.0, 1.0)
    use_teacher_p = np.where(teacher_conf >= config.confidence_floor, use_teacher_p, 0.0)
    draws = rng.random(len(hard_labels))
    mixed = np.where(draws < use_teacher_p, teacher_labels, hard_labels)
    log.info(
        "Distillation: replaced %d / %d hard labels with teacher labels (mean teacher conf %.3f)",
        int((mixed != hard_labels).sum()),
        len(hard_labels),
        float(teacher_conf.mean()) if len(teacher_conf) else 0.0,
    )
    return mixed.astype(np.int64)


def distill(
    train_graphs: list[TypedGraph],
    train_hard_labels: np.ndarray,
    teacher_probs: np.ndarray,
    *,
    dev_graphs: list[TypedGraph] | None = None,
    dev_labels: np.ndarray | None = None,
    config: DistillationConfig | None = None,
    tm_config: GraphTMConfig | None = None,
) -> tuple[GraphTMVerifier, dict]:
    """Fit a GraphTMVerifier with soft-label distillation.

    Returns the trained verifier and the per-epoch history.
    """
    cfg = config or DistillationConfig()
    distilled_labels = mix_labels(train_hard_labels, teacher_probs, cfg)
    verifier = GraphTMVerifier(tm_config)
    history = verifier.fit_from_typed_graphs(
        train_graphs,
        distilled_labels,
        dev_graphs=dev_graphs,
        dev_labels=dev_labels,
    )
    history["distillation_config"] = cfg.__dict__
    return verifier, history
