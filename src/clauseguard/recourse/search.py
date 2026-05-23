"""Greedy minimum-edit counterfactual search on the evidence graph.

For a sample whose verifier outputs ``current_label != target_label``,
search a bounded sequence of :class:`EvidenceEdit` operations that flip
the verifier to ``target_label``. Mirrors
``graphtm-cbr/graphtm/recourse/search.py:greedy_minimal_edit`` exactly
in structure; the only domain change is that the edits operate on
triples instead of RDKit RWMol objects.

Why greedy and not BFS / branch-and-bound: the project-wide reliability
rule says "no BFS". Bounded candidate set + greedy descent gives
sub-second latency per sample even on CPU.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..extraction.triple_extractor import Triple
from ..graphs.dual_graph import DualGraphBuilder, TypedGraph
from .candidates import EvidenceEdit, apply_edit, candidates_from_firing_clauses


# --------------------------------------------------------------------------
# SearchTrace
# --------------------------------------------------------------------------

@dataclass
class SearchTrace:
    """Bookkeeping for a single recourse call."""

    applied_edits: list[EvidenceEdit] = field(default_factory=list)
    scores_before: list[np.ndarray] = field(default_factory=list)
    scores_after: list[np.ndarray] = field(default_factory=list)
    candidates_tried: int = 0
    candidates_rejected: int = 0
    flipped: bool = False
    total_latency_ms: float = 0.0
    steps_latency_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "applied_edits": [
                {
                    "op": e.op,
                    "target_idx": e.target_idx,
                    "new_triple": (
                        e.new_triple.render() if e.new_triple is not None else None
                    ),
                    "rationale": e.rationale,
                }
                for e in self.applied_edits
            ],
            "candidates_tried": self.candidates_tried,
            "candidates_rejected": self.candidates_rejected,
            "flipped": self.flipped,
            "total_latency_ms": self.total_latency_ms,
            "steps_latency_ms": list(self.steps_latency_ms),
        }


# --------------------------------------------------------------------------
# Scoring helper
# --------------------------------------------------------------------------

def _score_margin(scores: np.ndarray, target_label: int) -> float:
    """Margin in favour of the target label: ``score(target) - max(others)``."""
    target_score = float(scores[target_label])
    other = float(scores[[i for i in range(len(scores)) if i != target_label]].max())
    return target_score - other


# --------------------------------------------------------------------------
# Greedy search
# --------------------------------------------------------------------------

def greedy_minimal_evidence_edit(
    *,
    verifier: Any,
    builder: DualGraphBuilder,
    claim_triples: Sequence[Triple],
    evidence_triples: Sequence[Triple],
    target_label: int = 0,
    max_edits: int = 3,
    max_candidates: int = 50,
    candidate_factory: Callable[
        [Sequence[Triple], Sequence[Triple]], list[EvidenceEdit]
    ]
    | None = None,
) -> tuple[list[EvidenceEdit], bool, SearchTrace]:
    """Search a sequence of evidence-graph edits that flips the verifier to ``target_label``.

    The verifier may be a single :class:`GraphTMVerifier` or an
    :class:`EnsembleVerifier`; we only require ``class_sums_from_typed_graphs``
    (one of the two existing methods).

    Returns ``(applied_edits, flipped, trace)``.
    """
    trace = SearchTrace()
    t_total = time.perf_counter()

    current_evidence = tuple(evidence_triples)
    current_graph = builder.build(claim_triples, current_evidence)
    current_scores = _scores_for(verifier, [current_graph])[0]
    trace.scores_before.append(np.asarray(current_scores))

    if (
        int(np.argmax(current_scores)) == target_label
        and _score_margin(current_scores, target_label) > 0
    ):
        trace.flipped = True
        trace.total_latency_ms = (time.perf_counter() - t_total) * 1000.0
        return list(trace.applied_edits), True, trace

    for step in range(max_edits):
        t_step = time.perf_counter()
        if candidate_factory is None:
            candidates = candidates_from_firing_clauses(
                claim_triples=claim_triples,
                evidence_triples=current_evidence,
                target_label=target_label,
                max_candidates=max_candidates,
            )
        else:
            candidates = candidate_factory(claim_triples, current_evidence)
        if not candidates:
            break

        best_edit: EvidenceEdit | None = None
        best_scores: np.ndarray | None = None
        best_margin = -np.inf

        for edit in candidates:
            trace.candidates_tried += 1
            new_evidence = apply_edit(current_evidence, edit)
            new_graph = builder.build(claim_triples, new_evidence)
            new_scores = _scores_for(verifier, [new_graph])[0]
            margin = _score_margin(new_scores, target_label)
            if margin > best_margin:
                best_margin = margin
                best_edit = edit
                best_scores = new_scores
        if best_edit is None:
            trace.candidates_rejected += len(candidates)
            break

        # Commit the best edit.
        trace.applied_edits.append(best_edit)
        current_evidence = apply_edit(current_evidence, best_edit)
        current_scores = best_scores
        trace.scores_after.append(np.asarray(best_scores))
        trace.steps_latency_ms.append((time.perf_counter() - t_step) * 1000.0)

        if int(np.argmax(current_scores)) == target_label and best_margin > 0:
            trace.flipped = True
            break

    trace.total_latency_ms = (time.perf_counter() - t_total) * 1000.0
    return list(trace.applied_edits), trace.flipped, trace


def _scores_for(verifier: Any, graphs: list[TypedGraph]) -> np.ndarray:
    """Return shape ``[n_graphs, n_classes]`` scores."""
    if hasattr(verifier, "class_sums"):
        sums = verifier.class_sums(graphs)
    else:
        sums = verifier.class_sums_from_typed_graphs(graphs)
    return np.asarray(sums)
