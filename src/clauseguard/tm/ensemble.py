"""K-seed GraphTMVerifier ensemble.

Mirrors the K=5 ensemble in ``graphtm-cbr``: train ``K`` independent
verifiers with different seeds, predict via majority vote on hard
labels and average vote-sums on soft scores. The receipts and recourse
are computed per-member and aggregated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..graphs.dual_graph import TypedGraph
from .graphtm_verifier import GraphTMVerifier


@dataclass
class EnsembleVerifier:
    members: list[GraphTMVerifier]

    @property
    def k(self) -> int:
        return len(self.members)

    def predict(self, graphs: list[TypedGraph]) -> np.ndarray:
        votes = np.stack([m.predict_from_typed_graphs(graphs) for m in self.members])
        # majority vote, ties broken by lowest label id (deterministic).
        n_classes = self.members[0].config.n_classes
        counts = np.zeros((votes.shape[1], n_classes), dtype=np.int64)
        for k in range(votes.shape[0]):
            for i, lab in enumerate(votes[k]):
                counts[i, lab] += 1
        return counts.argmax(axis=1)

    def class_sums(self, graphs: list[TypedGraph]) -> np.ndarray:
        sums = np.stack([m.class_sums_from_typed_graphs(graphs) for m in self.members])
        # Average across members; shape ``[n, n_classes]``.
        return sums.mean(axis=0)

    def version_id(self) -> str:
        from ..utils.io import sha256_of

        members_meta = [m.version_id() for m in self.members]
        return f"ensemble-k{self.k}@{sha256_of(members_meta)[:12]}"
