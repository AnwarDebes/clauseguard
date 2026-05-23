"""Build a typed evidence-side graph from evidence triples.

Identical structure to :mod:`claim_graph` but the resulting graph is
labelled by ``source = 1`` (evidence) in the dual graph. Kept as a
separate function for clarity and to leave room for evidence-specific
preprocessing (e.g., merging multiple sentences from the same Wikipedia
page).
"""
from __future__ import annotations

from collections.abc import Sequence

from ..extraction.triple_extractor import Triple
from .claim_graph import build_claim_graph


def build_evidence_graph(
    triples: Sequence[Triple],
) -> tuple[tuple[str, ...], list[tuple[int, int, str]], list[int]]:
    """Shape identical to :func:`build_claim_graph`.

    Kept as a thin wrapper so future evidence-specific preprocessing
    (sentence merging, redundancy collapse, evidence-weight features)
    has a clear extension point.
    """
    return build_claim_graph(triples)
