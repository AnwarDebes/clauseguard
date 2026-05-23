"""ClauseGuard: Interpretable verification of LLM outputs via Graph Tsetlin Machines.

A symbolic audit layer that consumes LLM-generated atomic claim triples and
a context knowledge graph and emits per-claim Support/Refute/NotEnoughInfo
labels with human-readable Boolean clauses and SAT-verifiable receipts.

Top-level imports keep the public API short:

    from clauseguard import (
        Triple,
        Claim,
        TypedGraph,
        DualGraphBuilder,
        GraphTMVerifier,
        ClauseReceipt,
    )
"""
from __future__ import annotations

__version__ = "0.1.0"

from .extraction.triple_extractor import Claim, Triple
from .graphs.dual_graph import DualGraphBuilder, TypedGraph
from .tm.graphtm_verifier import GraphTMVerifier
from .verify.certificate import ClauseReceipt

__all__ = [
    "Triple",
    "Claim",
    "TypedGraph",
    "DualGraphBuilder",
    "GraphTMVerifier",
    "ClauseReceipt",
    "__version__",
]
