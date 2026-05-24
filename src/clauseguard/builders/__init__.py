"""Pluggable evidence-graph builders.

The default evidence-graph builder lives in
:mod:`clauseguard.graphs.evidence_graph` and is exposed here as
:func:`build_default_evidence_graph` so callers can reach both back ends
through a single import.

The subword-dep builder is an optional back end ported from paper-a
(arXiv 2510.XXXXX). It uses BPE subword tokens as nodes and typed
spaCy dependency edges, instead of the entity-level triples used by
the default builder.
"""
from __future__ import annotations

from ..graphs.evidence_graph import build_evidence_graph as build_default_evidence_graph
from .subword_dep_evidence import SubwordDepEvidenceBuilder

__all__ = [
    "build_default_evidence_graph",
    "SubwordDepEvidenceBuilder",
]
