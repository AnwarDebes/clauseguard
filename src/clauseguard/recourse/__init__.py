"""Evidence-graph counterfactual recourse.

For a Refuted (or NotEnoughInfo) verification, what minimal evidence
triples, added, removed, or relation-swapped, would flip the
decision to Supported? This is the EU AI Act Article 14
"effective human oversight" artifact: a concrete pointer to what the
human reviewer would need to fix.

The design mirrors ``graphtm-cbr/graphtm/recourse/`` but the edit
operations work on (s, r, o) triples rather than on RDKit atom/bond
graphs.
"""
from .candidates import (
    EvidenceEdit,
    candidates_from_firing_clauses,
    apply_edit,
    VALID_OPS,
)
from .search import (
    SearchTrace,
    greedy_minimal_evidence_edit,
)
from .output import render_recourse_report

__all__ = [
    "EvidenceEdit",
    "candidates_from_firing_clauses",
    "apply_edit",
    "VALID_OPS",
    "SearchTrace",
    "greedy_minimal_evidence_edit",
    "render_recourse_report",
]
