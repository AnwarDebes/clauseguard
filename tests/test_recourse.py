"""Tests for the recourse / candidate-generation modules."""
from __future__ import annotations

import pytest

from clauseguard.extraction.triple_extractor import Triple
from clauseguard.recourse.candidates import (
    EvidenceEdit,
    VALID_OPS,
    apply_edit,
    candidates_from_firing_clauses,
)


def _t(s, r, o, pol=1):
    return Triple(subject=s, relation=r, object=o, polarity=pol)


def test_evidence_edit_op_validation():
    with pytest.raises(ValueError):
        EvidenceEdit(op="unknown_op", target_idx=0, new_triple=None)


def test_evidence_edit_add_triple_requires_target_minus_one():
    t = _t("a", "instance_of", "b")
    with pytest.raises(ValueError):
        EvidenceEdit(op="add_triple", target_idx=0, new_triple=t)


def test_evidence_edit_remove_must_be_none_triple():
    t = _t("a", "instance_of", "b")
    with pytest.raises(ValueError):
        EvidenceEdit(op="remove_triple", target_idx=0, new_triple=t)


def test_apply_add_triple():
    base = (_t("a", "instance_of", "b"),)
    edit = EvidenceEdit(op="add_triple", target_idx=-1, new_triple=_t("c", "instance_of", "d"))
    out = apply_edit(base, edit)
    assert len(out) == 2


def test_apply_remove_triple():
    base = (_t("a", "instance_of", "b"), _t("c", "instance_of", "d"))
    edit = EvidenceEdit(op="remove_triple", target_idx=0, new_triple=None)
    out = apply_edit(base, edit)
    assert len(out) == 1
    assert out[0].subject == "c"


def test_apply_swap_relation():
    base = (_t("a", "instance_of", "b"),)
    new = _t("a", "located_in", "b")
    edit = EvidenceEdit(op="swap_relation", target_idx=0, new_triple=new)
    out = apply_edit(base, edit)
    assert out[0].relation == "located_in"


def test_candidates_add_for_uncovered_claim():
    claim = [_t("alice", "instance_of", "researcher")]
    evidence: list = []
    cands = candidates_from_firing_clauses(claim, evidence)
    ops = {c.op for c in cands}
    assert "add_triple" in ops


def test_candidates_swap_relation_when_pair_present_but_relation_differs():
    claim = [_t("drug_x", "treats", "disease_y")]
    evidence = [_t("drug_x", "causes", "disease_y")]
    cands = candidates_from_firing_clauses(claim, evidence)
    assert any(c.op == "swap_relation" for c in cands)


def test_valid_ops_const_complete():
    assert set(VALID_OPS) == {"add_triple", "remove_triple", "swap_relation", "fix_entity_link"}
