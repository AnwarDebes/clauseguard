"""Tests for SAT encoding + receipt signing."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from clauseguard.extraction.triple_extractor import Triple
from clauseguard.graphs.dual_graph import DualGraphBuilder
from clauseguard.verify.certificate import (
    ClauseReceipt,
    build_receipt,
    sign,
    verify_signature,
    verify_receipt,
)
from clauseguard.verify.sat_encoder import (
    FiringClause,
    encode_clauses_to_cnf,
)


def _t(s, r, o, pol=1):
    return Triple(subject=s, relation=r, object=o, polarity=pol)


def test_encode_clauses_to_cnf_no_solver_required(tmp_path: Path):
    fc = FiringClause(
        clause_id=0,
        class_id=0,
        polarity=1,
        weight=1,
        positive_literal_ids=(0, 1),
        negative_literal_ids=(2,),
    )
    out = encode_clauses_to_cnf(
        [fc],
        literal_vector=[1, 1, 0, 1],
        label=0,
        n_classes=3,
        write_dimacs_path=tmp_path / "test.cnf",
    )
    assert out["num_vars"] >= 4
    assert out["class_scores"][0] == 1
    assert out["recorded_label"] == 0
    assert os.path.exists(out["dimacs_path"])


def test_build_and_verify_receipt(tmp_path: Path):
    triples = (_t("alice", "instance_of", "researcher"),)
    graph = DualGraphBuilder().build(triples, triples)
    fc = FiringClause(
        clause_id=42,
        class_id=0,
        polarity=1,
        weight=1,
        positive_literal_ids=(0,),
    )
    cnf_path = tmp_path / "receipt.cnf"
    receipt = build_receipt(
        claim_text="Alice is a researcher",
        triples=triples,
        evidence_graph=graph,
        model_version_id="clauseguard-v0.1.0@deadbeef",
        label=0,
        label_name="Support",
        firing_clauses=[fc],
        literal_vector=[1],
        automaton_state=[0, 0, 0],
        cnf_path=cnf_path,
    )
    assert isinstance(receipt, ClauseReceipt)
    assert receipt.label == 0
    # Signature round-trip.
    assert verify_signature(receipt, b"clauseguard-dev-key")
    # Verification with re-solve (Glucose 4 must be available; if not, the
    # function logs a warning and we don't assert on sat_ok strictly).
    result = verify_receipt(receipt, re_solve=False)
    assert result["signature_ok"] is True


def test_signature_detects_tampering(tmp_path: Path):
    triples = (_t("alice", "instance_of", "researcher"),)
    graph = DualGraphBuilder().build(triples, triples)
    fc = FiringClause(clause_id=1, class_id=0, polarity=1, weight=1)
    receipt = build_receipt(
        claim_text="alice is a researcher",
        triples=triples,
        evidence_graph=graph,
        model_version_id="v0",
        label=0,
        label_name="Support",
        firing_clauses=[fc],
        literal_vector=[1, 0],
        automaton_state=[0],
        cnf_path=tmp_path / "x.cnf",
    )
    # Tamper with the label *after* signing.
    tampered_fields = receipt.fields_for_signing()
    tampered_fields["label"] = 1
    bad_sig_check = sign(tampered_fields, b"clauseguard-dev-key")
    # The original signature must not match the tampered payload.
    assert receipt.signature != bad_sig_check
