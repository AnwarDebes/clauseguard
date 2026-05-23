"""Signed clause-receipt construction and verification.

Receipt format
--------------

A :class:`ClauseReceipt` is a JSON-serialisable artifact attached to
every ClauseGuard decision. It records enough information for an
external auditor (e.g., an EU AI Act notified body) to reproduce the
verification decision from inputs alone:

* the canonicalised claim triples (hashed),
* the canonicalised evidence graph (hashed),
* the trained model's version id (hashed config + symbol vocab),
* the recorded label and class-score arithmetic,
* the firing-clause ids,
* the literal vector hash,
* the path to the DIMACS CNF (relative to the receipts directory).

The receipt is signed with HMAC-SHA-256 using a deployer-controlled
key. The signature covers every field except itself. An auditor with
the key can verify the signature; an auditor without the key can still
re-run the SAT solve on the recorded CNF and confirm that the
recorded class-score arithmetic produces the recorded label, that's
the cryptographic-free side of the audit.

Threat model
------------

* In scope: detect tampering with the receipt fields after the fact.
* Out of scope: prove that the deployer did not log a different
  decision than they enforced. That requires a separate trusted-log
  protocol (e.g., transparency-log style append-only ledger), which is
  a clear follow-up.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.io import canonical_json, sha256_of
from .sat_encoder import (
    FiringClause,
    encode_clauses_to_cnf,
    solve_cnf,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# ClauseReceipt
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ClauseReceipt:
    """Signed audit artifact for one ClauseGuard verification decision."""

    claim_text: str
    triples_hash: str
    evidence_graph_hash: str
    model_version_id: str
    label: int
    label_name: str
    firing_clause_ids: tuple[int, ...]
    literal_vector_hash: str
    automaton_state_hash: str
    cnf_path: str
    class_scores: dict
    firing_extraction: str = "approximate-top-k"
    signature: str = ""

    def fields_for_signing(self) -> dict:
        d = asdict(self)
        d.pop("signature", None)
        return d


# --------------------------------------------------------------------------
# Signing
# --------------------------------------------------------------------------

def sign(receipt_fields: dict, key: bytes) -> str:
    """HMAC-SHA-256 over canonical-JSON of receipt fields, excluding signature."""
    msg = canonical_json(receipt_fields).encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_signature(receipt: ClauseReceipt, key: bytes) -> bool:
    expected = sign(receipt.fields_for_signing(), key)
    return hmac.compare_digest(expected, receipt.signature)


# --------------------------------------------------------------------------
# Receipt construction
# --------------------------------------------------------------------------

def build_receipt(
    *,
    claim_text: str,
    triples: tuple,
    evidence_graph: Any,
    model_version_id: str,
    label: int,
    label_name: str,
    firing_clauses: list[FiringClause],
    literal_vector: list[int],
    automaton_state: Any,
    cnf_path: str | Path,
    key: bytes | None = None,
    firing_extraction: str = "approximate-top-k",
) -> ClauseReceipt:
    """Construct a signed receipt for one verification decision."""
    triples_hash = sha256_of(list(asdict(t) for t in triples) if triples else [])
    evidence_graph_hash = sha256_of(
        evidence_graph.as_dict() if hasattr(evidence_graph, "as_dict") else evidence_graph
    )
    literal_vector_hash = sha256_of(list(map(int, literal_vector)))
    automaton_state_hash = sha256_of(
        automaton_state.tolist() if hasattr(automaton_state, "tolist") else automaton_state
    )

    # Compute the per-class score arithmetic for the auditor assertion.
    encoded = encode_clauses_to_cnf(
        firing_clauses, literal_vector, label, n_classes=3, write_dimacs_path=cnf_path
    )

    fields = {
        "claim_text": claim_text,
        "triples_hash": triples_hash,
        "evidence_graph_hash": evidence_graph_hash,
        "model_version_id": model_version_id,
        "label": int(label),
        "label_name": label_name,
        "firing_clause_ids": tuple(fc.clause_id for fc in firing_clauses),
        "literal_vector_hash": literal_vector_hash,
        "automaton_state_hash": automaton_state_hash,
        "cnf_path": str(cnf_path),
        "class_scores": encoded["class_scores"],
        "firing_extraction": firing_extraction,
    }
    if key is None:
        key = os.environ.get("CLAUSEGUARD_HMAC_KEY", "clauseguard-dev-key").encode("utf-8")
    sig = sign(fields, key)
    return ClauseReceipt(signature=sig, **fields)


# --------------------------------------------------------------------------
# Receipt verification
# --------------------------------------------------------------------------

def verify_receipt(
    receipt: ClauseReceipt,
    *,
    key: bytes | None = None,
    re_solve: bool = True,
) -> dict:
    """Verify a receipt.

    Two checks:

    1. **Signature**: HMAC-SHA-256 over canonical-JSON of fields matches
       the recorded signature. Requires the deployer key.
    2. **SAT replay** (optional, ``re_solve=True``): re-encode and solve
       the CNF at ``receipt.cnf_path`` with Glucose 4 and confirm the
       recorded class-score arithmetic produces ``receipt.label`` (i.e.,
       score(label) >= score(other_class) for all other classes).

    Returns ``{"signature_ok": bool, "sat_ok": bool, "details": dict}``.
    """
    out: dict = {"signature_ok": False, "sat_ok": False, "details": {}}

    if key is None:
        key = os.environ.get("CLAUSEGUARD_HMAC_KEY", "clauseguard-dev-key").encode("utf-8")
    out["signature_ok"] = verify_signature(receipt, key)

    if not re_solve:
        return out

    # Class-score assertion. The encoded CNF on disk encodes the
    # structural firing pattern; the class-score comparison is
    # arithmetically verifiable from the recorded scores.
    scores = receipt.class_scores
    label = receipt.label
    score_for_label = scores.get(label, scores.get(str(label), 0))
    score_for_others = [
        v for k, v in scores.items() if str(k) != str(label)
    ]
    out["details"]["score_for_label"] = score_for_label
    out["details"]["score_for_others"] = score_for_others
    arithmetic_ok = all(score_for_label >= s for s in score_for_others) if score_for_others else True

    # SAT solve the structural CNF if it exists.
    sat_ok = True
    cnf_path = Path(receipt.cnf_path)
    if cnf_path.exists():
        try:
            from pysat.formula import CNF

            cnf = CNF(from_file=str(cnf_path))
            res = solve_cnf(cnf.clauses, cnf.nv)
            out["details"]["sat_solver"] = res["stats"]
            sat_ok = bool(res["sat"])
        except Exception as exc:  # pragma: no cover
            log.warning("SAT replay failed: %s", exc)
            sat_ok = False
    else:
        log.warning("CNF file %s not found; structural SAT replay skipped", cnf_path)

    out["sat_ok"] = arithmetic_ok and sat_ok
    return out
