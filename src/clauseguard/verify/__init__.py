"""SAT-verifiable clause receipts.

Two public types:

* :class:`ClauseReceipt`, the signed audit artifact attached to every
  ClauseGuard decision.
* :func:`build_receipt` / :func:`verify_receipt`, round-trip helpers.

The encoded receipt CNF is written to disk under
``results/receipts/<receipt_id>.cnf``; an auditor running Glucose 4 on
that file must reach the same satisfiability outcome the receipt
records, otherwise the receipt is rejected.
"""
from .certificate import (
    ClauseReceipt,
    build_receipt,
    sign,
    verify_receipt,
    verify_signature,
)
from .sat_encoder import encode_clauses_to_cnf

__all__ = [
    "ClauseReceipt",
    "build_receipt",
    "verify_receipt",
    "verify_signature",
    "sign",
    "encode_clauses_to_cnf",
]
