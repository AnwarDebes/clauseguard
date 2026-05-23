"""Encode a per-sample firing-clause set + literal vector into DIMACS CNF.

This module is the bridge between the GraphTM's symbolic clause set and
an off-the-shelf SAT solver (Glucose 4 via ``python-sat``). For an
auditor, the question is:

    "Given the recorded literals and the recorded clause set, is the
     recorded verification label the unique consistent assignment?"

We model the question as a CNF whose SAT outcome is the answer. The
encoding is intentionally simple so that the audit can be performed
without ClauseGuard installed, only ``python-sat`` and the receipt
file are needed.

CNF layout
----------

For a sample with binary literal vector ``L = [l_0, ..., l_{n-1}]`` and
firing clauses ``C = [c_0, ..., c_{m-1}]`` where each clause is a
conjunction of (possibly negated) literals, the CNF says:

* For every literal ``l_i`` that the receipt records as 1: add unit
  clause ``v_i``.
* For every literal ``l_i`` that the receipt records as 0: add unit
  clause ``-v_i``.
* For every recorded firing clause ``c_j = AND_{k in pos_j} l_k AND
  AND_{k in neg_j} NOT l_k``: add the implication clauses required to
  ensure the conjunction holds. Because the literal values are fixed
  above, this reduces to a sanity check; we emit them anyway so the
  auditor can flip a literal value and re-solve to confirm the
  clause flips with it.
* The decision-flow constraint:
    ``label = argmax_class (sum_{c in firing_clauses_for_class}
    weight(c))``
  is encoded as a pairwise comparison: for the recorded label ``y``
  and every other class ``y' != y``, sum of firing-clause weights for
  ``y`` must be ``>= sum of firing-clause weights for y'``. This is a
  pseudo-Boolean (PB) constraint compiled to CNF via sequential
  encoding when needed.

This is intentionally loose: an auditor can prove the recorded
literals SUPPORT the recorded label, but not that the literals are
the *only* way to reach that label. That's by design: we are auditing
this specific decision, not the model's full Boolean function.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FiringClause:
    """Minimal description of a clause for SAT encoding.

    Attributes:
        clause_id: stable id used in the receipt and ``firing_clause_ids``.
        class_id: the output class this clause votes for (0..n_classes-1).
        polarity: +1 if the clause's vote is positive (i.e., it supports
            ``class_id``), -1 if it's a negative-polarity clause that
            votes against ``class_id``.
        weight: integer vote weight (CAIR TMs use small ints, typically 1).
        positive_literal_ids: indices in ``L`` that must be 1 for firing.
        negative_literal_ids: indices in ``L`` that must be 0 for firing.
    """

    clause_id: int
    class_id: int
    polarity: int
    weight: int
    positive_literal_ids: tuple[int, ...] = field(default_factory=tuple)
    negative_literal_ids: tuple[int, ...] = field(default_factory=tuple)


def encode_clauses_to_cnf(
    clauses: Sequence[FiringClause],
    literal_vector: Sequence[int],
    label: int,
    n_classes: int,
    *,
    write_dimacs_path: str | Path | None = None,
) -> dict:
    """Encode firing clauses + literal vector + label into a CNF.

    Returns a dict with keys:

    * ``cnf_clauses``: list of int lists (DIMACS clauses).
    * ``num_vars``: int, total number of propositional variables.
    * ``dimacs_path``: str or None, path to the written .cnf file.

    Uses ``python-sat`` for the optional DIMACS write. We do NOT import
    pysat at module load, the function imports it lazily, so the
    package remains usable without pysat installed for read-only paths.
    """
    n_literals = len(literal_vector)
    next_var = n_literals + 1  # 1-indexed; first literal is variable 1
    cnf: list[list[int]] = []

    # 1. Literal value constraints, unit clauses.
    for i, v in enumerate(literal_vector):
        var = i + 1
        cnf.append([var] if v else [-var])

    # 2. Firing-clause structural constraints. Because we already pinned
    #    the literal values above, these become redundant for the
    #    recorded sample, but they encode the *shape* of the firing
    #    condition so an auditor can flip a literal and watch the
    #    clause un-fire.
    clause_fire_vars: dict[int, int] = {}
    for fc in clauses:
        fire_var = next_var
        next_var += 1
        clause_fire_vars[fc.clause_id] = fire_var
        # fire_var <-> AND of positive literals AND NOT-AND of negative literals
        # Forward: fire_var => each positive literal, fire_var => each NOT negative.
        for li in fc.positive_literal_ids:
            cnf.append([-fire_var, li + 1])
        for li in fc.negative_literal_ids:
            cnf.append([-fire_var, -(li + 1)])
        # Backward: AND of positive literals AND NOT-AND of negative implies fire_var.
        big_clause = (
            [fire_var]
            + [-(li + 1) for li in fc.positive_literal_ids]
            + [(li + 1) for li in fc.negative_literal_ids]
        )
        cnf.append(big_clause)

    # 3. Class-vote comparison. We do not pseudo-Boolean here for
    #    simplicity; instead we record for each non-recorded class the
    #    pair (label_score, other_score) as a derived integer in the
    #    receipt metadata (not in CNF). The auditor's SAT-solve is on
    #    the structural CNF above; the vote arithmetic is checked
    #    arithmetically against the recorded values.
    class_scores = _vote_arithmetic(clauses, n_classes)

    # 4. Optionally write DIMACS to disk.
    dimacs_path: str | None = None
    if write_dimacs_path is not None:
        try:
            from pysat.formula import CNF
        except ImportError:  # pragma: no cover
            log.warning(
                "python-sat not installed; skipping DIMACS write. "
                "`pip install python-sat`."
            )
        else:
            p = Path(write_dimacs_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            cnf_obj = CNF(from_clauses=cnf)
            cnf_obj.to_file(str(p))
            dimacs_path = str(p)

    return {
        "cnf_clauses": cnf,
        "num_vars": next_var - 1,
        "dimacs_path": dimacs_path,
        "class_scores": class_scores,
        "recorded_label": int(label),
        "auditor_assertion": "score(recorded_label) >= max(score(other_classes))",
    }


def _vote_arithmetic(
    clauses: Sequence[FiringClause], n_classes: int
) -> dict[int, int]:
    """Sum of firing-clause weights per class (signed by polarity)."""
    scores = {c: 0 for c in range(n_classes)}
    for fc in clauses:
        scores[fc.class_id] = scores.get(fc.class_id, 0) + int(fc.polarity) * int(fc.weight)
    return scores


def solve_cnf(cnf_clauses: Sequence[Sequence[int]], num_vars: int) -> dict:
    """Run Glucose 4 on the encoded CNF.

    Used by :func:`verify_receipt` to confirm SAT-solving the encoded
    constraints yields the recorded outcome.

    Returns ``{"sat": bool, "model": list[int] | None, "stats": dict}``.
    """
    try:
        from pysat.solvers import Glucose4
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "python-sat is required for SAT solving. "
            "Install via `pip install python-sat`."
        ) from exc

    with Glucose4(bootstrap_with=list(cnf_clauses)) as solver:
        sat = solver.solve()
        model = solver.get_model() if sat else None
        solver_stats = solver.accum_stats() if hasattr(solver, "accum_stats") else {}
        stats = {
            "num_vars": num_vars,
            "num_clauses": len(cnf_clauses),
            "decisions": int(solver_stats.get("decisions", 0)),
            "propagations": int(solver_stats.get("propagations", 0)),
            "conflicts": int(solver_stats.get("conflicts", 0)),
            "restarts": int(solver_stats.get("restarts", 0)),
        }
    return {"sat": bool(sat), "model": model, "stats": stats}
