"""Candidate evidence-graph edit generation for recourse.

For a Refuted prediction whose firing clauses contain literals pointing
at evidence-side nodes / relations, the candidate generator emits a
bounded set of :class:`EvidenceEdit` operations whose application could
plausibly flip the verification label.

Mirrors ``graphtm-cbr/graphtm/recourse/candidates.py`` but the edit
universe is now triple-level (add, remove, swap relation, fix entity
link) rather than atom/bond-level.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..extraction.triple_extractor import (
    REL_VOCAB,
    Triple,
    canonicalise_entity,
    canonicalise_relation,
)


VALID_OPS: tuple[str, ...] = (
    "add_triple",
    "remove_triple",
    "swap_relation",
    "fix_entity_link",
)


# --------------------------------------------------------------------------
# EvidenceEdit
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceEdit:
    """A single edit on the evidence-graph side.

    Operations:

    * ``add_triple``: insert a new evidence triple.
    * ``remove_triple``: drop an evidence triple.
    * ``swap_relation``: change the relation type on an evidence triple.
    * ``fix_entity_link``: replace one entity surface form on an
      evidence triple (typical use: a typo or NER error in evidence
      extraction).

    Attributes:
        op: one of :data:`VALID_OPS`.
        target_idx: index of the evidence triple being edited (or
            ``-1`` for ``add_triple``).
        new_triple: the resulting triple after the edit; ``None`` for
            ``remove_triple``.
        rationale: short string explaining why this edit was generated
            (e.g., "Refuted because clause C42 requires relation
            'treats' between drug X and disease Y; evidence had 'causes'").
    """

    op: str
    target_idx: int
    new_triple: Triple | None
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.op not in VALID_OPS:
            raise ValueError(f"unknown op {self.op!r}; expected one of {VALID_OPS}")
        if self.op == "remove_triple" and self.new_triple is not None:
            raise ValueError("remove_triple must set new_triple=None")
        if self.op == "add_triple" and self.target_idx != -1:
            raise ValueError("add_triple must use target_idx=-1")
        if self.op != "remove_triple" and self.new_triple is None:
            raise ValueError(f"{self.op} requires new_triple")


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

def apply_edit(
    triples: Sequence[Triple],
    edit: EvidenceEdit,
) -> tuple[Triple, ...]:
    """Apply a single edit to a tuple of evidence triples.

    Returns a new tuple; the input is not mutated.
    """
    triples = list(triples)
    if edit.op == "add_triple":
        if edit.new_triple is None:
            raise ValueError("add_triple requires new_triple")
        triples.append(edit.new_triple)
    elif edit.op == "remove_triple":
        if 0 <= edit.target_idx < len(triples):
            triples.pop(edit.target_idx)
    elif edit.op in ("swap_relation", "fix_entity_link"):
        if 0 <= edit.target_idx < len(triples) and edit.new_triple is not None:
            triples[edit.target_idx] = edit.new_triple
    return tuple(triples)


# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------

def candidates_from_firing_clauses(
    claim_triples: Sequence[Triple],
    evidence_triples: Sequence[Triple],
    *,
    target_label: int = 0,
    max_candidates: int = 50,
) -> list[EvidenceEdit]:
    """Generate a bounded set of candidate evidence edits.

    Strategy (mirrors graphtm-cbr's clause-walk + literal-attribution):

    1. For each claim triple ``(s, r, o)`` not corroborated by any
       evidence triple, emit an ``add_triple`` candidate that would
       provide that corroboration.
    2. For each evidence triple ``(s', r', o')`` that contradicts a
       claim triple by relation type or polarity, emit a
       ``swap_relation`` candidate that aligns it with the claim.
    3. For each evidence triple whose subject/object is similar (but
       not identical) to a claim entity surface form, emit a
       ``fix_entity_link`` candidate that aligns the entity.
    4. For each evidence triple with a relation outside :data:`REL_VOCAB`
       (i.e., the open-bucket fallback), emit a ``swap_relation``
       candidate for each plausible canonical relation.

    Bounded by ``max_candidates``. Order is deterministic.
    """
    out: list[EvidenceEdit] = []
    evidence_by_pair: dict[tuple[str, str], list[int]] = {}
    for i, t in enumerate(evidence_triples):
        evidence_by_pair.setdefault((t.subject, t.object), []).append(i)
    evidence_subjects = {t.subject for t in evidence_triples}
    evidence_objects = {t.object for t in evidence_triples}

    # 1. Add corroborating triples for uncovered claim triples.
    for ct in claim_triples:
        if (ct.subject, ct.object) in evidence_by_pair:
            # already mentioned (even if relation differs), covered by case 2 below
            continue
        out.append(
            EvidenceEdit(
                op="add_triple",
                target_idx=-1,
                new_triple=ct,
                rationale=f"add evidence corroborating claim triple {ct.render()}",
            )
        )

    # 2. Swap relation on contradicting evidence.
    for ct in claim_triples:
        for ei in evidence_by_pair.get((ct.subject, ct.object), []):
            et = evidence_triples[ei]
            if et.relation == ct.relation and et.polarity == ct.polarity:
                continue
            new_t = replace(et, relation=ct.relation, polarity=ct.polarity)
            out.append(
                EvidenceEdit(
                    op="swap_relation",
                    target_idx=ei,
                    new_triple=new_t,
                    rationale=(
                        f"align evidence relation {et.relation} -> {ct.relation} "
                        f"between {ct.subject!r} and {ct.object!r}"
                    ),
                )
            )

    # 3. Fix entity links for near-match surface forms.
    claim_entities = {t.subject for t in claim_triples} | {t.object for t in claim_triples}
    for ei, et in enumerate(evidence_triples):
        for ce in claim_entities:
            if not ce:
                continue
            if ce == et.subject or ce == et.object:
                continue
            if _surface_similar(ce, et.subject):
                new_t = replace(et, subject=ce)
                out.append(
                    EvidenceEdit(
                        op="fix_entity_link",
                        target_idx=ei,
                        new_triple=new_t,
                        rationale=(
                            f"link evidence subject {et.subject!r} -> {ce!r}"
                        ),
                    )
                )
            if _surface_similar(ce, et.object):
                new_t = replace(et, object=ce)
                out.append(
                    EvidenceEdit(
                        op="fix_entity_link",
                        target_idx=ei,
                        new_triple=new_t,
                        rationale=(
                            f"link evidence object {et.object!r} -> {ce!r}"
                        ),
                    )
                )

    # 4. Promote _open_ relations on evidence to plausible canonical ones
    #    if a claim mentions that subject-object pair with a canonical
    #    relation.
    for ei, et in enumerate(evidence_triples):
        if et.relation != "_open_":
            continue
        for ct in claim_triples:
            if (ct.subject, ct.object) != (et.subject, et.object):
                continue
            if ct.relation == "_open_":
                continue
            new_t = replace(et, relation=ct.relation, polarity=ct.polarity)
            out.append(
                EvidenceEdit(
                    op="swap_relation",
                    target_idx=ei,
                    new_triple=new_t,
                    rationale=(
                        f"promote evidence relation _open_ -> {ct.relation}"
                    ),
                )
            )
    # Deduplicate by (op, target_idx, rendered new_triple).
    seen: set[tuple[str, int, str]] = set()
    deduped: list[EvidenceEdit] = []
    for e in out:
        key = (e.op, e.target_idx, e.new_triple.render() if e.new_triple else "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
        if len(deduped) >= max_candidates:
            break
    return deduped


def _surface_similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """Cheap surface similarity for entity-link candidates.

    Uses normalised edit distance via Python's built-in
    ``difflib.SequenceMatcher``. Threshold tuned to flag obvious typos
    while avoiding random near-matches.
    """
    if not a or not b:
        return False
    if a == b:
        return False
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio() >= threshold
