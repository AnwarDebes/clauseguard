"""Atomic-claim decomposition + (s, r, o) triple extraction.

This module defines the public types ``Triple``, ``Claim``, and the
abstract base classes ``ClaimDecomposer`` and ``TripleExtractor``. Concrete
implementations live in sibling modules:

  * :class:`RegexTripleExtractor` (here), deterministic, dependency-free,
    used in tests and CI runs without GPU. Surface-level only, it will
    miss complex multi-clause sentences.

  * :class:`QwenTripleExtractor` (``qwen_extractor.py``), production
    extractor using Qwen2.5-1.5B-Instruct via HuggingFace. Deterministic
    with temperature 0 and a fixed seed.

Both extractors emit a deterministic, hashable ``tuple[Triple, ...]`` so
the downstream SAT-verifiable receipt is reproducible from inputs alone.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

# --------------------------------------------------------------------------
# Canonical relation vocabulary
#
# Chosen to span the relation types that dominate FEVER / FactKG / HaluEval
# and that map cleanly onto Wikidata properties. Anything outside this
# vocab is bucketed as ``_open_``; the GraphTM still consumes it but
# treats the relation type as a single opaque symbol.
# --------------------------------------------------------------------------
REL_VOCAB: tuple[str, ...] = (
    "instance_of",         # P31
    "subclass_of",          # P279
    "occupation",           # P106
    "located_in",           # P131 / P276
    "country",              # P17
    "born_in",              # P19
    "died_in",              # P20
    "date_of_birth",        # P569
    "date_of_death",        # P570
    "spouse",               # P26
    "parent_of",            # P40 / P22 / P25
    "member_of",            # P463
    "employer",             # P108
    "field_of_work",        # P101
    "educated_at",          # P69
    "author_of",            # P50
    "director_of",          # P57
    "actor_in",             # P161
    "founder_of",           # P112
    "ceo_of",               # P169
    "headquartered_in",     # P159
    "industry",             # P452
    "manufacturer",         # P176
    "publisher",            # P123
    "language",             # P407 / P103
    "religion",             # P140
    "ethnicity",            # P172
    "currency",             # P38
    "capital",              # P36
    "population",           # P1082
    "area",                 # P2046
    "elevation",            # P2044
    "wins",                 # sports
    "score",                # numerical
    "treats",               # medical: drug treats disease
    "causes",               # medical: cause/effect
    "interacts_with",       # medical: drug-drug
    "indicated_for",        # medical
    "contraindicated_for",  # medical
    "dosage",               # medical
    "side_effect",          # medical
    "approved_by",          # regulatory
    "synonym_of",           # lexical
    "antonym_of",           # lexical
    "is_a",                 # generic taxonomic
    "has_property",         # generic
    "has_value",            # generic
    "_open_",               # fallback bucket
)
_REL_INDEX = {r: i for i, r in enumerate(REL_VOCAB)}


# --------------------------------------------------------------------------
# Public types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Triple:
    """A canonicalised (subject, relation, object) triple.

    Attributes:
        subject: canonicalised entity surface form (lower-cased, whitespace-trimmed).
        relation: one of ``REL_VOCAB``; ``_open_`` if not in the closed vocab.
        object: canonicalised entity, literal, or numeric (as string).
        polarity: ``+1`` for asserted, ``-1`` for negated.
        source_span: char offsets ``(start, end)`` into the original text.
        confidence: extractor-reported confidence in ``[0, 1]``.
    """

    subject: str
    relation: str
    object: str
    polarity: int = 1
    source_span: tuple[int, int] = (0, 0)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.polarity not in (-1, 1):
            raise ValueError(f"polarity must be -1 or +1, got {self.polarity}")
        if self.relation not in _REL_INDEX:
            raise ValueError(
                f"relation {self.relation!r} not in REL_VOCAB; "
                f"use canonicalise_relation() before constructing Triple"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def relation_id(self) -> int:
        return _REL_INDEX[self.relation]

    def render(self) -> str:
        """Render to a single human-readable line for clause export.

        Example: ``"Barack Obama --[born_in]-> Honolulu"``.
        Negated triples are wrapped in ``NOT(...)``.
        """
        s = f"{self.subject} --[{self.relation}]-> {self.object}"
        return f"NOT({s})" if self.polarity == -1 else s


@dataclass(frozen=True)
class Claim:
    """A single atomic claim extracted from an LLM output.

    Attributes:
        text: surface text of the atomic claim.
        triples: tuple of canonicalised triples covering this claim.
        extractor_id: short identifier for the extractor that produced
            this claim, used in the audit trail. Includes a SHA prefix
            of the extractor weights when applicable.
    """

    text: str
    triples: tuple[Triple, ...] = field(default_factory=tuple)
    extractor_id: str = "regex@v0.1"


# --------------------------------------------------------------------------
# Canonicalisation helpers
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[\s\.\,\;\:\!\?\"\'\(\)\[\]]+")
_WHITESPACE = re.compile(r"\s+")


def canonicalise_entity(s: str) -> str:
    """Canonical surface form of an entity.

    Lower-cased, leading/trailing punctuation stripped, internal whitespace
    collapsed to single spaces. Numeric literals are kept as-is.
    """
    if s is None:
        return ""
    s = str(s).strip()
    s = _PUNCT.sub(" ", s).strip()
    s = _WHITESPACE.sub(" ", s)
    return s.lower()


def canonicalise_relation(s: str) -> str:
    """Map an arbitrary relation surface form into the closed REL_VOCAB.

    Unknown relations are bucketed as ``_open_`` so the GraphTM can still
    consume them, but downstream clause export marks them visually so a
    human auditor can spot relation-vocab drift.
    """
    if s is None:
        return "_open_"
    key = canonicalise_entity(s).replace(" ", "_")
    if key in _REL_INDEX:
        return key
    # A few obvious surface-form aliases mapped to the canonical relation.
    aliases = {
        "is": "instance_of",
        "type": "instance_of",
        "kind_of": "subclass_of",
        "in": "located_in",
        "from": "born_in",
        "married_to": "spouse",
        "child_of": "parent_of",
        "works_for": "employer",
        "works_at": "employer",
        "studied_at": "educated_at",
        "wrote": "author_of",
        "directed": "director_of",
        "acted_in": "actor_in",
        "founded": "founder_of",
        "ceo": "ceo_of",
        "located": "located_in",
        "capital_of": "capital",
        "treats_for": "treats",
        "causes_of": "causes",
        "indicated": "indicated_for",
        "contraindicated": "contraindicated_for",
        "approves": "approved_by",
    }
    return aliases.get(key, "_open_")


# --------------------------------------------------------------------------
# Abstract bases
# --------------------------------------------------------------------------

class ClaimDecomposer(ABC):
    """Decompose an LLM output string into a list of atomic claims."""

    @abstractmethod
    def decompose(self, llm_output: str) -> list[Claim]: ...

    @property
    def id(self) -> str:
        """Extractor identifier used in audit trails."""
        return type(self).__name__.lower()


class TripleExtractor(ABC):
    """Extract a tuple of canonicalised triples from a claim string."""

    @abstractmethod
    def extract(self, claim_text: str) -> tuple[Triple, ...]: ...

    @property
    def id(self) -> str:
        return type(self).__name__.lower()


# --------------------------------------------------------------------------
# RegexTripleExtractor, dependency-free fallback
# --------------------------------------------------------------------------

class RegexTripleExtractor(TripleExtractor):
    """Deterministic, dependency-free extractor for tests and small slices.

    Handles surface patterns of the form ``X is Y``, ``X was born in Y``,
    ``X is located in Y``, ``X is a Y``, ``X did not Y``, etc. Will miss
    complex multi-clause sentences. Use :class:`QwenTripleExtractor` for
    production runs.
    """

    _PATTERNS: tuple[tuple[re.Pattern[str], str, bool], ...] = (
        (re.compile(r"\b(.+?)\s+is\s+a\s+(.+)", re.I), "instance_of", False),
        (re.compile(r"\b(.+?)\s+is\s+an\s+(.+)", re.I), "instance_of", False),
        (re.compile(r"\b(.+?)\s+was\s+born\s+in\s+(.+)", re.I), "born_in", False),
        (re.compile(r"\b(.+?)\s+died\s+in\s+(.+)", re.I), "died_in", False),
        (re.compile(r"\b(.+?)\s+is\s+located\s+in\s+(.+)", re.I), "located_in", False),
        (re.compile(r"\b(.+?)\s+is\s+in\s+(.+)", re.I), "located_in", False),
        (re.compile(r"\b(.+?)\s+is\s+the\s+capital\s+of\s+(.+)", re.I), "capital", False),
        (re.compile(r"\b(.+?)\s+is\s+married\s+to\s+(.+)", re.I), "spouse", False),
        (re.compile(r"\b(.+?)\s+wrote\s+(.+)", re.I), "author_of", False),
        (re.compile(r"\b(.+?)\s+directed\s+(.+)", re.I), "director_of", False),
        (re.compile(r"\b(.+?)\s+founded\s+(.+)", re.I), "founder_of", False),
        (re.compile(r"\b(.+?)\s+is\s+the\s+ceo\s+of\s+(.+)", re.I), "ceo_of", False),
        (re.compile(r"\b(.+?)\s+treats\s+(.+)", re.I), "treats", False),
        (re.compile(r"\b(.+?)\s+causes\s+(.+)", re.I), "causes", False),
        (re.compile(r"\b(.+?)\s+is\s+not\s+(.+)", re.I), "instance_of", True),
        (re.compile(r"\b(.+?)\s+does\s+not\s+(.+)", re.I), "_open_", True),
        (re.compile(r"\b(.+?)\s+did\s+not\s+(.+)", re.I), "_open_", True),
    )

    def extract(self, claim_text: str) -> tuple[Triple, ...]:
        out: list[Triple] = []
        text = claim_text.strip()
        for pat, rel, negated in self._PATTERNS:
            m = pat.match(text)
            if not m:
                continue
            subj = canonicalise_entity(m.group(1))
            obj = canonicalise_entity(m.group(2))
            if not subj or not obj:
                continue
            out.append(
                Triple(
                    subject=subj,
                    relation=rel,
                    object=obj,
                    polarity=-1 if negated else 1,
                    source_span=(m.start(), m.end()),
                    confidence=0.7,
                )
            )
            break  # first matching pattern only, surface-level extractor
        return tuple(out)

    @property
    def id(self) -> str:
        return "regex@v0.1"


# --------------------------------------------------------------------------
# Convenience: sentence-splitting decomposer using RegexTripleExtractor
# --------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z])")


class SimpleClaimDecomposer(ClaimDecomposer):
    """Sentence-level claim decomposer.

    Splits on sentence-final punctuation followed by a capital letter, then
    extracts triples for each sentence. Useful for tests; the production
    decomposer is :class:`QwenClaimDecomposer` which uses an LLM to break
    complex sentences into atomic propositions.
    """

    def __init__(self, extractor: TripleExtractor | None = None) -> None:
        self.extractor = extractor or RegexTripleExtractor()

    def decompose(self, llm_output: str) -> list[Claim]:
        sentences = _SENT_SPLIT.split(llm_output.strip()) if llm_output.strip() else []
        out: list[Claim] = []
        for s in sentences:
            s = s.strip().rstrip(".!?")
            if not s:
                continue
            triples = self.extractor.extract(s)
            out.append(Claim(text=s, triples=triples, extractor_id=self.extractor.id))
        return out

    @property
    def id(self) -> str:
        return f"simple+{self.extractor.id}"


def deduplicate_triples(triples: Iterable[Triple]) -> tuple[Triple, ...]:
    """Stable de-duplication of triples by (subject, relation, object, polarity)."""
    seen: set[tuple[str, str, str, int]] = set()
    out: list[Triple] = []
    for t in triples:
        key = (t.subject, t.relation, t.object, t.polarity)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return tuple(out)
