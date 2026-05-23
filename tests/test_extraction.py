"""Unit tests for the extraction module."""
from __future__ import annotations

import pytest

from clauseguard.extraction.triple_extractor import (
    REL_VOCAB,
    Claim,
    RegexTripleExtractor,
    SimpleClaimDecomposer,
    Triple,
    canonicalise_entity,
    canonicalise_relation,
    deduplicate_triples,
)


def test_triple_immutable():
    t = Triple(subject="alice", relation="instance_of", object="researcher")
    with pytest.raises(Exception):
        t.subject = "bob"


def test_triple_polarity_validation():
    with pytest.raises(ValueError):
        Triple(subject="a", relation="instance_of", object="b", polarity=2)


def test_triple_relation_validation():
    with pytest.raises(ValueError):
        Triple(subject="a", relation="bogus_relation", object="b")


def test_triple_render_negation():
    t = Triple(subject="a", relation="instance_of", object="b", polarity=-1)
    assert t.render().startswith("NOT(")


def test_canonicalise_entity_collapses_whitespace_and_lowercases():
    assert canonicalise_entity("  Barack  Obama  ") == "barack obama"


def test_canonicalise_relation_known():
    assert canonicalise_relation("born_in") == "born_in"


def test_canonicalise_relation_alias():
    assert canonicalise_relation("married to") == "spouse"


def test_canonicalise_relation_open_bucket():
    assert canonicalise_relation("xyzzy") == "_open_"


def test_regex_extractor_simple_is_a():
    e = RegexTripleExtractor()
    out = e.extract("Barack Obama is a politician")
    assert len(out) == 1
    assert out[0].subject == "barack obama"
    assert out[0].relation == "instance_of"
    assert out[0].object == "politician"
    assert out[0].polarity == 1


def test_regex_extractor_negation():
    e = RegexTripleExtractor()
    out = e.extract("the moon is not a planet")
    assert len(out) == 1
    assert out[0].polarity == -1


def test_simple_decomposer_splits_sentences():
    dec = SimpleClaimDecomposer()
    out = dec.decompose("Alice is a researcher. Bob is a clinician.")
    assert len(out) == 2
    assert out[0].text.startswith("Alice")
    assert out[1].text.startswith("Bob")


def test_deduplicate_triples():
    t1 = Triple(subject="a", relation="instance_of", object="b")
    t2 = Triple(subject="a", relation="instance_of", object="b")  # dup
    t3 = Triple(subject="a", relation="instance_of", object="c")
    out = deduplicate_triples([t1, t2, t3])
    assert len(out) == 2


def test_rel_vocab_contains_open():
    assert "_open_" in REL_VOCAB
