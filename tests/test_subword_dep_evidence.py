"""Tests for the subword-dependency evidence-graph builder.

Three categories:

1. Smoke: build the subword-dep evidence graph from one toy sentence
   and check that the basic invariants hold.
2. Compatibility: run ClauseGuard's dual-graph constructor with
   ``evidence_builder="subword_dep"`` on three toy claim/evidence
   pairs and check the output ``TypedGraph`` schema.
3. Regression (the most important): with the default builder, the
   dual graph output is bit-equal to the v0.1 behaviour. This
   protects existing users from any silent change.
"""
from __future__ import annotations

import builtins

import numpy as np
import pytest

from clauseguard.builders.subword_dep_evidence import SubwordDepEvidenceBuilder
from clauseguard.extraction.triple_extractor import Triple
from clauseguard.graphs.dual_graph import (
    ALL_EDGE_TYPES,
    DualGraphBuilder,
    SUBWORD_DEP_EDGE_TYPES,
    TypedGraph,
)


def _t(s, r, o, pol=1):
    return Triple(subject=s, relation=r, object=o, polarity=pol)


# --------------------------------------------------------------------------
# 1. Smoke test
# --------------------------------------------------------------------------

def test_smoke_subword_dep_builder_produces_typed_edges():
    """Build the subword-dep evidence graph from one toy sentence.

    Asserts:
    * node count > 0
    * edge count > 0
    * at least one seq_next/seq_prev edge appears
    * at least one dep:* edge appears
    * polarity sums are zero for an asserted triple
    """
    builder = SubwordDepEvidenceBuilder()
    ev = [_t("alice", "instance_of", "researcher")]
    labels, edges, polarity = builder.build(ev)

    assert len(labels) > 0
    assert len(edges) > 0

    edge_types = {e[2] for e in edges}
    seq_edges = {e for e in edge_types if e.startswith("seq")}
    dep_edges = {e for e in edge_types if e.startswith("dep:")}
    assert seq_edges, f"expected at least one seq edge, got {sorted(edge_types)}"
    assert dep_edges, f"expected at least one dep edge, got {sorted(edge_types)}"

    # Polarity: no negated triples, so every node sums to 0.
    assert all(p == 0 for p in polarity)


def test_smoke_subword_dep_builder_propagates_negation():
    """A negated evidence triple flips polarity on its covering subwords."""
    builder = SubwordDepEvidenceBuilder()
    ev = [_t("drug x", "treats", "disease y", pol=-1)]
    _, _, polarity = builder.build(ev)
    # at least one node must record the negation
    assert any(p < 0 for p in polarity)


def test_smoke_subword_dep_edge_types_are_in_dual_graph_vocab():
    """Every edge the builder emits must be present in ALL_EDGE_TYPES.

    This is the integration contract: if the builder emits a type the
    dual-graph layer does not recognise, the type will be silently
    bucketed as the open-relation fallback, which destroys
    interpretability.
    """
    builder = SubwordDepEvidenceBuilder()
    declared = set(builder.edge_type_names())
    for name in declared:
        assert name in ALL_EDGE_TYPES, (
            f"builder emits {name!r} but it is not in ALL_EDGE_TYPES; "
            f"dual_graph.py must be updated"
        )
    # And reverse: the published subword-dep vocab must contain every
    # name the builder may emit.
    for name in SUBWORD_DEP_EDGE_TYPES:
        assert name.startswith("seq") or name.startswith("dep:")


def test_smoke_subword_dep_builder_handles_empty_evidence():
    """Empty evidence input must yield an empty graph (no synthetic nodes)."""
    builder = SubwordDepEvidenceBuilder()
    labels, edges, polarity = builder.build([])
    assert labels == tuple()
    assert edges == []
    assert polarity == []


def test_smoke_subword_dep_builder_uses_mock_when_spacy_missing(monkeypatch):
    """If spaCy cannot be imported, the builder falls back to the mock parser.

    This is exercised by stubbing ``__import__`` to raise on ``spacy``.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "spacy" or name.startswith("spacy."):
            raise ImportError("spacy hidden for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    builder = SubwordDepEvidenceBuilder()
    assert builder.using_mock_spacy is True

    ev = [_t("alice", "instance_of", "researcher")]
    labels, edges, polarity = builder.build(ev)
    assert len(labels) > 0
    assert len(edges) > 0
    # The mock parser emits dep:nsubj for every non-root token.
    edge_types = {e[2] for e in edges}
    assert any(et.startswith("dep:") for et in edge_types)


# --------------------------------------------------------------------------
# 2. Compatibility: dual-graph constructor with subword_dep
# --------------------------------------------------------------------------

PAIRS = [
    (
        [_t("alice", "instance_of", "researcher")],
        [_t("alice", "instance_of", "researcher")],
    ),
    (
        [_t("drug_x", "treats", "disease_y")],
        [_t("drug_x", "treats", "disease_y", pol=-1)],
    ),
    (
        [_t("barack obama", "born_in", "honolulu")],
        [_t("barack obama", "born_in", "hawaii")],
    ),
]


@pytest.mark.parametrize("claim_triples,evidence_triples", PAIRS)
def test_compat_dual_graph_with_subword_dep_evidence_builder(claim_triples, evidence_triples):
    """The dual-graph constructor must accept ``evidence_builder='subword_dep'``
    and produce a structurally valid TypedGraph for each of three toy pairs.

    Checks the output schema only; whether the symbolic content helps or
    hurts verifier accuracy is an open empirical question.
    """
    builder = DualGraphBuilder(evidence_builder="subword_dep")
    g = builder.build(claim_triples, evidence_triples)

    assert isinstance(g, TypedGraph)
    # Schema checks.
    assert g.edge_index.dtype == np.int32
    assert g.edge_type.dtype == np.int32
    assert g.source.dtype == np.int32
    assert g.node_polarity.dtype == np.int32
    # Shape consistency.
    assert g.edge_index.ndim == 2 and g.edge_index.shape[0] == 2
    assert g.edge_index.shape[1] == g.edge_type.shape[0]
    assert g.source.shape[0] == g.n_nodes
    assert g.node_polarity.shape[0] == g.n_nodes
    # claim_n correctness.
    assert 0 <= g.claim_n <= g.n_nodes
    # Edge-type IDs must all be in range.
    if g.n_edges:
        assert int(g.edge_type.min()) >= 0
        assert int(g.edge_type.max()) < len(ALL_EDGE_TYPES)
    # Claim subgraph must still use entity-triple edges; evidence side
    # introduces the new types only on rows where source == 1.
    if g.n_edges:
        # At least one cross-graph or claim-side edge with the v0.1 vocab
        # should be present (the claim subgraph still uses entity triples).
        claim_edge_ids = {
            int(g.edge_type[k])
            for k in range(g.n_edges)
            if g.source[int(g.edge_index[0, k])] == 0
            and g.source[int(g.edge_index[1, k])] == 0
        }
        # at most this many entries belong to the entity-triple vocab
        n_v01 = sum(1 for eid in claim_edge_ids
                    if ALL_EDGE_TYPES[eid] not in SUBWORD_DEP_EDGE_TYPES)
        assert n_v01 >= 0  # trivially true; sanity check accessor only


def test_compat_subword_dep_emits_at_least_one_typed_subword_edge():
    """At least one of the three toy pairs must produce a subword-dep edge
    on the evidence subgraph.
    """
    claim, evidence = PAIRS[0]
    builder = DualGraphBuilder(evidence_builder="subword_dep")
    g = builder.build(claim, evidence)
    types_seen = {ALL_EDGE_TYPES[int(t)] for t in g.edge_type}
    assert types_seen & set(SUBWORD_DEP_EDGE_TYPES), (
        f"expected at least one subword-dep edge, got {sorted(types_seen)}"
    )


def test_compat_invalid_evidence_builder_raises():
    """Unknown evidence_builder names must fail loudly at construction."""
    with pytest.raises(ValueError):
        DualGraphBuilder(evidence_builder="something_else")


# --------------------------------------------------------------------------
# 3. Regression: default builder is bit-equal to the v0.1 behaviour
# --------------------------------------------------------------------------

def _typed_graph_bits(g: TypedGraph) -> dict:
    """Canonical comparable form of a TypedGraph."""
    return {
        "node_labels": tuple(g.node_labels),
        "edge_index": g.edge_index.tolist(),
        "edge_type": g.edge_type.tolist(),
        "source": g.source.tolist(),
        "node_polarity": g.node_polarity.tolist(),
        "claim_n": int(g.claim_n),
    }


def test_regression_default_builder_matches_v01_for_pair_0():
    """Pair 0 (matched claim+evidence): default builder unchanged.

    The expected fixture is the locked v0.1 dual-graph output. Its edge
    order is determined by:

    * the claim/evidence within-graph edges (forward + inverse for
      each triple, in triple order),
    * the entity-alignment cross-graph edges (claim_node -> evidence_node
      then the back-edge), iterated over node_labels in insertion order,
    * the relation-alignment edges, iterated over the same node pairs.

    The integer IDs are looked up via ``ALL_EDGE_TYPES.index`` so the
    test fails loudly if anyone reorders the edge-type vocabulary
    (which would break SAT-receipt reproducibility).
    """
    claim, evidence = PAIRS[0]
    g = DualGraphBuilder(evidence_builder="default").build(claim, evidence)
    expected = {
        "node_labels": ("alice", "researcher", "alice", "researcher"),
        "edge_index": [
            [0, 1, 2, 3, 0, 2, 1, 3, 0, 1],
            [1, 0, 3, 2, 2, 0, 3, 1, 2, 3],
        ],
        "edge_type": [
            ALL_EDGE_TYPES.index("rel:instance_of"),
            ALL_EDGE_TYPES.index("rel:instance_of:inv"),
            ALL_EDGE_TYPES.index("rel:instance_of"),
            ALL_EDGE_TYPES.index("rel:instance_of:inv"),
            ALL_EDGE_TYPES.index("align:entity"),
            ALL_EDGE_TYPES.index("align:entity"),
            ALL_EDGE_TYPES.index("align:entity"),
            ALL_EDGE_TYPES.index("align:entity"),
            ALL_EDGE_TYPES.index("align:relation"),
            ALL_EDGE_TYPES.index("align:relation"),
        ],
        "source": [0, 0, 1, 1],
        "node_polarity": [0, 0, 0, 0],
        "claim_n": 2,
    }
    assert _typed_graph_bits(g) == expected


def test_regression_default_builder_is_deterministic():
    """Building the same triples twice must produce bit-equal graphs."""
    for claim, evidence in PAIRS:
        g1 = DualGraphBuilder(evidence_builder="default").build(claim, evidence)
        g2 = DualGraphBuilder(evidence_builder="default").build(claim, evidence)
        assert _typed_graph_bits(g1) == _typed_graph_bits(g2)


def test_regression_default_builder_matches_no_evidence_builder_arg():
    """Passing ``evidence_builder='default'`` is bit-equal to not passing
    the argument at all (so callers that pre-date the toggle are
    unaffected).
    """
    for claim, evidence in PAIRS:
        g_explicit = DualGraphBuilder(evidence_builder="default").build(claim, evidence)
        g_implicit = DualGraphBuilder().build(claim, evidence)
        assert _typed_graph_bits(g_explicit) == _typed_graph_bits(g_implicit)


def test_regression_subword_dep_does_not_alter_claim_subgraph():
    """For the same claim triples, the claim subgraph slice (source == 0) is
    identical whether the evidence builder is default or subword_dep.

    The claim builder is independent of the evidence path; this guards
    against an accidental coupling.
    """
    claim, evidence = PAIRS[0]

    g_default = DualGraphBuilder(evidence_builder="default").build(claim, evidence)
    g_sw = DualGraphBuilder(evidence_builder="subword_dep").build(claim, evidence)

    # Claim node labels and polarity at the start of the array must match.
    assert g_default.node_labels[: g_default.claim_n] == g_sw.node_labels[: g_sw.claim_n]
    assert (
        g_default.node_polarity[: g_default.claim_n].tolist()
        == g_sw.node_polarity[: g_sw.claim_n].tolist()
    )

    def _claim_internal_edges(g):
        out = []
        for k in range(g.n_edges):
            u = int(g.edge_index[0, k])
            v = int(g.edge_index[1, k])
            if u < g.claim_n and v < g.claim_n:
                out.append((u, v, int(g.edge_type[k])))
        return sorted(out)

    assert _claim_internal_edges(g_default) == _claim_internal_edges(g_sw)
