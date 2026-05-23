"""Unit tests for the graph construction module."""
from __future__ import annotations

import numpy as np

from clauseguard.extraction.triple_extractor import Triple
from clauseguard.graphs.claim_graph import build_claim_graph
from clauseguard.graphs.dual_graph import (
    ALL_EDGE_TYPES,
    DualGraphBuilder,
    TypedGraph,
)


def _t(s, r, o, pol=1):
    return Triple(subject=s, relation=r, object=o, polarity=pol)


def test_claim_graph_node_dedup():
    triples = [
        _t("alice", "instance_of", "researcher"),
        _t("alice", "born_in", "berlin"),
    ]
    labels, edges, polarity = build_claim_graph(triples)
    # 3 nodes: alice, researcher, berlin
    assert labels == ("alice", "researcher", "berlin")
    # 4 edges: forward + inverse for each triple
    assert len(edges) == 4


def test_claim_graph_polarity():
    triples = [_t("alice", "instance_of", "researcher", pol=-1)]
    _, _, pol = build_claim_graph(triples)
    assert pol[0] == -1  # alice
    assert pol[1] == -1  # researcher


def test_dual_graph_combines_subgraphs():
    claim = [_t("alice", "instance_of", "researcher")]
    evidence = [_t("alice", "instance_of", "researcher")]
    g = DualGraphBuilder().build(claim, evidence)
    assert g.n_nodes == 4  # alice, researcher, alice', researcher'
    assert g.claim_n == 2
    assert g.source[0] == 0 and g.source[2] == 1


def test_dual_graph_entity_alignment_edges():
    """When the same entity appears on both sides, align:entity edges are emitted."""
    claim = [_t("alice", "instance_of", "researcher")]
    evidence = [_t("alice", "born_in", "berlin")]
    g = DualGraphBuilder(emit_relation_alignment=False, emit_contradiction=False).build(
        claim, evidence
    )
    align_id = ALL_EDGE_TYPES.index("align:entity")
    assert int((g.edge_type == align_id).sum()) >= 2  # at least the alice<->alice pair


def test_dual_graph_contradiction_signature():
    claim = [_t("drug_x", "treats", "disease_y", pol=1)]
    evidence = [_t("drug_x", "treats", "disease_y", pol=-1)]
    g = DualGraphBuilder().build(claim, evidence)
    contradict_id = ALL_EDGE_TYPES.index("contradict:negation")
    assert int((g.edge_type == contradict_id).sum()) > 0


def test_dual_graph_isomorphism_under_triple_permutation():
    triples = [
        _t("a", "instance_of", "b"),
        _t("a", "located_in", "c"),
        _t("d", "instance_of", "e"),
    ]
    g1 = DualGraphBuilder().build(triples, triples)
    g2 = DualGraphBuilder().build(list(reversed(triples)), list(reversed(triples)))
    # Same node set (as a multiset), same edge count.
    assert sorted(g1.node_labels) == sorted(g2.node_labels)
    assert g1.n_edges == g2.n_edges


def test_typed_graph_as_dict_round_trip():
    triples = [_t("a", "instance_of", "b")]
    g = DualGraphBuilder().build(triples, triples)
    d = g.as_dict()
    assert d["claim_n"] == 2  # a, b
    assert "edge_index" in d
    assert "node_polarity" in d
