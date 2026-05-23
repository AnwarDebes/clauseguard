"""Build a typed claim-side graph from a tuple of Triples.

Nodes are entities (canonicalised surface forms) and edges are the
relations connecting them. The same entity appearing in multiple triples
becomes a single node, this de-duplication is essential because clauses
need to reference the *same* node across triples.

Edge types are taken from
:mod:`clauseguard.graphs.dual_graph.GRAPH_EDGE_TYPES`, which is the
union of:

* ``rel:<canonical_relation>`` for forward edges,
* ``rel:<canonical_relation>:inv`` for back-edges (so clauses can walk
  either direction),
* ``polarity:neg`` self-edge attached to nodes whose owning triple is
  negated, lets clauses condition on negation locally without separate
  node feature plumbing.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..extraction.triple_extractor import REL_VOCAB, Triple


@dataclass
class _NodeIndex:
    """Stable, insertion-ordered entity -> node-id map."""

    label_to_id: dict[str, int]
    labels: list[str]

    def get(self, label: str) -> int:
        if label not in self.label_to_id:
            self.label_to_id[label] = len(self.labels)
            self.labels.append(label)
        return self.label_to_id[label]


def build_claim_graph(
    triples: Sequence[Triple],
) -> tuple[tuple[str, ...], list[tuple[int, int, str]], list[int]]:
    """Build the node/edge structure for the claim-side graph.

    Returns:
        node_labels: tuple of entity surface forms (length n_nodes).
        edges: list of ``(src, dst, edge_type_name)`` tuples.
        node_polarity: per-node polarity sum, ``len(node_labels)`` entries.
            Nodes attached to negated triples accumulate -1; nodes attached
            only to positive triples stay at 0. The dual-graph layer can
            turn this into a binary literal.
    """
    idx = _NodeIndex(label_to_id={}, labels=[])
    edges: list[tuple[int, int, str]] = []
    polarity: dict[int, int] = {}

    for t in triples:
        s = idx.get(t.subject)
        o = idx.get(t.object)
        edge_name = f"rel:{t.relation}"
        edges.append((s, o, edge_name))
        edges.append((o, s, f"{edge_name}:inv"))
        if t.polarity == -1:
            polarity[s] = polarity.get(s, 0) - 1
            polarity[o] = polarity.get(o, 0) - 1

    node_polarity = [polarity.get(i, 0) for i in range(len(idx.labels))]
    return tuple(idx.labels), edges, node_polarity
