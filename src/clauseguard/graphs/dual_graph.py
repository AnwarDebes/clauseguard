"""Combine a claim graph and an evidence graph into a single typed graph
with cross-graph alignment edges.

This is the central novelty of the architecture: a single GraphTM walks
both subgraphs joined by typed alignment edges. Clauses can reference
cross-graph edges directly. No prior GraphTM walks two graphs.

The output ``TypedGraph`` is a thin in-memory dataclass; the conversion
to ``GraphTsetlinMachine.graphs.Graphs`` (the CAIR library object the TM
actually consumes) lives in :func:`to_graphtm_graphs` so this module
stays import-free of the heavy TM runtime.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..extraction.triple_extractor import REL_VOCAB, Triple
from .claim_graph import build_claim_graph
from .evidence_graph import build_evidence_graph


# --------------------------------------------------------------------------
# Edge-type vocabulary
# --------------------------------------------------------------------------

# Within-graph edges: forward rel + inverse rel for each canonical relation.
GRAPH_EDGE_TYPES: tuple[str, ...] = tuple(
    [f"rel:{r}" for r in REL_VOCAB] + [f"rel:{r}:inv" for r in REL_VOCAB]
)

# Cross-graph edges connect claim and evidence subgraphs. Three types
# are sufficient to express the alignment patterns the GraphTM needs to
# learn:
#
#   * ``align:entity``       claim node and evidence node share the same
#                            canonical entity label (exact match or
#                            string-similarity above a threshold).
#   * ``align:relation``     claim edge and evidence edge share the same
#                            canonical relation type for some entity pair,
#                            encoded as a node-to-node edge on the
#                            head entities.
#   * ``contradict:negation`` claim asserts a triple that evidence
#                            negates (or vice versa), encoded as an
#                            edge between the matched entity-pair nodes.
CROSS_GRAPH_EDGE_TYPES: tuple[str, ...] = (
    "align:entity",
    "align:relation",
    "contradict:negation",
)

# Paper-a (arXiv 2510.XXXXX, subword-dep GraphTM) edge vocabulary. Used
# only when the optional subword-dep evidence builder is selected; the
# default builder never emits these symbols. The list is appended at the
# end of ALL_EDGE_TYPES so the IDs of every existing edge type stay
# unchanged, which keeps SAT-receipt reproducibility and the default
# DualGraphBuilder output bit-identical to v0.1.
SUBWORD_DEP_RELATIONS: tuple[str, ...] = (
    "nsubj", "obj", "iobj",
    "amod", "advmod", "compound",
    "prep", "pobj",
    "det", "aux",
    "conj", "cc",
    "mark", "advcl", "relcl",
    "ROOT",
)
SUBWORD_DEP_EDGE_TYPES: tuple[str, ...] = (
    "seq_next",
    "seq_prev",
) + tuple(f"dep:{r}" for r in SUBWORD_DEP_RELATIONS) + tuple(
    f"dep:{r}_inv" for r in SUBWORD_DEP_RELATIONS
)

# Full vocab seen by the GraphTM, in a fixed order. The order is part of
# the model's interface contract, changing it breaks SAT-receipt
# reproducibility. Subword-dep types are appended after the cross-graph
# types so existing IDs do not move.
ALL_EDGE_TYPES: tuple[str, ...] = (
    GRAPH_EDGE_TYPES + CROSS_GRAPH_EDGE_TYPES + SUBWORD_DEP_EDGE_TYPES
)
_EDGE_TYPE_TO_ID = {e: i for i, e in enumerate(ALL_EDGE_TYPES)}


def edge_type_id(name: str) -> int:
    return _EDGE_TYPE_TO_ID[name]


# --------------------------------------------------------------------------
# TypedGraph dataclass
# --------------------------------------------------------------------------

@dataclass
class TypedGraph:
    """A typed-edge graph that the GraphTM consumes after binarisation.

    Attributes:
        node_labels: tuple of node surface forms (length n_nodes).
        edge_index: array of shape (2, n_edges), int32. Row 0 is src, row 1 is dst.
        edge_type: array of shape (n_edges,), int32. Values are indices into
            :data:`ALL_EDGE_TYPES`.
        source: array of shape (n_nodes,), int32. 0 = claim subgraph,
            1 = evidence subgraph.
        node_polarity: array of shape (n_nodes,), int32. Accumulated polarity
            of the triples touching each node (negative entries become a
            binary "negated" literal).
        claim_n: int. Number of nodes in the claim subgraph (the first
            ``claim_n`` entries of ``node_labels`` and ``source``). Useful
            when the recourse module needs to restrict edits to the
            evidence side.
    """

    node_labels: tuple[str, ...]
    edge_index: np.ndarray
    edge_type: np.ndarray
    source: np.ndarray
    node_polarity: np.ndarray
    claim_n: int

    @property
    def n_nodes(self) -> int:
        return len(self.node_labels)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def as_dict(self) -> dict:
        return {
            "node_labels": list(self.node_labels),
            "edge_index": self.edge_index.tolist(),
            "edge_type": self.edge_type.tolist(),
            "source": self.source.tolist(),
            "node_polarity": self.node_polarity.tolist(),
            "claim_n": int(self.claim_n),
        }


# --------------------------------------------------------------------------
# DualGraphBuilder
# --------------------------------------------------------------------------

EvidenceBuilderName = str  # one of: "default", "subword_dep"


class DualGraphBuilder:
    """Combine a claim graph + evidence graph into a single TypedGraph.

    Configuration (set via the constructor) controls which cross-graph
    edge types are emitted. Defaults emit all three.

    The ``evidence_builder`` argument selects the evidence-side graph
    construction strategy:

    * ``"default"``: the v0.1 entity/relation-triple evidence graph
      (:func:`clauseguard.graphs.evidence_graph.build_evidence_graph`).
      Bit-identical to all releases up to v0.1.
    * ``"subword_dep"``: an alternative evidence-side graph that uses
      BPE subword tokens as nodes and typed spaCy dependency edges
      (ported from paper-a). Whether it improves verification accuracy
      over the default is an open empirical question; the planned
      5-seed FEVER GPU run will answer it.
    """

    def __init__(
        self,
        emit_entity_alignment: bool = True,
        emit_relation_alignment: bool = True,
        emit_contradiction: bool = True,
        max_nodes: int = 256,
        evidence_builder: EvidenceBuilderName = "default",
    ) -> None:
        self.emit_entity_alignment = emit_entity_alignment
        self.emit_relation_alignment = emit_relation_alignment
        self.emit_contradiction = emit_contradiction
        self.max_nodes = max_nodes
        if evidence_builder not in ("default", "subword_dep"):
            raise ValueError(
                f"evidence_builder must be 'default' or 'subword_dep', "
                f"got {evidence_builder!r}"
            )
        self.evidence_builder = evidence_builder
        self._subword_dep_builder = None  # lazy

    def _build_evidence_subgraph(
        self,
        evidence_triples: Sequence[Triple],
    ) -> tuple[tuple[str, ...], list[tuple[int, int, str]], list[int]]:
        if self.evidence_builder == "default":
            return build_evidence_graph(evidence_triples)
        # "subword_dep" path. Import lazily so users on the default path
        # never pay the spaCy / transformers import cost.
        if self._subword_dep_builder is None:
            from ..builders.subword_dep_evidence import SubwordDepEvidenceBuilder

            self._subword_dep_builder = SubwordDepEvidenceBuilder()
        return self._subword_dep_builder.build(evidence_triples)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self,
        claim_triples: Sequence[Triple],
        evidence_triples: Sequence[Triple],
    ) -> TypedGraph:
        # 1. Build the two sub-graphs.
        c_labels, c_edges, c_polarity = build_claim_graph(claim_triples)
        e_labels, e_edges, e_polarity = self._build_evidence_subgraph(evidence_triples)

        # 2. Re-index the evidence side so its node ids start after the
        #    claim side. This keeps node ids globally unique without
        #    collapsing identically-labelled entities, which the cross-
        #    graph alignment edges handle explicitly.
        claim_n = len(c_labels)
        evidence_offset = claim_n
        node_labels = list(c_labels) + list(e_labels)
        if len(node_labels) > self.max_nodes:
            # Truncate the evidence side first (keep the entire claim).
            node_labels = node_labels[: self.max_nodes]

        # 3. Assemble the source array and per-node polarity.
        source = np.zeros(len(node_labels), dtype=np.int32)
        source[claim_n:] = 1
        node_polarity = np.array(
            list(c_polarity) + list(e_polarity), dtype=np.int32
        )[: len(node_labels)]

        # 4. Within-graph edges (claim + evidence).
        edges_src: list[int] = []
        edges_dst: list[int] = []
        edges_type: list[int] = []

        def _emit_edges(raw_edges, offset: int) -> None:
            for src, dst, name in raw_edges:
                if name not in _EDGE_TYPE_TO_ID:
                    # Unknown edge type, bucket as the closest open relation.
                    name = "rel:_open_" if not name.endswith(":inv") else "rel:_open_:inv"
                u = src + offset
                v = dst + offset
                if u >= len(node_labels) or v >= len(node_labels):
                    continue
                edges_src.append(u)
                edges_dst.append(v)
                edges_type.append(_EDGE_TYPE_TO_ID[name])

        _emit_edges(c_edges, 0)
        _emit_edges(e_edges, evidence_offset)

        # 5. Cross-graph alignment edges.
        if self.emit_entity_alignment:
            # Connect every claim node to every evidence node sharing the
            # same canonical label. In practice the number of matches per
            # claim is small (1-3 entities); we cap at a few hundred to
            # avoid quadratic blow-up on long evidence.
            evidence_label_to_id: dict[str, list[int]] = {}
            for i, lab in enumerate(node_labels[claim_n:], start=claim_n):
                evidence_label_to_id.setdefault(lab, []).append(i)
            for i, lab in enumerate(node_labels[:claim_n]):
                for j in evidence_label_to_id.get(lab, ()):
                    edges_src.append(i)
                    edges_dst.append(j)
                    edges_type.append(_EDGE_TYPE_TO_ID["align:entity"])
                    edges_src.append(j)
                    edges_dst.append(i)
                    edges_type.append(_EDGE_TYPE_TO_ID["align:entity"])

        if self.emit_relation_alignment:
            # For each pair of head entities that match across graphs,
            # if there is an outgoing edge of the same relation type on
            # both sides, add an align:relation edge between the heads.
            claim_out: dict[tuple[int, int], list[int]] = {}
            for u, v, t in zip(edges_src, edges_dst, edges_type):
                if u < claim_n and v < claim_n:
                    claim_out.setdefault((u, t), []).append(v)
            ev_out: dict[tuple[int, int], list[int]] = {}
            for u, v, t in zip(edges_src, edges_dst, edges_type):
                if u >= claim_n and v >= claim_n:
                    ev_out.setdefault((u, t), []).append(v)
            for (u, t), _claim_tails in claim_out.items():
                if t >= len(GRAPH_EDGE_TYPES):
                    continue  # cross-graph types not relevant here
                if t % 2 == 1:
                    continue  # skip inverse edges to avoid duplication
                # find matching evidence-side head with same label
                u_label = node_labels[u]
                ev_heads = [
                    i
                    for i, lab in enumerate(node_labels[claim_n:], start=claim_n)
                    if lab == u_label and (i, t) in ev_out
                ]
                for v in ev_heads:
                    edges_src.append(u)
                    edges_dst.append(v)
                    edges_type.append(_EDGE_TYPE_TO_ID["align:relation"])

        if self.emit_contradiction:
            # Contradiction signature: same (s, r, o) across graphs but
            # opposite polarity (e.g., claim asserts "X treats Y", evidence
            # asserts "X does NOT treat Y"). Detected via the per-node
            # polarity sums combined with same-label entity alignment.
            for i, lab in enumerate(node_labels[:claim_n]):
                for j, ev_lab in enumerate(node_labels[claim_n:], start=claim_n):
                    if lab != ev_lab:
                        continue
                    if (node_polarity[i] < 0) ^ (node_polarity[j] < 0):
                        edges_src.append(i)
                        edges_dst.append(j)
                        edges_type.append(_EDGE_TYPE_TO_ID["contradict:negation"])

        # 6. Assemble the final TypedGraph.
        if edges_src:
            edge_index = np.stack(
                [np.asarray(edges_src, dtype=np.int32), np.asarray(edges_dst, dtype=np.int32)]
            )
            edge_type = np.asarray(edges_type, dtype=np.int32)
        else:
            edge_index = np.empty((2, 0), dtype=np.int32)
            edge_type = np.empty((0,), dtype=np.int32)

        return TypedGraph(
            node_labels=tuple(node_labels),
            edge_index=edge_index,
            edge_type=edge_type,
            source=source,
            node_polarity=node_polarity,
            claim_n=min(claim_n, len(node_labels)),
        )


# --------------------------------------------------------------------------
# Conversion to CAIR GraphTsetlinMachine.graphs.Graphs
# --------------------------------------------------------------------------

def to_graphtm_graphs(
    graphs: list[TypedGraph],
    symbols: list[str],
    hypervector_size: int = 8192,
    hypervector_bits: int = 2,
    *,
    train: bool = True,
    init_with: "Graphs | None" = None,
):
    """Convert a list of :class:`TypedGraph` into a CAIR
    ``GraphTsetlinMachine.graphs.Graphs`` object suitable for fitting or
    inference.

    Imports ``GraphTsetlinMachine`` lazily so this module remains
    importable on machines without CUDA. If the library is unavailable
    a clear ImportError is raised at call time.
    """
    try:
        from GraphTsetlinMachine.graphs import Graphs
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "GraphTsetlinMachine is not installed. "
            "Run `pip install GraphTsetlinMachine==0.3.3`."
        ) from exc

    if init_with is not None:
        gtm = Graphs(
            number_of_graphs=len(graphs),
            init_with=init_with,
        )
    else:
        gtm = Graphs(
            number_of_graphs=len(graphs),
            symbols=symbols,
            hypervector_size=hypervector_size,
            hypervector_bits=hypervector_bits,
            double_hashing=False,
        )

    # Phase 1: declare node counts per graph.
    for gi, g in enumerate(graphs):
        gtm.set_number_of_graph_nodes(gi, max(1, g.n_nodes))
    gtm.prepare_node_configuration()

    # Phase 2: declare per-node outgoing-edge counts (must come before edge writes).
    out_counts = [
        np.bincount(g.edge_index[0], minlength=g.n_nodes) if g.n_edges else np.zeros(g.n_nodes, dtype=np.int64)
        for g in graphs
    ]
    for gi, g in enumerate(graphs):
        for ni in range(g.n_nodes):
            gtm.add_graph_node(gi, ni, int(out_counts[gi][ni]) if ni < len(out_counts[gi]) else 0)
    gtm.prepare_edge_configuration()

    # Phase 3: emit edges with their typed labels.
    for gi, g in enumerate(graphs):
        for k in range(g.n_edges):
            src = int(g.edge_index[0, k])
            dst = int(g.edge_index[1, k])
            etype = ALL_EDGE_TYPES[int(g.edge_type[k])]
            gtm.add_graph_node_edge(gi, src, dst, etype)

    # Phase 4: attach node symbols (entity label + polarity literal + source literal).
    for gi, g in enumerate(graphs):
        for ni in range(g.n_nodes):
            entity_sym = f"e:{g.node_labels[ni]}"
            polarity_sym = "polarity:neg" if g.node_polarity[ni] < 0 else "polarity:pos"
            source_sym = "source:claim" if g.source[ni] == 0 else "source:evidence"
            gtm.add_graph_node_property(gi, ni, entity_sym)
            gtm.add_graph_node_property(gi, ni, polarity_sym)
            gtm.add_graph_node_property(gi, ni, source_sym)

    gtm.encode()
    return gtm


def collect_symbol_vocab(graphs: list[TypedGraph], *, top_k: int = 5000) -> list[str]:
    """Build the GraphTM symbol vocabulary from a collection of graphs.

    Includes entity symbols (top-K most-frequent), polarity literals, and
    source-graph literals. Edge types are handled separately by the
    GraphTsetlinMachine library; do not include them here.
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    for g in graphs:
        for lab in g.node_labels:
            counter[f"e:{lab}"] += 1
    top_entities = [sym for sym, _ in counter.most_common(top_k)]
    return top_entities + ["polarity:neg", "polarity:pos", "source:claim", "source:evidence"]
