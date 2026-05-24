"""Subword-dependency evidence-graph builder.

Alternative evidence-side graph constructor ported from paper-a
(`paper-a-subword-dep-graphtm`, arXiv 2510.XXXXX). Where the default
evidence builder represents evidence as a small entity-level graph
(one node per canonical entity, edges typed by relation), this builder
represents evidence as a per-document subword multigraph:

* nodes are BPE subword tokens from a BERT tokenizer (tokenizer only,
  no model weights),
* sequential edges (``seq_next``, ``seq_prev``) chain adjacent subwords,
* dependency edges (``dep:nsubj``, ``dep:dobj``, ..., and inverses)
  carry the spaCy dependency tree of the underlying text, aligned to
  the first subword of each spaCy token.

The motivation is finer-grained linguistic structure on the evidence
side. Whether this actually improves verification accuracy on FEVER /
HaluEval / FActScore / MedHallBench is an open empirical question and
must be answered by the planned 5-seed GPU run; nothing in this module
makes that claim.

The public interface mirrors :func:`clauseguard.graphs.evidence_graph.
build_evidence_graph`: the ``build`` method returns the same
``(node_labels, edges, polarity)`` triple, so :class:`DualGraphBuilder`
can plug either back end behind the same call site.

spaCy fallback
--------------
spaCy may not be installed on every machine. The builder probes for
spaCy at construction time. If ``import spacy`` succeeds and
``spacy.load("en_core_web_sm")`` succeeds, the real dependency parser
is used. Otherwise the builder falls back to a deterministic mock
parser that whitespace-tokenises the input, treats the last token as
``ROOT``, and connects every other token to the root with ``dep:nsubj``.
The mock exists so unit tests can run on a machine without spaCy; it
does NOT pretend to be a useful linguistic parser.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

from ..extraction.triple_extractor import Triple


# Dependency relation labels carried over from paper-a. Edge IDs are
# allocated in :mod:`clauseguard.graphs.dual_graph` so the receipt is
# stable across releases.
_DEP_RELATIONS: tuple[str, ...] = (
    "nsubj", "obj", "iobj",
    "amod", "advmod", "compound",
    "prep", "pobj",
    "det", "aux",
    "conj", "cc",
    "mark", "advcl", "relcl",
    "ROOT",
)
_DEP_RELATION_SET = set(_DEP_RELATIONS)

# Default tokenizer for the subword node layer.
_DEFAULT_TOKENIZER_NAME = "bert-base-uncased"
_DEFAULT_SPACY_MODEL = "en_core_web_sm"


class _MockSpacyToken:
    """Minimal duck-typed stand-in for a spaCy ``Token`` object.

    Used when spaCy or the configured spaCy model cannot be loaded.
    Exposes the four attributes the builder reads: ``i``, ``idx``,
    ``text``, ``dep_``, and ``head`` (where ``head`` is another
    ``_MockSpacyToken``).
    """

    __slots__ = ("i", "idx", "text", "dep_", "head")

    def __init__(self, i: int, idx: int, text: str, dep_: str):
        self.i = i
        self.idx = idx
        self.text = text
        self.dep_ = dep_
        self.head = self  # default self-loop, overwritten after construction


def _mock_parse(text: str) -> list[_MockSpacyToken]:
    """Deterministic whitespace tokeniser with a trivial dep tree.

    The last token is labelled ``ROOT`` and points at itself; every
    other token is labelled ``nsubj`` and points at the root. This is
    not a useful parser, it exists so that the unit tests can run on a
    machine where spaCy / its English model is unavailable.
    """
    tokens: list[_MockSpacyToken] = []
    if not text:
        return tokens
    idx = 0
    raw = text.split()
    if not raw:
        return tokens
    for i, word in enumerate(raw):
        tok = _MockSpacyToken(i=i, idx=idx, text=word, dep_="nsubj")
        tokens.append(tok)
        idx += len(word) + 1  # +1 for the space separator
    # last token becomes the ROOT, pointing at itself.
    tokens[-1].dep_ = "ROOT"
    tokens[-1].head = tokens[-1]
    for tok in tokens[:-1]:
        tok.head = tokens[-1]
    return tokens


class SubwordDepEvidenceBuilder:
    """Evidence-side graph builder with subword nodes and typed dep edges.

    Public interface mirrors :func:`build_evidence_graph`:

    >>> b = SubwordDepEvidenceBuilder()
    >>> labels, edges, polarity = b.build(evidence_triples)

    ``edges`` is a list of ``(src_idx, dst_idx, edge_type_name)`` tuples
    whose edge-type names are registered in
    :data:`clauseguard.graphs.dual_graph.ALL_EDGE_TYPES`.

    Parameters
    ----------
    tokenizer_name : str
        HuggingFace tokenizer identifier. Used for BPE subword
        segmentation only; no model weights are loaded.
    spacy_model : str
        spaCy model identifier for the dependency parser. If the model
        is unavailable, the builder falls back to the deterministic
        mock parser (see module docstring).
    max_subwords : int
        Cap on the number of subword nodes per evidence document.
    use_sequential_edges : bool
        Whether to emit ``seq_next`` / ``seq_prev`` edges.
    use_dep_edges : bool
        Whether to emit ``dep:<rel>`` / ``dep:<rel>_inv`` edges.
    """

    def __init__(
        self,
        tokenizer_name: str = _DEFAULT_TOKENIZER_NAME,
        spacy_model: str = _DEFAULT_SPACY_MODEL,
        max_subwords: int = 128,
        use_sequential_edges: bool = True,
        use_dep_edges: bool = True,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.spacy_model = spacy_model
        self.max_subwords = max_subwords
        self.use_sequential = use_sequential_edges
        self.use_deps = use_dep_edges

        self._tokenizer = None
        self._nlp = None
        self._using_mock_spacy = False
        self._init_backends()

    # ------------------------------------------------------------------
    # Backend init
    # ------------------------------------------------------------------
    def _init_backends(self) -> None:
        """Load the BPE tokenizer and the spaCy parser if available.

        Failures fall back to whitespace tokenization and the mock dep
        parser. The fallback is recorded on ``self._using_mock_spacy``
        for tests to assert against.
        """
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        except Exception:
            self._tokenizer = None  # whitespace fallback in _tokenize()

        try:
            import spacy

            self._nlp = spacy.load(self.spacy_model)
            self._using_mock_spacy = False
        except Exception:
            self._nlp = None
            self._using_mock_spacy = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _evidence_text(self, triples: Sequence[Triple]) -> tuple[str, list[tuple[int, int, int]]]:
        """Render evidence triples to a single text plus per-triple char spans.

        Each triple is rendered as ``subject relation object`` separated
        by spaces (no markup, so spaCy / the BPE tokenizer parse it as
        ordinary text). The returned span list ``[(start, end, polarity),
        ...]`` records the char range each triple occupies in the joined
        text, used downstream for polarity propagation onto subword
        nodes.
        """
        parts: list[str] = []
        spans: list[tuple[int, int, int]] = []
        cursor = 0
        for t in triples:
            piece = f"{t.subject} {t.relation} {t.object}".strip()
            if not piece:
                continue
            if parts:
                cursor += 1  # for the joining space
            start = cursor
            end = start + len(piece)
            spans.append((start, end, t.polarity))
            parts.append(piece)
            cursor = end
        text = " ".join(parts)
        return text, spans

    def _tokenize(
        self, text: str
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """BPE-tokenize ``text``.

        Returns ``(subword_tokens, offset_mapping)``. Falls back to
        whitespace tokenization if the BPE tokenizer is unavailable.
        """
        if self._tokenizer is not None:
            encoding = self._tokenizer(
                text,
                truncation=True,
                max_length=self.max_subwords,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            token_ids = encoding["input_ids"]
            offsets = list(encoding["offset_mapping"])
            subwords = self._tokenizer.convert_ids_to_tokens(token_ids)
            return list(subwords), offsets
        # Whitespace fallback.
        subwords: list[str] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for word in text.split():
            start = text.find(word, cursor)
            end = start + len(word)
            subwords.append(word)
            offsets.append((start, end))
            cursor = end
            if len(subwords) >= self.max_subwords:
                break
        return subwords, offsets

    def _parse(self, text: str):
        """Run the dep parser, falling back to the mock when needed."""
        if self._nlp is not None:
            return list(self._nlp(text))
        return _mock_parse(text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self,
        evidence_triples: Sequence[Triple],
    ) -> tuple[tuple[str, ...], list[tuple[int, int, str]], list[int]]:
        """Build a subword-dep evidence subgraph from evidence triples.

        Output shape matches :func:`build_evidence_graph` so the result
        can be consumed directly by :class:`DualGraphBuilder`:

        * ``node_labels``: tuple of subword strings (length n_nodes).
        * ``edges``: list of ``(src_idx, dst_idx, edge_type_name)``.
        * ``node_polarity``: per-node int polarity sum. A subword that
          covers a char span belonging to a negated triple accumulates
          -1; subwords covering only asserted triples stay at 0.

        Empty evidence yields an empty graph: ``((), [], [])``. The
        caller (``DualGraphBuilder``) handles the empty-evidence case
        already, so no synthetic placeholder node is emitted here.
        """
        if not evidence_triples:
            return tuple(), [], []

        text, polarity_spans = self._evidence_text(evidence_triples)
        if not text:
            return tuple(), [], []

        subwords, offsets = self._tokenize(text)
        if not subwords:
            return tuple(), [], []
        n = len(subwords)

        # Char-span coverage to spaCy tokens.
        parsed = self._parse(text)
        spacy_to_subword: dict[int, list[int]] = {}
        for sp_idx, sp_token in enumerate(parsed):
            sp_start = int(sp_token.idx)
            sp_end = sp_start + len(sp_token.text)
            matched: list[int] = []
            for sw_idx, (sw_start, sw_end) in enumerate(offsets):
                if sw_end <= sp_start or sw_start >= sp_end:
                    continue
                matched.append(sw_idx)
            if matched:
                spacy_to_subword[sp_idx] = matched

        edges: list[tuple[int, int, str]] = []

        # Sequential edges (chain of subwords).
        if self.use_sequential:
            for i in range(n - 1):
                edges.append((i, i + 1, "seq_next"))
                edges.append((i + 1, i, "seq_prev"))

        # Dependency edges, aligned to the first subword of each token.
        if self.use_deps:
            for sp_token in parsed:
                dep_label = getattr(sp_token, "dep_", "")
                if dep_label not in _DEP_RELATION_SET:
                    continue
                child_sws = spacy_to_subword.get(sp_token.i, [])
                head_sws = spacy_to_subword.get(getattr(sp_token.head, "i", -1), [])
                if not child_sws or not head_sws:
                    continue
                c_idx = child_sws[0]
                h_idx = head_sws[0]
                if c_idx == h_idx:
                    continue  # skip self-loops (e.g. ROOT)
                edges.append((h_idx, c_idx, f"dep:{dep_label}"))
                edges.append((c_idx, h_idx, f"dep:{dep_label}_inv"))

        # Polarity propagation: any subword whose char span overlaps a
        # negated triple's span accumulates -1.
        polarity = [0] * n
        for sw_idx, (sw_start, sw_end) in enumerate(offsets):
            if sw_start == sw_end:
                continue
            for span_start, span_end, pol in polarity_spans:
                if sw_end <= span_start or sw_start >= span_end:
                    continue
                if pol == -1:
                    polarity[sw_idx] -= 1

        return tuple(subwords), edges, polarity

    # ------------------------------------------------------------------
    # Introspection (for tests / debugging)
    # ------------------------------------------------------------------
    @property
    def using_mock_spacy(self) -> bool:
        """True when the deterministic mock parser is in use."""
        return self._using_mock_spacy

    def edge_type_names(self) -> Iterable[str]:
        """All edge-type names this builder may emit."""
        out: list[str] = []
        if self.use_sequential:
            out.extend(["seq_next", "seq_prev"])
        if self.use_deps:
            for r in _DEP_RELATIONS:
                out.append(f"dep:{r}")
                out.append(f"dep:{r}_inv")
        return tuple(out)
