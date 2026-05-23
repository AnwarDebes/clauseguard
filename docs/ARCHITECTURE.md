# ClauseGuard: Architecture Contract

Single source of truth. Every module ships against the interfaces
below; integration is gated on these contracts.

## Pillar

A **dual-graph walking Hierarchical Graph Tsetlin Machine** (HGTM)
that simultaneously evaluates clauses on a *claim graph* (the LLM's
output, decomposed into (s, r, o) triples) and an *evidence graph*
(retrieved from a knowledge graph or RAG corpus), and emits a
Support/Refute/NotEnoughInfo label together with the firing
clauses, a SAT-verifiable receipt, and a counterfactual
evidence-graph edit that would have flipped the decision.

## Stack

- **CUDA-C kernels** lifted from `graphtm-cbr/graphtm/cuda/kernels.cu`
  (forward, feedback, class-sum reduction, bit-packed TA updates).
  Carried over wholesale; the only change is that the per-node
  feature tensor now has two source-graph labels (claim / evidence).
- **Python orchestration**: PyTorch (LLM teacher only),
  GraphTsetlinMachine (CAIR), NumPy host glue, NetworkX (graph
  operations), python-sat (Glucose 4 solver).
- **Reference CPU**: `clauseguard.tm.hierarchical_tm` is a
  canonical Granmo & Saha port (mirrors graphtm-cbr's reference)
  used for numerical-parity tests of the CUDA path.

## Module map (8 modules)

```
src/clauseguard/
  extraction/    M1  LLM output -> atomic-claim triples
  graphs/        M2  triples -> typed graph + cross-graph edges
  tm/            M3  dual-graph walking HGTM + distillation
  recourse/      M4  evidence-graph counterfactual edit search
  verify/        M5  SAT-encode clauses + sign clause receipts
  data/          M6  FEVER, HaluEval, FActScore, MedHallBench loaders
  eval/          M7  metrics, adversarial harness, logger
  utils/         M8  seeding, deterministic IO helpers
```

## Invariants (enforced by tests/test_integration)

1. **Triple round-trip:** `triples = extract(claim); claim_text =
   render(triples); assert extract(render(triples)) == triples`
   on a fixed seed of 100 FEVER samples.
2. **Graph isomorphism:** `claim_graph(triples)` and
   `claim_graph(triples_permuted)` must produce isomorphic
   GraphTM-compatible Graphs objects (entity-label graphs are
   permutation-invariant by construction).
3. **CUDA-CPU parity:** forward output of the CUDA HGTM matches
   the NumPy reference within 1 vote unit on a fixed seed of 32
   FEVER training examples.
4. **No silent CPU fallback:** if `cuda.is_available()` is False
   and `--device cuda` was requested, raise immediately (graphtm-cbr
   invariant 4 carried over). The user explicitly wants GPU.
5. **SAT-receipt determinism:** for a fixed model version, the
   tuple `(claim_graph_hash, evidence_graph_hash, model_version_id)`
   uniquely determines the verification label and the firing-clause
   set; replay must reproduce both bit-for-bit.

## Interface contracts (frozen)

### M1: extraction/

```python
# src/clauseguard/extraction/triple_extractor.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Triple:
    subject: str           # canonicalised entity surface form
    relation: str          # relation label (one of REL_VOCAB or "_open_")
    object: str            # entity, literal, or numeric
    polarity: int = 1      # 1 = asserted, -1 = negated
    source_span: tuple[int, int] = (0, 0)  # char span in original text

@dataclass(frozen=True)
class Claim:
    text: str
    triples: tuple[Triple, ...]
    extractor_id: str      # "qwen2.5-1.5b@<sha>", for audit trail

class ClaimDecomposer:
    def decompose(self, llm_output: str) -> list[Claim]: ...

class TripleExtractor:
    def extract(self, claim_text: str) -> tuple[Triple, ...]: ...
```

### M2: graphs/

```python
# src/clauseguard/graphs/dual_graph.py
import numpy as np
from dataclasses import dataclass

@dataclass
class TypedGraph:
    """Mirrors graphtm-cbr's GraphTensor but extended for typed edges."""
    node_labels: tuple[str, ...]    # node symbols (entities / values)
    node_features: np.ndarray       # [n_nodes, F]   uint8 0/1 (binarised)
    edge_index: np.ndarray          # [2, n_edges]   int32
    edge_type: np.ndarray           # [n_edges]      int32 (typed)
    source: np.ndarray              # [n_nodes]      int32: 0=claim, 1=evidence

class DualGraphBuilder:
    """Combines a claim graph and an evidence graph into one TypedGraph
    with cross-graph alignment edges (entity-link, relation-match,
    contradiction-link). The GraphTM treats them as a single typed
    graph; clauses can reference cross-graph edge types directly."""
    def build(self,
              claim_triples: tuple[Triple, ...],
              evidence_triples: tuple[Triple, ...]) -> TypedGraph: ...
```

### M3: tm/

```python
# src/clauseguard/tm/graphtm_verifier.py
import numpy as np

class GraphTMVerifier:
    """Dual-graph walking HGTM. Wraps GraphTsetlinMachine.MultiClassGraphTsetlinMachine
    with a typed-edge schema, 3 classes (Support/Refute/NEI), and
    clause-export hooks for the verify/ module."""
    n_classes: int = 3
    n_clauses_per_class: int
    threshold_T: int
    specificity_s: float
    depth: int                          # AND-OR-AND-OR-AND tree depth (default 5)
    hypervector_dim: int                # VSA D (default 8192)

    def fit(self, graphs, labels: np.ndarray, epochs: int = 50): ...
    def predict(self, graphs) -> np.ndarray: ...                   # [n,] int
    def predict_with_clauses(self, graphs) -> list[dict]: ...      # per-sample firing clauses
    def export_clauses(self) -> list["ClauseSpec"]: ...            # for SAT + UI
```

### M4: recourse/

```python
# src/clauseguard/recourse/search.py
@dataclass(frozen=True)
class GraphEdit:
    op: str                      # "add_triple" | "remove_triple" | "swap_relation" | "fix_entity_link"
    triple: Triple | None
    new_value: str | None
    target_label: int            # the verification label we want to reach

def greedy_minimal_evidence_edit(
    model: GraphTMVerifier,
    claim_graph: TypedGraph,
    evidence_graph: TypedGraph,
    target_label: int = 0,       # 0 = Support
    max_edits: int = 3,
    candidates_from_clauses: bool = True,
) -> tuple[list[GraphEdit], bool]:  # (applied_edits, flipped)
    """Mirrors graphtm-cbr/graphtm/recourse/search.py:greedy_minimal_edit
    but operates on EVIDENCE-graph edits (we cannot edit the LLM's
    claim, we can only point at what evidence would change the
    verdict, that is the human-oversight artifact for Article 14)."""
```

### M5: verify/

```python
# src/clauseguard/verify/certificate.py
from dataclasses import dataclass

@dataclass(frozen=True)
class ClauseReceipt:
    claim_text: str
    triples_hash: str               # SHA-256 of canonicalised triples
    evidence_graph_hash: str        # SHA-256 of canonicalised evidence
    model_version_id: str           # "clauseguard-v0.1.0@<weights_sha>"
    label: int                      # 0=Support, 1=Refute, 2=NEI
    firing_clause_ids: tuple[int, ...]
    literal_vector_hash: str        # SHA-256 of the binarised literal vector
    automaton_state_hash: str       # SHA-256 of the relevant TA states
    cnf_export_path: str            # path to .cnf DIMACS file
    signature: str                  # HMAC-SHA-256 over the above

def sign(receipt_fields: dict, key: bytes) -> str: ...
def verify(receipt: ClauseReceipt, key: bytes) -> bool: ...

# src/clauseguard/verify/sat_encoder.py
def encode_clauses_to_cnf(
    clauses: list["ClauseSpec"], literals: np.ndarray, label: int
) -> "CNF":
    """Encode the firing-clause set and the literal vector for a single
    sample as a propositional CNF. The CNF is satisfiable iff the
    decision can be reproduced from those literals alone (i.e., no
    external state leaked into the decision). Auditor runs Glucose 4
    on the CNF in <50 ms."""
```

### M6: data/

```python
# src/clauseguard/data/fever.py
def load_fever(split: str = "train",
               max_samples: int | None = None,
               cache_dir: str | None = None) -> list[dict]:
    """Returns [{claim, label, evidence_sentences, evidence_triples}, ...]"""

# Same shape for halueval.py, factscore.py, medhall.py.
```

### M7: eval/

```python
# src/clauseguard/eval/metrics.py
def label_accuracy(y_true, y_pred) -> float: ...
def fever_score(y_true, y_pred, ev_true, ev_pred) -> float: ...
def auroc(y_true, y_score) -> float: ...
def clause_set_size(model) -> int: ...
def median_clause_length(model) -> float: ...
def sat_verification_latency(receipts) -> dict: ...
```

### M8: utils/

```python
# src/clauseguard/utils/seeding.py
def seed_all(seed: int): ...   # numpy, torch, random, env vars
```

## Hyperparameters (default, locked unless an experiment overrides)

| Hyper-parameter | Value | Source |
|---|---|---|
| n_clauses_per_class | 1000 | graphtm-cbr Phase 2 winner |
| threshold T | 1500 | sweep over {500, 1000, 1500, 2000} on FEVER dev |
| specificity s | 10.0 | CAIR GraphTM IMDB default, retuned on dev |
| HGTM depth | 5 | canonical Granmo & Saha |
| hypervector dim D | 8192 | graphtm-cbr |
| hypervector sparsity | 0.10 | graphtm-cbr |
| epochs | 50 | budget; early-stop on dev loss |
| seed list | 42, 123, 456, 789, 1337 | matches paper-a / paper-b |

## Compute budget (V100-SXM3-32GB)

| Phase | Approx. wall-clock on V100 |
|---|---|
| Triple extraction (FEVER train, 145 K samples, Qwen2.5-1.5B, bs=32) | ~6 h |
| Triple extraction (HaluEval 35 K) | ~1.5 h |
| GraphTM fit (FEVER train, 50 ep, K=5 ensemble) | ~3 h |
| Eval (all 4 benchmarks, all seeds) | ~1 h |
| Adversarial eval (TextAttack TextFooler + BERT-Attack) | ~2 h |
| Recourse eval (200 Refuted samples, max 3 edits) | ~30 min |
| SAT receipt generation + verification | ~5 min |

Total budget: ~14 GPU-hours per seed, ~70 GPU-hours over 5 seeds.

## File-by-file responsibility map

| Path | Responsibility | Owner |
|---|---|---|
| `src/clauseguard/extraction/claim_decomposer.py` | LLM-output → atomic claims | M1 |
| `src/clauseguard/extraction/triple_extractor.py` | claim → (s, r, o) triples | M1 |
| `src/clauseguard/extraction/qwen_extractor.py` | concrete Qwen2.5-1.5B implementation | M1 |
| `src/clauseguard/graphs/claim_graph.py` | claim triples → typed graph | M2 |
| `src/clauseguard/graphs/evidence_graph.py` | evidence triples → typed graph | M2 |
| `src/clauseguard/graphs/dual_graph.py` | combine + cross-graph edges | M2 |
| `src/clauseguard/graphs/binarize.py` | literal binarisation | M2 |
| `src/clauseguard/tm/graphtm_verifier.py` | dual-graph HGTM | M3 |
| `src/clauseguard/tm/distillation.py` | LLM-judge → TM distillation | M3 |
| `src/clauseguard/tm/ensemble.py` | K-seed ensembling | M3 |
| `src/clauseguard/recourse/candidates.py` | evidence-edit candidate gen | M4 |
| `src/clauseguard/recourse/search.py` | greedy search | M4 |
| `src/clauseguard/recourse/output.py` | recourse report rendering | M4 |
| `src/clauseguard/verify/sat_encoder.py` | clauses → DIMACS CNF | M5 |
| `src/clauseguard/verify/certificate.py` | sign + verify receipts | M5 |
| `src/clauseguard/data/fever.py` | FEVER loader | M6 |
| `src/clauseguard/data/halueval.py` | HaluEval loader | M6 |
| `src/clauseguard/data/factscore.py` | FActScore loader | M6 |
| `src/clauseguard/data/medhall.py` | MedHallBench loader | M6 |
| `src/clauseguard/eval/metrics.py` | task + interpretability metrics | M7 |
| `src/clauseguard/eval/adversarial.py` | TextAttack harness | M7 |
| `src/clauseguard/eval/logger.py` | per-sample JSONL provenance | M7 |
| `src/clauseguard/utils/seeding.py` | seed_all | M8 |
| `src/clauseguard/utils/io.py` | JSON, pickle, hashing | M8 |

## Out-of-scope (for this paper)

- Live federation across multiple LLM providers (architectural sketch
  only; experimental federation is a follow-up).
- Multilingual fact verification (English only in v0.1; XLM-R
  extension is a clear follow-up).
- Generative correction of LLM output (we say *why* and *what would
  fix it*; we do not rewrite the LLM's text).
