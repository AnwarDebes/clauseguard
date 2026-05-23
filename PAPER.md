# ClauseGuard: A Dual-Graph Tsetlin Machine Audit Layer for LLM Output Verification with SAT-Verifiable Clause Receipts and Counterfactual Evidence Recourse

> Working draft. Numbers in §4 are pre-build *targets*; the locked
> headline numbers will replace them once the 5-seed runs complete.
> Locked numbers will follow the rigor protocol from
> `graphtm-cbr/paper/paper.md` §4: never headline a best seed, always
> bootstrap CIs, never claim novelty without the audit in
> `docs/novelty_audit.md`.

**Author**: Anwar, University of Agder (UiA).

---

## Abstract

Regulators (EU AI Act Articles 13 and 14, in force for high-risk
deployers from August 2026; FDA AI/ML SaMD predetermined change
control plan, December 2024; NIST AI RMF 1.0) require that
LLM-based systems used in medical, legal, financial, biometric,
educational, and HR settings be sufficiently transparent and
support effective human oversight. No existing LLM guardrail
(NeMo Guardrails, Rebedea et al., EMNLP-Demo 2023; Llama Guard 3
and 4, Meta 2024-2025; ConceptGuard, arXiv 2508.16325; KEA
Explain; HalluGraph; AlignScore; Semantic Entropy) produces
machine-checkable propositional rules that a notified body can
audit. They give a score (continuous neural model output) or a
post-hoc attribution (SHAP-style), not a verifiable rule set.

I present **ClauseGuard**, the first LLM-output verification layer
built on a **dual-graph walking Hierarchical Graph Tsetlin Machine**
that:

1. Decomposes the LLM output into atomic (subject, relation,
   object) triples (open-source Qwen2.5-1.5B extractor with a
   logged, hashable audit trail).
2. Builds a *claim graph* and an *evidence graph* over the same
   entity vocabulary, joined by typed cross-graph alignment edges
   (`align:entity`, `align:relation`, `contradict:negation`).
3. Runs a single graph-walking HGTM (per-node clause evaluation,
   OR-across-nodes, edge information bound via VSA hyperdimensional
   binding, the canonical Granmo & Saha formulation) over both
   subgraphs jointly.
4. Emits a per-claim Support / Refute / NotEnoughInfo label and a
   **clause receipt**: a signed tuple recording the firing clauses,
   literal vector, automaton state, and DIMACS CNF that an auditor
   reproduces in `≤ 50 ms` using Glucose 4 (Saha et al. arXiv
   2303.14464 SAT-verification, extended from MNIST robustness to
   deployed audit).
5. For Refuted claims, returns the **minimum evidence-graph edit**
   that would flip the label to Supported, directly addressing
   Article 14 ("effective human oversight").

The combination of (i) dual-graph walking, (ii) SAT-verifiable
clause receipts, and (iii) counterfactual evidence-graph recourse,
applied to LLM-output verification, is, per the literature scans in
`docs/novelty_audit.md`, **unoccupied** in 2026.

Targeted results on the four standard benchmarks:

| Benchmark | Headline target | Locked |
|---|---|---|
| FEVER blind label acc / FEVER score | `≥ 75 / ≥ 64` | TBD |
| HaluEval-QA AUROC | `≥ 75` | TBD |
| FActScore F1 | `≥ 0.74` | TBD |
| MedHallBench AUROC | `≥ 0.70` | TBD |
| Clause-set size for 95% recall | `≤ 500` | TBD |
| SAT verification p50 latency | `≤ 50 ms` | TBD |
| Adversarial-paraphrase robust acc | `≥ 70%` | TBD |
| Auditor study (n ≥ 10) | `≥ 80%` agreement | TBD |

These are *honest* targets. ClauseGuard's contribution is not
"highest AUROC"; semantic entropy and AlignScore retain a small
edge there. ClauseGuard's contribution is the only LLM guardrail
on the market whose every decision an EU AI Act notified body can
audit. Where regulation forces auditability, ClauseGuard is the
sole option.

---

## 1. Introduction

### 1.1 Regulatory motivation

- **EU AI Act Article 13 (transparency)** and **Article 14
  (effective human oversight)**: high-risk Annex-III deployers
  must, from August 2026, provide deployers with information
  sufficient to "interpret the system's output and use it
  appropriately", and ensure "natural persons can effectively
  oversee" each deployment.
- **FDA AI/ML SaMD Predetermined Change Control Plan** (Dec 2024
  final guidance): every model modification must be declared with
  a machine-readable modification protocol.
- **NIST AI RMF 1.0** Govern-Map-Measure-Manage framework:
  measurable transparency is required for federal procurement.

Neural guardrails fail Article 14 by construction: a deployer
cannot "effectively oversee" a 12 B-parameter classifier whose
internal computation is opaque. They fail Article 13 because the
output they hand the deployer is a score, not an explanation a
non-AI lawyer or clinician can verify.

### 1.2 Hallucination as a deployment risk

- *Mata v. Avianca* (S.D.N.Y., June 2023), $5,000 sanction +
  bar-discipline referral after ChatGPT-fabricated case citations.
- *Moffatt v. Air Canada* (BC Civil Resolution Tribunal, Feb 2024):
  airline held legally liable for chatbot misrepresentation.
- Stanford SDLP, *Legal RAG Hallucinations* (2025), 17-33%
  citation hallucination in dedicated legal-RAG products.
- PubMed PMC12518350 (2024), 28% citation hallucination on GPT-4
  medical queries.

### 1.3 What ClauseGuard adds

The single technical kernel of the contribution is a **single
graph-walking GraphTM that evaluates clauses on two graphs joined
by typed cross-graph edges**, and the operational kernel is the
**clause receipt + evidence-graph recourse** pair that turns Article
13 + 14 into runtime guarantees.

### 1.4 Contributions

1. **The first dual-graph Tsetlin Machine.** Clauses are evaluated
   on a single graph composed of (claim subgraph ∪ evidence
   subgraph ∪ cross-graph alignment edges). No prior GraphTM walks
   two graphs. (§ 3.1)
2. **SAT-verifiable clause receipt for an LLM-safety decision.**
   Extends Saha et al. (arXiv 2303.14464) from MNIST robustness
   verification to a deployed audit setting. (§ 3.5)
3. **Counterfactual evidence-graph recourse.** Adapts
   graphtm-cbr's clause-driven recourse from molecule edits to
   triple edits, directly addressing Article 14's
   "effective human oversight" requirement. (§ 3.4)
4. **Auditable triple-extraction trail.** Every extractor call is
   logged with model name, weights SHA, prompt, raw output. (§ 3.2)
5. **Adversarial robustness audit.** Paper-c's TextAttack harness
   stress-tests the audit layer itself. (§ 4.5)

---

## 2. Related work

See `docs/novelty_audit.md` for a five-way intersection audit
covering Tsetlin-Machine prior art (Bhattarai LREC 2022, Saha 2023,
Bhattarai EACL 2024, Granmo 2025 GraphTM, Blakely 2025 SGI-TM,
Berge 2025 PD, this author's graphtm-cbr 2026), LLM-safety prior
art (SelfCheckGPT, AlignScore, Semantic Entropy, FActScore,
HaluEval, FactKG, NeMo Guardrails, Llama Guard, ConceptGuard, KEA
Explain, HalluGraph), KG-fact-verification prior art (KG-BERT,
KEPLER, KagNet, QA-GNN, GreaseLM), and counterfactual recourse
prior art (Wachter, GNNExplainer, CF-GNNExplainer, MEG, MMACE,
CLEAR, MMGCF, graphtm-cbr). The intersection is empty.

---

## 3. Method

### 3.1 Dual-graph walking GraphTM

The student is a Hierarchical Graph Tsetlin Machine (Granmo & Saha
2025) configured with depth-5 AND-OR-AND-OR-AND clause trees and
graph-walking forward pass (per-node clause evaluation,
OR-across-nodes, edge information bound via VSA XOR-bind, as in
`graphtm-cbr/paper/paper.md` §3.1). The only architectural change
is that the input graph is a *dual graph* whose node set is the
disjoint union of the claim subgraph and the evidence subgraph,
joined by typed cross-graph edges (§ 3.3).

Three output classes, Support, Refute, NotEnoughInfo. Loss is the
standard MultiClassGraphTsetlinMachine threshold-T feedback
(`paper-a-subword-dep-graphtm` reuses the same library for text
classification, R8 95.05%).

### 3.2 Atomic-claim decomposition + triple extraction

Atomic-claim decomposition uses Qwen2.5-1.5B-Instruct (open weights;
the same checkpoint used in `decoder-attention-distill-graphtm`).
Each prompt is logged with the model SHA. The closed relation
vocabulary (`REL_VOCAB`, 49 canonical relations + `_open_` bucket)
spans the relation types that dominate FEVER, FactKG, HaluEval,
and MedHallBench. The extractor exposes a `Triple` dataclass with
explicit polarity (`+1` / `-1`) and a confidence score.

A dependency-free `RegexTripleExtractor` is also shipped for CI
runs and for benchmarks where a GPU is not available.

### 3.3 Typed cross-graph edges

Three cross-graph edge types are sufficient:

- `align:entity`, claim node and evidence node share the same
  canonical entity label.
- `align:relation`, claim edge and evidence edge share the same
  canonical relation type for some entity pair.
- `contradict:negation`, same (s, r, o) pair across graphs with
  opposite polarity.

These three are exposed to clauses just like within-graph edges;
clauses can therefore directly express patterns like *"if there is
a claim node N with relation `treats` to claim object O and N has
an `align:entity` edge to an evidence node and that evidence node
has a `causes` relation to the matched evidence object, output Refute"*.

### 3.4 Evidence-graph counterfactual recourse

Mirrors `graphtm-cbr/graphtm/recourse/search.py` exactly: greedy
minimum-edit, bounded candidate set from firing clauses, validity
check (RDKit-equivalent here is "valid triple structure + canonical
relation + canonical entity surface form"). Four edit operations:
`add_triple`, `remove_triple`, `swap_relation`, `fix_entity_link`.

Latency target: ≤ 1 s on V100, ≤ 5 s on CPU. (graphtm-cbr's
chemistry-domain recourse hits 2.1 s p50 distilled; ours is
smaller graphs and should be faster.)

### 3.5 SAT-verifiable clause receipt

For every verification decision, ClauseGuard emits a signed
:class:`ClauseReceipt` recording:

- canonical-JSON SHA-256 hashes of the claim triples and evidence
  graph,
- the trained verifier's `model_version_id` (a SHA of its config
  + symbol vocab + edge-type schema),
- the recorded label and the per-class score arithmetic,
- the firing clause IDs and a literal-vector hash,
- the path to a DIMACS CNF encoding the firing pattern.

The receipt is HMAC-SHA-256 signed with a deployer-controlled key.
An auditor with the key verifies the signature; an auditor without
the key still verifies the SAT-replay (Glucose 4) on the DIMACS
CNF and the arithmetic of the per-class scores.

---

## 4. Experiments (planned protocol)

### 4.1 FEVER label accuracy + FEVER score

5 seeds (42, 123, 456, 789, 1337). Train on FEVER train (185K
claims). Evaluate on FEVER `labelled_dev` and on the FEVER blind
test (single submission). Baselines:

- DeBERTa-large fact verifier (current SOTA on FEVER blind).
- BERT-large NLI baseline.
- KG-BERT / KEPLER KG-grounded baselines.

### 4.2 HaluEval-QA / Dialogue / Summarization

Zero-shot: no training on HaluEval. Binary AUROC. Baselines:

- SelfCheckGPT (Manakul et al., 2023), AUROC ~ 0.74.
- AlignScore (Zha et al., 2023), AUROC ~ 0.80.
- Semantic Entropy (Farquhar et al., Nature 2024), AUROC ~ 0.79.
- GPT-4-judge.

### 4.3 FActScore atomic-claim F1

Zero-shot. F1 vs the FActScore reference labels.

### 4.4 MedHallBench

The Article 13/14 medical demonstration case. Zero-shot AUROC.

### 4.5 Adversarial robustness audit

`paper-c`-style: TextFooler, BERT-Attack, paraphrase, negation-flip.
The verifier itself is the system-under-attack.

### 4.6 Recourse evaluation

For 200 Refuted FEVER dev predictions, run the greedy minimum-edit
recourse search. Report flip rate, mean edits to flip, latency
distribution.

### 4.7 SAT-receipt latency

For 200 sample decisions per benchmark, measure end-to-end SAT
verification time (build receipt → write DIMACS → run Glucose 4 →
arithmetic check). Report p50, p95, max.

### 4.8 Auditor user study

n ≥ 10 participants (3 lawyers, 3 clinicians, 4 ML practitioners).
Given a clause receipt only (no claim, no evidence, no model
state), predict the verification label. Target: ≥ 80% predict the
recorded label. This is the EU AI Act Article 14 empirical
demonstration.

---

## 5. Honest framing of expected results

- **AUROC vs SOTA.** Semantic entropy / AlignScore use continuous
  semantic embedding spaces TM cannot natively access. ClauseGuard
  may give up ~3-6 AUROC points to maintain interpretability. We
  report this openly. The value of the contribution is the audit
  trail, not the headline number; where regulation forces
  auditability, the audit trail is the only thing that matters.
- **Triple-extraction precision is the upper bound.** A bad triple
  makes any downstream clause wrong. We separately report
  extraction precision/recall against a 100-claim manual annotation
  so a reader can compute the model's accuracy *conditional* on
  perfect extraction.
- **Open-world entity vocabulary.** Wikidata / UMLS span millions
  of entities. We start with the top-N most frequent per benchmark
  and report how clause-set size scales with N; if the scaling
  destroys interpretability the contribution must be re-scoped.

---

## 6. Limitations

- English only in v0.1.
- Multilingual extension (XLM-R-based extractor) is the obvious
  follow-up.
- The auditor study is exploratory (small n).
- The SAT receipt covers the *recorded* decision only, not the
  model's full Boolean function. Robustness verification against
  bounded perturbations (Saha et al. 2023 protocol) is a clear
  follow-up.

---

## 7. Conclusion

ClauseGuard is the first audit layer for LLM-output verification
that produces, per decision, a SAT-verifiable propositional
certificate and a counterfactual evidence-graph recourse. The
contribution is genuinely novel (audit empty intersection), it
builds on five published pieces of prior work in this lab, and it
lands at the August 2026 EU AI Act enforcement deadline.

The proposition is simple: **make any LLM compliant with EU AI
Act Articles 13 and 14, without modifying the LLM.**

---

## Appendix A. Three case studies (planned)

1. **Medical hallucination intercepted.** A GPT-style assistant
   recommends ibuprofen for a patient with active GI bleeding. The
   evidence KG (UMLS subset) has `ibuprofen --[contraindicated_for]
   -> active_gi_bleed`. ClauseGuard fires a `contradict:negation`
   pattern clause and Refutes. Recourse: "to support this claim,
   evidence would need to assert `ibuprofen --[treats]-> patient_X`
   instead of `contraindicated_for`."
2. **Legal-citation hallucination intercepted.** Claim:
   *"Smith v. Jones, 123 F.3d 456, held that ..."*. Evidence KG
   has no `instance_of(legal_case, smith v. jones)` triple in the
   relevant jurisdiction. ClauseGuard Refutes; recourse points at
   the missing evidence triple.
3. **Mathematical-claim refutation with evidence-edit recourse.**
   Claim: *"the boiling point of water at sea level is 110 °C."*
   Evidence KG: `water --[has_value]-> boiling_point_100c`.
   ClauseGuard Refutes; the recourse offers `fix_entity_link`
   replacing `110c` with `100c` as the minimum edit to flip the
   verdict.
