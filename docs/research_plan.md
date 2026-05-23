# ClauseGuard: Research Plan

## Premise

LLMs hallucinate, and existing guardrails are themselves opaque
neural models. EU AI Act Article 13 (transparency) and Article 14
(human oversight) come into force for high-risk Annex-III deployers
in August 2026 and have no compliant solution today. ClauseGuard is
the first interpretable, SAT-verifiable LLM-output verification
layer.

## Hypothesis

1. A dual-graph walking Graph Tsetlin Machine, trained on
   FEVER+FActKG+HaluEval with KG-grounded evidence, achieves
   competitive label accuracy on FEVER (≥ 75% blind) while
   producing rule sets of ≤ 500 clauses with median clause length
   ≤ 6 literals.
2. SAT verification of clause receipts is feasible in ≤ 50 ms
   per claim using Glucose 4.
3. Counterfactual evidence-graph recourse on Refuted claims
   reaches ≥ 80% flip rate within ≤ 3 edits.
4. ClauseGuard is robust to TextAttack paraphrase / numeric
   perturbation / negation flip at ≥ 70% robust accuracy on the
   Counterfactual-IMDb-style perturbation protocol from paper-c.

## Scope

In scope: English-only fact verification on
- FEVER (Wikipedia evidence, 185 K claims, 3-class)
- HaluEval (35 K LLM-hallucinated samples, binary AUROC)
- FActScore (atomic-claim factual precision, GPT-3.5/4 outputs)
- MedHallBench (medical LLM hallucinations, AUROC)

Out of scope:
- Multilingual fact verification (clear follow-up; XLM-R extension).
- Generative correction (we point at the evidence that would fix
  the claim; we do not rewrite the LLM output).
- Online federation across multiple LLM providers (architecture
  sketch only).
- Open-domain reasoning beyond top-N Wikidata / UMLS entities.

## Phases

### Phase 1: Triple extraction (E1)

Use Qwen2.5-1.5B-Instruct (already in `$QWEN_TEACHER_DIR`) as the
atomic-claim decomposer and (s, r, o) triple extractor. Verified
choice: open weights, deterministic with fixed seed and temperature
0, traceable (SHA-pinnable).

- Prompt template in `src/clauseguard/extraction/prompts.py`.
- Output format: JSON-lines, one per claim, with raw extractor
  trace for audit.
- Stress-test on 200 FEVER train claims; manual eyeball pass.
- Acceptance: triple precision ≥ 0.85, recall ≥ 0.70 on a held-out
  100-claim manual annotation set.

Scripts:
- `experiments/extract_triples_fever.py`
- `experiments/extract_triples_halueval.py`
- `experiments/extract_triples_factscore.py`
- `experiments/extract_triples_medhall.py`

### Phase 2: Graph construction (E2)

Build claim graph + evidence graph per sample. Combine into a
single `TypedGraph` with cross-graph edges
{entity-link, relation-match, contradiction-link}.

- `src/clauseguard/graphs/claim_graph.py`, `evidence_graph.py`,
  `dual_graph.py`.
- Sanity check: `tests/test_graphs.py` verifies isomorphism under
  triple permutation and round-trip from JSON.

### Phase 3: Train baseline GraphTM (E3)

- Phase 3a: claim graph only (no evidence), sanity baseline.
  Expected behaviour: chance-level on FEVER (because we cannot
  verify a claim without evidence).
- Phase 3b: evidence graph only, also chance, but for the opposite
  reason (we cannot verify *this specific* claim without seeing it).
- Phase 3c: dual graph, the contract. Target FEVER dev label
  accuracy ≥ 70%.

Hyperparameters in `configs/base.yaml`.

### Phase 4: Distillation (E4), paper-b's pattern, applied

Take a strong LLM judge (Llama-3.1-8B-Instruct or GPT-4-judge) and
soft-label a chunk of FEVER train; train the GraphTM on the
hard+soft label combination. Mirrors graphtm-cbr's GIN→HGTM
distillation, but the teacher is now a fact-checking LLM.

Hypothesis: distillation lifts the GraphTM's calibration without
hurting interpretability (clause count stays ≤ 500).

### Phase 5: SAT verification (E5)

For 200 sample predictions per benchmark, encode the firing clauses
+ literal vector + label as a DIMACS CNF. Run Glucose 4 (via
python-sat). Measure verification latency. Report median p50, p95.

Target: median ≤ 50 ms; p95 ≤ 200 ms.

### Phase 6: Recourse (E6)

For Refuted predictions on FEVER dev, run greedy-minimum
evidence-graph edit search. Measure:

- Flip rate within ≤ 3 edits.
- Mean number of edits per successful recourse.
- Wall-clock per recourse (target ≤ 1 s on V100, ≤ 5 s CPU).

### Phase 7: Adversarial robustness (E7), paper-c's pipeline

Run TextAttack TextFooler + BERT-Attack on FEVER dev claims, and
the Counterfactual-Edits protocol from paper-c on the same claims.
Compare ClauseGuard vs BERT/DeBERTa/Llama-Guard baselines.

### Phase 8: Auditor user study (E8)

n ≥ 10 participants (mix of: 3 lawyers, 3 clinicians, 4 ML
practitioners). Given a clause receipt only (no claim, no model),
predict the verification label. Target: ≥ 80% agreement with the
recorded label.

This is the EU-AI-Act-Art-14 "effective human oversight" empirical
demonstration.

### Phase 9: Generalization (E9)

Apply the trained model to HaluEval, FActScore, MedHallBench
*without retraining* (zero-shot). Measure AUROC. Compare to
GPT-4-judge, SelfCheckGPT, AlignScore, Semantic Entropy.

Expectation: ClauseGuard underperforms SOTA by 2-6 AUROC points
but produces SAT-verifiable rules that no competitor can produce.
Honest framing.

## Evaluation methodology

- 5 seeds per configuration: 42, 123, 456, 789, 1337 (matches
  paper-a, paper-b, paper-c).
- Paired Wilcoxon signed-rank tests via `src/clauseguard/eval/stats.py`
  for ablation comparisons.
- Bootstrap 95% CI via `bootstrap_ci`, n=1000 resamples.
- Report mean ± std and median. Never headline the best seed.
- Bonferroni-correct across the 4-benchmark x 3-method comparison
  table.
- All per-sample predictions and clause traces saved to JSONL for
  audit.

## Compute budget (V100-SXM3-32GB)

| Phase | GPU-hours | Wall clock (1× V100) |
|---|---|---|
| 1: extraction | 12 | 1.5 days |
| 2: graphs (CPU heavy) | 4 | half day |
| 3: train baseline | 6 | half day |
| 4: distillation | 15 | 2 days |
| 5: SAT verify | 1 | half hour |
| 6: recourse | 3 | 4 hours |
| 7: adversarial | 12 | 1.5 days |
| 8: user study (offline) |, | 2 weeks elapsed |
| 9: generalization | 4 | half day |
| **Total** | **~57** | **~7 days** |

Parallelize seeds 2× on a 2-GPU node if available.

## Risks (honest)

1. **Triple-extraction quality bottlenecks the system.** Mitigation:
   audit extraction precision/recall separately; report it as a
   first-class number; provide a manual-annotation slice for fair
   comparison to GPT-4 extraction. If Qwen-1.5B precision < 0.80,
   escalate to Qwen-7B or DeBERTa-based REBEL.
2. **GraphTM clause count explodes with open vocabulary.** Mitigation:
   restrict to top-N Wikidata/UMLS entities; report per-claim clause
   set size and length distributions; gracefully degrade to a
   confidence-weighted "abstain" label if no clause fires above
   threshold.
3. **AUROC gap vs neural SOTA.** Honest framing: ClauseGuard's
   contribution is the audit trail, not the headline number. The
   AUROC must be *competitive* (within ~5 points of SOTA) but does
   not need to win. Report it openly.
4. **SAT-verification time grows with clause set.** Mitigation:
   per-sample CNF is bounded by firing clauses (typically < 50), not
   total clauses; Glucose 4 handles this trivially. Worst case
   profiled and reported.
5. **Auditor user study has small n.** Mitigation: pre-register
   the protocol and sample size; report effect size and CI; treat
   the study as exploratory evidence, not the headline.

## Deliverables (in this repo)

- `experiments/train_fever.py` and dataset-specific eval scripts.
- 5-seed result JSONs per benchmark.
- `results/clauseguard_5seeds.json` aggregated.
- `paper/paper.md` with locked numbers.
- `paper/figures/*` confusion matrices, clause-length histograms,
  recourse trace examples.
- 3 case studies in `paper/case_studies/`:
  1. Medical hallucination intercepted by ClauseGuard (MedHallBench).
  2. Legal-citation hallucination intercepted (FActScore Legal).
  3. Mathematical-claim refutation with evidence-edit recourse.
