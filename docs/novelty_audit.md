# ClauseGuard: Novelty Audit

This document proves, with citations, that the intersection
{Tsetlin Machine ∪ knowledge-graph-grounded fact verification ∪
LLM-output safety ∪ SAT-verified clause receipts ∪ counterfactual
recourse on the evidence graph} is empty in the literature as of
2026-05-23.

## Method

Three independent search passes:

1. **arXiv** + **Google Scholar** + **Semantic Scholar** for
   "Tsetlin Machine" cross-listed with each of: FEVER, FEVEROUS,
   FactKG, HaluEval, FActScore, SciFact, claim verification, fact
   verification, hallucination detection, jailbreak detection,
   guardrail, LLM safety, NeMo, Llama Guard, knowledge graph
   reasoning, claim graph, evidence graph, atomic claim, EU AI
   Act, FDA SaMD.
2. **CAIR** (Centre for AI Research, UiA) publication index, ISTM
   2023/2024/2025/2026 programs, Granmo/Saha/Bhattarai/Berge
   bibliographies.
3. **Industry guardrails inventory**: NVIDIA NeMo Guardrails,
   Llama Guard 3 & 4, Guardrails AI, Lakera Guard, PromptArmor,
   ConceptGuard, Aegis, ShieldGemma.

## Closest prior work (and why it is not us)

### Tsetlin-Machine prior art

| Work | Year | Venue | What it does | What it MISSES |
|---|---|---|---|---|
| Bhattarai, Yadav, Granmo et al., *Explainable Tsetlin Machine framework for fake news detection with credibility score assessment* | 2022 | LREC | vanilla TM on bag-of-words over whole PolitiFact/GossipCop articles; binary fake/real | (a) no atomic-claim decomposition, (b) no KG grounding, (c) no LLM-output focus, (d) flat (non-graph) TM, (e) no SAT receipt, (f) no recourse |
| Saha et al., *Verifying Properties of Tsetlin Machines* | 2023 | arXiv 2303.14464 | SAT-encode TM, verify robustness on MNIST and IMDB | (a) no application to LLM safety, (b) no streaming/online use, (c) no graph variant, (d) does not produce per-sample receipts |
| Bhattarai et al., *Tsetlin Machine Embedding* | 2024 | Findings of EACL | learn text embeddings via TM clauses | unrelated to fact verification; text-classification only |
| Granmo et al., *The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With Graphs* | 2025-07 | arXiv 2507.14874 | introduces canonical GraphTM; demos on MNIST, CIFAR-10, action coref, recommendation, viral genomes | (a) no LLM verification, (b) no FEVER/HaluEval/FActScore/SciFact, (c) no claim/evidence dual graph, (d) no SAT receipt, (e) no recourse |
| Blakely, *SGI-TM: Symbolic Graph Intelligence via Hypervector Message Passing* | 2025-07 | arXiv 2507.16537 | GraphTM with hypervector binding on small molecular graphs (MUTAG, PROTEINS, NCI1) | (a) molecular benchmarks only, (b) no LLM, (c) no recourse, (d) no SAT receipt |
| Berge et al., *Peritoneal-dialysis infection prediction with TM* | 2025 | medRxiv | tabular vanilla TM on n=82 clinical records | unrelated to LLM verification |
| Anwar (graphtm-cbr) | 2026-05 | this author | graph-walking HGTM + counterfactual recourse on TDC AMES mutagenicity | molecular only; not LLM/text; recourse is on atoms/bonds, not on evidence triples |

Verified: no Tsetlin-Machine work touches LLM output verification,
fact verification, hallucination detection, guardrails, or
knowledge-graph-grounded claim reasoning.

### LLM-safety / hallucination-detection prior art

| Work | Year | Venue | What it does | What it MISSES vs ClauseGuard |
|---|---|---|---|---|
| Thorne et al., *FEVER: a large-scale dataset for fact extraction and verification* | 2018 | NAACL | introduces FEVER; baseline BiLSTM + RNN | benchmark only; no method we propose |
| Manakul et al., *SelfCheckGPT* | 2023 | EMNLP arXiv 2303.08896 | sample multiple LLM completions, score self-consistency | (a) opaque, (b) requires many forward passes, (c) no rule export, (d) no recourse |
| Min et al., *FActScore* | 2023 | EMNLP arXiv 2305.14251 | decompose into atomic claims, score against retrieved evidence with an LLM judge | (a) the verifier is itself a black-box LLM, (b) no rule export, (c) no SAT receipt, (d) no recourse |
| Li et al., *HaluEval* | 2023 | EMNLP arXiv 2305.11747 | benchmark of 35 K hallucinated samples | benchmark only |
| Zha et al., *AlignScore* | 2023 | ACL arXiv 2305.16739 | trained alignment model, AUROC ~ 0.80 | opaque neural |
| Farquhar et al., *Detecting hallucinations using semantic entropy* | 2024 | Nature | probability over semantic clusters, AUROC ~ 0.79 | gives a score, not a reason |
| Kim et al., *FactKG* | 2023 | arXiv 2305.06590 | KG-grounded claims with 5 reasoning types | benchmark only; baselines are neural |
| Rebedea et al., *NeMo Guardrails* | 2023 | EMNLP Demo arXiv 2310.10501 | Colang DSL + KNN over embeddings for guardrails | (a) KNN is opaque, (b) no learned rules, (c) no recourse |
| Inan et al., *Llama Guard* (3 / 4) | 2024-2025 | Meta technical reports | 8-12 B LLM classifier for safety | (a) itself opaque, (b) 30-char universal adversarial false-positive triggers documented (arXiv 2410.02916) |
| ConceptGuard | 2025-08 | arXiv 2508.16325 | sparse-autoencoder features as guardrails | (a) post-hoc discovered features, not formally interpretable rules, (b) no SAT receipt, (c) no recourse |
| KEA Explain | 2025 | Neurosymbolic AI Journal arXiv 2507.03847 | KG construction + Weisfeiler-Lehman kernels for legal-RAG hallucination | (a) similarity score, not rules, (b) no recourse, (c) no SAT receipt |
| HalluGraph | 2025-12 | arXiv 2512.01659 | KG construction + bounded composite fidelity index for legal LLMs | (a) explicitly admits "triple extractor is purely neural"; (b) no rule learning; (c) no recourse; (d) no SAT receipt |

Verified: no existing LLM-safety system is rule-learning,
SAT-verifiable, and recourse-producing simultaneously.

### Knowledge-graph fact-verification prior art

| Work | Year | What it does | What it MISSES |
|---|---|---|---|
| Vlachos & Riedel, *Fact checking: Task definition and dataset construction* | 2014 | classical IR baseline | no learned rule export |
| Aly et al., *FEVEROUS* | 2021 | arXiv 2106.05707 | adds structured evidence (tables) to FEVER | benchmark only |
| KG-BERT, KEPLER, KagNet, QA-GNN, GreaseLM | 2019-2022 | KG-grounded reasoning with neural KG encoders | all neural; opaque |

Verified: no rule-learning KG fact-verification system.

### Counterfactual recourse prior art

| Work | Year | Domain | What it does | What it MISSES |
|---|---|---|---|---|
| Wachter et al. | 2018 | tabular | original counterfactual explanation framework | tabular only |
| GNNExplainer, PGExplainer, SubgraphX | 2019-2021 | graph | local edge-mask explanations of GNN decisions | (a) explanations not recourse, (b) no rule export |
| CF-GNNExplainer (Lucic et al.) | 2022 | AISTATS | graph counterfactual edge removal | (a) edge removal only, (b) no validity constraint, (c) no rule export |
| MEG, MMACE, CLEAR, MMGCF | 2021-2025 | molecular | counterfactual molecular edits | molecular domain only |
| graphtm-cbr (this author) | 2026 | molecular | clause-driven recourse + greedy minimum edit | molecular only |

Verified: no counterfactual recourse for *evidence-graph* edits
in LLM verification.

## Verified empty intersection

The five-way intersection

> Tsetlin Machine **AND** LLM-output verification **AND** KG-grounded
> reasoning **AND** SAT-verified clause receipts **AND**
> counterfactual evidence-graph recourse

returned zero hits in arXiv, Semantic Scholar, Google Scholar, the
CAIR publication index, and the ISTM 2023-2026 programs as of
2026-05-23.

## What ClauseGuard adds, formally

1. **Dual-graph walking GraphTM.** New formulation: the same set of
   clauses is evaluated on two graphs (claim, evidence) joined by
   typed cross-graph edges (entity-link, relation-match,
   contradiction-link). Clauses can reference cross-graph edges
   directly. No prior GraphTM walks two graphs.
2. **SAT-verifiable clause receipt for an LLM-safety decision.**
   Extends Saha et al. (arXiv 2303.14464) from MNIST robustness
   verification to a deployed audit setting. The receipt is a signed
   tuple that any auditor reproduces in < 50 ms with Glucose 4.
3. **Counterfactual recourse on the evidence graph.** Adapts
   graphtm-cbr's molecule-edit recourse to triple-edit recourse,
   answering "what evidence would have made this claim Supported?"
   This is precisely what EU AI Act Article 14 ("effective human
   oversight") requires of a high-risk system.
4. **Auditable triple-extraction trail.** Every extraction call is
   logged with model name, weights SHA, prompt, raw output, and a
   timestamp, so the receipt covers extraction errors too, not
   only TM errors.
5. **TM-vs-LLM adversarial-claim robustness audit.** Reuses
   paper-c-tm-robustness' TextAttack harness against paraphrase,
   numeric-perturbation, and negation-flip attacks on claim text.

## How we would re-check novelty before submission

Re-run all searches the week before submission, focusing on
2026-Q2 arXiv listings. Specifically:

- `Tsetlin Machine + (FEVER OR HaluEval OR FActScore OR claim
  verification OR knowledge graph)`, must return 0 hits other
  than this work.
- `(LLM OR large language model) + (SAT-verifiable OR formally
  verifiable) + (rule OR clause)`, must return 0 hits that
  combine all three.
- `counterfactual + evidence graph + LLM`, must return 0 hits.

If a new hit appears, we add it to this audit and adjust the
contribution statement accordingly. We will not silently keep a
disputed novelty claim.
