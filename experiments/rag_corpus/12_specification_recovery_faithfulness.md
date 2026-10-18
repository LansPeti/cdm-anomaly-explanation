# Specification-Recovery Faithfulness

A formal framework for measuring whether a constrained-explanation
pipeline preserves the semantic content of an injected anomaly across
the full detector → evidence-pack → narrator chain.

This document is the theoretical companion of
`src/specification_recovery.py` and `experiments/recovery_faithfulness.py`.

---

## 1. Motivation

An internal design review identifies the synthetic-only
evaluation as the project's strongest external-validity weakness: every
recall and FP number is in-distribution against data engineered to
exhibit the four anomaly classes. Standard plausibility / completeness
metrics for explanations require human annotation, which (a) does not
scale, (b) cannot be reproduced across teams, and (c) cannot evaluate
**faithfulness** — whether the explanation reflects the actual cause of
the anomaly rather than something the LLM imagined.

The proposed framework turns the synthetic property of the data into a
**measurement instrument**: if we know what anomaly was injected (because
we injected it), and we have an automatic way to recover that statement
from the final explanation, then a faithful pipeline must satisfy

\[
\text{semantic similarity}(s_\text{spec}, s_\text{recovered}) \to 1
\]

where $s_\text{spec}$ is the natural-language anomaly specification used
to drive trace generation, and $s_\text{recovered}$ is reconstructed by
an independent LLM reading only the explanation produced by the pipeline.

## 2. Formal definition

Let:

- $\mathcal{S}$ = the space of admissible natural-language anomaly
  specifications. Each $s \in \mathcal{S}$ describes one anomaly type
  in plain English, paired with an optional structured class hint:
  $s = (s_\text{NL}, c)$ where $c \in \{\text{skipped step},
  \text{impossible}, \text{pingpong}, \text{unlikely branch}, \bot\}$
  and $\bot$ denotes "no class hint, free-form anomaly".

- $\mathcal{T}$ = the space of trade-lifecycle traces (sequences of
  states from `lifecycle_graph.STATES`).

- $M : \mathcal{S} \to \mathcal{T}$ = the **trace materializer**:
  takes a specification and produces a concrete trace exhibiting it.

- $\Pi : \mathcal{T} \to \Sigma$ = the **explanation pipeline**:
  the existing `detector → evidence_pack → AnomalyExplainer` chain
  producing a markdown explanation in language $\Sigma$.

- $R : \Sigma \to \mathcal{S}$ = the **anomaly recoverer**: a separate
  LLM step (with no access to the original specification) that reads
  the explanation and emits a recovered specification.

- $\sigma : \mathcal{S} \times \mathcal{S} \to [0, 1]$ = a
  **semantic similarity** function. Two implementations are tracked:
  $\sigma_\text{judge}$ (LLM-as-judge with explicit prompt) and
  $\sigma_\text{embed}$ (cosine similarity of OpenAI embeddings).

**Recovery score** for a single specification is

\[
\rho(s) = \sigma\!\big(s, R(\Pi(M(s)))\big).
\]

**Specification-Recovery Faithfulness** of a pipeline $\Pi$ over a
specification distribution $D$ is

\[
\mathcal{F}(\Pi; D) = \mathbb{E}_{s \sim D}\,\rho(s).
\]

The empirical estimator on a finite sample is the mean of $\rho$ over
sampled specifications, with a Clopper-Pearson confidence interval if
$\rho$ is binarised at a threshold (e.g. $\mathbf{1}[\rho > 0.7]$).

## 3. Algorithm

```
INPUT:  D — a distribution over (s_NL, c) pairs
        Π — the pipeline under test
        n — number of samples
        σ — similarity function

FOR i = 1..n:
    s ← sample from D                                # natural-language spec
    t ← M(s)                                         # synthesise trace
    e ← Π(t)                                         # generate explanation
    s' ← R(e)                                        # recover spec
    record ρᵢ = σ(s, s')

REPORT:  mean(ρ), 95% CI(mean), per-class breakdown
```

## 4. Why the loop is not circular in the reviewer's sense

The circularity critique applies to evaluations where the
**same artefact** is used as both query and oracle. Here the loop has
**three independent stages**:

| Stage | Source of state | Visible to next stage? |
|---|---|---|
| Specification $s$ | LLM-A (specifier) | NO — only via $t$ |
| Trace $t$ | Deterministic materializer $M$ | NO — only via $e$ |
| Explanation $e$ | Pipeline $\Pi$ | YES, but with no access to $s$ |
| Recovered $s'$ | LLM-B (recoverer, separate session) | scored against $s$ |

The recoverer $R$ never sees $s$ directly; it must reconstruct it from
$e$ alone. If $\Pi$ corrupts the semantic content (e.g. by
hallucinating a different anomaly class), $R$ produces a different $s'$
and $\rho$ drops. This is a **structural property** of the pipeline,
not a content-classification accuracy.

## 5. Distinguishing this from existing faithfulness metrics

| Metric | Domain | Requires | Provides |
|---|---|---|---|
| Attention rollout (Abnar & Zuidema 2020) | Transformer internals | Model weights | Heatmap |
| LIME / SHAP (Ribeiro 2016, Lundberg 2017) | Tabular / text | Local perturbations | Feature importance |
| Faithfulness-by-erasure (Atanasova 2020) | Token-level explanation | Oracle classifier | Edit distance |
| Human evaluation | Anything | Annotator pool | Rating |
| **Specification-Recovery Faithfulness** | **Constrained-explanation pipelines** | **LLM with reasoning ability** | **Semantic preservation score** |

The novel contribution is not the *idea* of round-trip evaluation —
back-translation has been used in machine translation since at least
Sennrich (2016). The novelty is its application to **explanation
pipelines for anomaly detection**, where the round-trip is from
**structured anomaly specification** to **natural-language explanation**
and back. We are not aware of prior published work that uses this exact
loop to measure faithfulness in process-mining or post-trade contexts.

## 6. Limitations and risks

**L1 (specifier bias).** If LLM-A and LLM-B share training data, they
may share systematic blind spots. Reportable risk: $\rho$ is upper-
bounded by the joint coverage of the two LLMs over the specification
space. Mitigation: cross-LLM ablation (specifier and recoverer from
different model families).

**L2 (recoverer overfitting to surface form).** If $R$ is trained on
similar prose styles, it may over-rely on lexical cues and over-recover.
Mitigation: pair the lexical-similarity metric $\sigma_\text{embed}$
with a structural similarity metric that requires correct anomaly-class
identification.

**L3 (specification-distribution shift).** If $D$ is biased toward
anomaly types the pipeline can already explain, $\mathcal{F}$ is
optimistic. Mitigation: include a stress-test mode where the specifier
generates novel anomaly types not seen in the few-shot examples.

**L4 (similarity-function dependence).** Different $\sigma$ produce
different $\rho$ for the same $(s, s')$ pair. Mitigation: report all
three of $\sigma_\text{judge}$, $\sigma_\text{embed}$, and a strict
class-match indicator separately.

## 7. Implementation roadmap

| Phase | Deliverable | Owner |
|---|---|---|
| 7.1 | `src/specification_recovery.py` skeleton + interfaces | us |
| 7.2 | `M` (trace materializer) reusing `data_generator.ANOMALY_INJECTORS` | us |
| 7.3 | `AnomalySpecifier` + `AnomalyRecoverer` LLM wrappers | us |
| 7.4 | Two $\sigma$ implementations (LLM-judge, embedding) | us |
| 7.5 | `experiments/recovery_faithfulness.py` runner | us |
| 7.6 | Unit tests (offline, mocked LLMs) | us |
| 7.7 | First empirical run, small $n$ | us |
| 7.8 | Cross-LLM extension (Claude / Llama / Mistral as $R$) | future |
| 7.9 | Paper §6 first draft | us |

## 8. Open research questions

These shape the experiments and the paper's discussion:

1. Does $\mathcal{F}$ stratify by anomaly class? Are some structurally
   easier to recover than others?
2. How does $\mathcal{F}$ change when we ablate the
   `validation_requirements` field (the contribution of §4)?
3. Is $\mathcal{F}$ stable under prompt perturbation in $\Pi$? Under
   model-version drift in the explainer?
4. What is the noise floor of $\mathcal{F}$ on a *random* explanation
   (not tied to the specification)?
5. Does the noise-floor experiment provide an empirical lower bound for
   "trivially achievable" $\mathcal{F}$ that the pipeline must exceed?
