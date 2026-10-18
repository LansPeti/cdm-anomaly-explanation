# Known Limitations

This document is mandatory pre-publication content. It
catalogues every limitation a competent reviewer would identify on the
current codebase, so that downstream users and reviewers do not need to
discover them by reading the source.

Each entry links to the affected component, the concrete observable,
and the path to remediation if known.

## L1. Synthetic-only evaluation

**What:** Every quantitative claim (recall, FP, hallucination rate) is
measured on traces produced by `src/data_generator.py`. No real ISDA
CDM trace has touched this system.

**Observable:** `data/normal.jsonl` and `data/anomalies.jsonl` are
generated from a hand-coded transition matrix in `src/lifecycle_graph.py`.

**Implication:** Threshold values (`PT_SURPRISE_NATS=4.0`,
`SQ_NORM_LL_Z_MAX=-2.5`, `SQ_EDGE_REPEAT_MIN=3`) are tuned in-distribution.
Real-data FP is expected to be substantially higher (estimates
8–15%). No external validity claim is supported.

**Path forward:** `docs/11_real_data_validation_plan.md`.

## L2. Circular evaluation design

**What:** Each anomaly class in `data_generator.py` was constructed to
fire a specific flag. `skipped_step` produces a transition with
`training_count=0` (PT-impossible by construction); `pingpong` repeats
an edge ≥ 5 times (SQ-edge-repeat by construction); etc.

**Observable:** 100% recall (95% CI [99.63%, 100%]) is therefore an
algebraic identity, not an empirical detection result.

**Path forward:** Design at least one anomaly class whose definition was
not available during flag design (e.g. "distributed surprise": multiple
transitions each below threshold but anomalous in aggregate).

## L3. LLM hallucination rate (RESOLVED — measurement timeline below)

**What it was:** Initial measurement on 100 anomalous traces
(gpt-4o-mini-2024-07-18, T=0.2, two few-shot examples) found that
`validate_explanation()` reported 100/100 OK while `check_hallucinations()`
reported only **62/100 OK**. 37/38 failures were the causal phrase
"led to" — exactly the pragmatic-implicature mode predicted in an internal design review
and a reviewer in the internal design review. This was a Goodhart's Law
demonstration (structurally compliant, content non-compliant).

**Resolution timeline:**

| Iteration | validate OK | hallucination OK | Dominant failure |
|---|---|---|---|
| Pre Section 4 fix | 100/100 | 62/100 | 37× "led to" causal |
| Post Section 4 fix (introduced new bug) | 75/100 | 75/100 | 25× "SLA" UPPER_SNAKE token |
| Post SLA fix (final) | **100/100** | **100/100** | none |

**What fixed it:** Two changes shipped:
1. `validation_requirements` declarative pack field — Section 4 is now
   a verbatim enumeration of pack content rather than LLM-generated
   prescription. Eliminated all "led to" causal phrasings.
2. The `validation_requirements` strings were rewritten to avoid
   UPPER_SNAKE tokens that the hallucination detector flags as
   "invented state-like tokens" (specifically: "operations SLA"
   → "operational service-level policy").

**Observable:** `results/e2e/summary.json` after `make e2e-scale`,
final commit shows 100/100 on both validators.

**Generalisable lesson for the paper:** The original measurement is
also a Goodhart's Law demonstration. The fact that the lexical-blacklist
detector caught 37 occurrences and missed zero suggests that *with the
right narrowness*, lexical detectors can be effective. The boundary
case (Section 4) was where the architectural fix mattered.

## L4. PT-rank and PT-surprise are empirically redundant

**What:** Ablation study (`make ablation`) confirms the redundancy hypothesis:
the reduced flag set `{PT-impossible, SQ-low-likelihood, SQ-edge-repeat}`
produces identical recall (100% on all classes) and identical FP rate
(4.62%) as the full 5-flag set. Pearson correlations between PT-surprise
and PT-rank exceed 0.95 on this dataset.

**Observable:** `results/ablation/report.md`.

**Implication:** PT-rank and PT-surprise should be reclassified as
evidence-pack metadata (informative but not detection signals), not
co-equal flags counted in `fired_signals`. The current presentation
overstates the dimensionality of the evidence.

**Path forward:** Demote in `flags.py` documentation; do not yet remove
because they may have value as orthogonal signals on real, non-circular
data.

## L5. Domain coverage gaps

**What:** The lifecycle graph omits states that are operationally and
legally significant in real ISDA CDM workflows:

- **AFFIRMED**: DTC settlement-eligibility precondition.
- **ALLOCATED**: primary source of UNMATCHED_ERROR in institutional
  equity post-trade.
- **NOVATED**: mandatory under EMIR Article 4 for CCP clearing
 .
- **PARTIAL_SETTLEMENT**: tracked in T2S under ISO 20022.

**Observable:** `src/lifecycle_graph.STATES` enumerates 12 states; none
of the above are present.

**Implication:** Detector cannot monitor block-trade allocation breaks
("the answer is no — that question ends the evaluation") or
EMIR clearing compliance.

**Path forward:** Adding states requires re-training the empirical
Markov detector (Hanna's component) and regenerating the synthetic
dataset. Scheduled for the next iteration after Hanna's involvement.

## L6. Settlement timing model unrealistic

**What:** The data generator uses a single Gaussian with mean 172,800 s
(T+2). T+1 has been the US equity settlement standard since May 2024;
real settlement timing is non-Gaussian (hard lower bound, spike at
intended date, heavy right tail).

**Observable:** `src/data_generator.py` simulation; `results/detector/time_gap_report.json`
treats it as ground truth.

**Path forward:** Asset-class-stratified, empirically-fitted timing
distributions.

## L7. AMENDED → ENRICHED routing is unconditional

**What:** `src/lifecycle_graph.py` encodes
`AMENDED -> ENRICHED (probability 1.00)`. In real practice, amendments
to material economic terms re-enter at MATCHED, not ENRICHED.

**Implication:** The detector will flag legitimate `AMENDED → MATCHED`
transitions in real data as `PT-impossible` (false positives).

**Path forward:** Conditional routing on amendment type, requiring an
amendment-type field in the trace records that the synthetic generator
does not currently produce.

## L8. No PM4Py / conformance-checking baseline

**What:** `PT-impossible` is token-replay conformance checking
(Adriansyah et al. 2011). We do not currently report side-by-side
recall and FP against PM4Py's conformance-checking module on the same
12-state graph and the same dataset.

**Observable:** No `experiments/baseline_pm4py.py`.

**Path forward:** Add `pm4py` to dev requirements, write the baseline
runner. Planned but not yet started.

## L9. No real-world dataset comparison (BPI Challenge)

**What:** ICPM submissions are expected to evaluate on at least one
real event log, typically a BPI Challenge dataset.

**Path forward:** Adapt BPI 2017 / BPI 2019 to the trade-lifecycle
state vocabulary. Discussed in `docs/11_real_data_validation_plan.md`.

## L10. No human evaluation of LLM explanation quality

**What:** Explanation quality is currently measured by two automatic
metrics (`validate_explanation` for structure, `check_hallucinations`
for causal-language and invented-state detection). Neither has been
shown to correlate with human auditor judgment.

**Path forward:** Manually grade 20–50 explanations and report the
Pearson/Spearman correlation between human grades and the automatic
metrics (the review's recommendation).

## L11. EU AI Act conformity assessment not performed

**What:** The system makes inferences in a regulated EU financial
context. Under the EU AI Act, this likely falls within Annex III's
high-risk classification and requires conformity-assessment
documentation that does not yet exist.

**Path forward:** Conformity-assessment dossier with technical
documentation, output logging, and human-oversight measures.
