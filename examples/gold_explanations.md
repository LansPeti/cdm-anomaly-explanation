# Gold-standard explanations

Hand-crafted reference outputs for each example pack. These are what the
LLM should produce given the corresponding pack. Used as:

  - **few-shot demonstrations** in `src/llm_explainer.py`
  - **upper-bound reference** in `eval/eval_runner.py`

Each explanation is followed by a "field-citation audit" listing exactly
which pack fields it references. If you change a pack, update the
explanation and re-run `pytest tests/` and `python -m eval.eval_runner`.

---

## 1. `pack_normal.json` — happy path

**Expected output:**

```
Normal trace, no findings.
```

That literal string is the only acceptable output for non-anomalous packs.
This rule is enforced by `validate_explanation()` and any LLM output that
deviates is treated as a failure.

---

## 2. `pack_skipped_step.json` — CAPTURED → MATCHED (skips ENRICHED)

```markdown
## 1. Observed Trace
The trade followed: NEW -> EXECUTED -> CAPTURED -> MATCHED -> CLEARED -> SETTLED (trace).
At step 2, the transition CAPTURED -> MATCHED was recorded
(transition_findings[0].observed_transition).

## 2. Why It Is Flagged
- The transition CAPTURED -> MATCHED was never observed in normal training data
  (transition_findings[0].training_count = 0, fired_flags includes
  PT-impossible).
- Its model probability is 0.0001
  (transition_findings[0].p_observed), consistent with a structurally
  invalid step.
- The trace's overall log-likelihood is 2.45 standard deviations below
  the normal distribution (sequence_level.norm_ll_z_score), reinforcing
  the per-transition signal.

## 3. What Normal Would Look Like
The model's expected continuation from CAPTURED was ENRICHED with
probability 0.96 (transition_findings[0].expected_top3[0]). A normal
trace would have inserted ENRICHED between CAPTURED and MATCHED.

## 4. Recommended Validation Step
Verify whether an ENRICHED event was processed but not logged, or
whether the matching step was reached without enrichment. The pattern
suggests a missing intermediate event rather than a routing error
(verdict.fired_signals).
```

**Citations used:** `trace`, `transition_findings[0].observed_transition`,
`transition_findings[0].training_count`, `transition_findings[0].p_observed`,
`sequence_level.norm_ll_z_score`,
`transition_findings[0].expected_top3[0]`, `verdict.fired_signals`.

---

## 3. `pack_impossible.json` — NEW → FAILED_SETTLEMENT

```markdown
## 1. Observed Trace
The trade started at NEW and immediately recorded FAILED_SETTLEMENT
(trace, transition_findings[0].observed_transition), then proceeded
through the canonical CAPTURED -> ENRICHED -> MATCHED -> CLEARED ->
SETTLED path.

## 2. Why It Is Flagged
- NEW -> FAILED_SETTLEMENT was never seen in normal training data
  (transition_findings[0].training_count = 0, PT-impossible fired).
- The transition is the model's 9th most likely successor
  (transition_findings[0].rank_among_candidates), with probability
  0.0001 (transition_findings[0].p_observed).
- The trace's normalized log-likelihood is 2.67 sigma below the normal
  baseline (sequence_level.norm_ll_z_score).

## 3. What Normal Would Look Like
From NEW, the canonical successor is EXECUTED with probability 0.97
(transition_findings[0].expected_top3[0]). The only other plausible
direct successor is REJECTED at 0.03
(transition_findings[0].expected_top3[1]). FAILED_SETTLEMENT is not
reachable directly from NEW under the model.

## 4. Recommended Validation Step
Determine whether this record represents a corrupted import (e.g.,
truncated trace from a downstream system that already failed
settlement), or a genuine schema-level event misclassification
(verdict.fired_signals).
```

---

## 4. `pack_pingpong.json` — 8 amendment cycles

```markdown
## 1. Observed Trace
The trade entered NEW and progressed normally to ENRICHED, then cycled
8 times through ENRICHED -> UNMATCHED_ERROR -> AMENDED -> ENRICHED before
proceeding to MATCHED -> CLEARED -> SETTLED. Total trace length is 31
(trace_length).

## 2. Why It Is Flagged
- The edge ENRICHED -> UNMATCHED_ERROR appears 8 times
  (sequence_level.max_edge_repeat, sequence_level.most_repeated_edge),
  while normal traces almost never repeat any edge
  (sequence_level.fired contains SQ-edge-repeat).
- Each individual transition within the cycle is plausible on its own
  (transition_findings is empty — no per-step flags fired), but the
  aggregate pattern is anomalous (verdict.fired_signals = ["SQ-edge-repeat"]).
- The normalized log-likelihood is only -0.36
  (sequence_level.norm_ll_z_score), which is why per-step surprise
  signals do not catch this — the cycle is composed of high-probability
  edges repeated abnormally often.

## 3. What Normal Would Look Like
A normal repair path enters UNMATCHED_ERROR at most once and proceeds
through AMENDED -> ENRICHED -> MATCHED in a single iteration. Repeated
re-entry of the same error edge is not characteristic of normal
lifecycle behaviour.

## 4. Recommended Validation Step
Investigate the 8 amendment cycles for an unresolved data-quality issue
(e.g., mismatched SSI or counterparty static data) that prevents
matching. The pattern suggests the system is retrying the same
enrichment without addressing the underlying mismatch.
```

This is the **most important** example because it demonstrates a case
where **NO per-transition flag fires**. The whole anomaly is captured by
`SQ-edge-repeat`. This is the empirical reason the detector must
implement sequence-level signals and cannot rely on per-transition
flags alone.

---

## 5. `pack_unlikely_branch.json` — early REJECTED

```markdown
## 1. Observed Trace
The trade was REJECTED immediately after NEW (trace). The trace contains
only two states (trace_length).

## 2. Why It Is Flagged
- The trace's normalized log-likelihood is 5.13 standard deviations
  below the normal baseline (sequence_level.norm_ll_z_score), the
  strongest sequence-level signal of the four anomaly classes.
- No per-transition flag fired (transition_findings is empty), because
  NEW -> REJECTED is a legal transition with probability ≈ 0.03 — rare
  but not impossible. The whole-trace signal is what catches it.
- verdict.fired_signals = ["SQ-low-likelihood"].

## 3. What Normal Would Look Like
From NEW, the canonical continuation is EXECUTED with probability 0.97.
A normal trace ends at SETTLED, CANCELLED, or (rarely) REJECTED late in
the lifecycle, not at step 1.

## 4. Recommended Validation Step
Determine whether the REJECTED outcome reflects a genuine business
reason (e.g., invalid order detected upstream) or a data-quality issue
(e.g., the rest of the trace was truncated). The pattern is consistent
with either; only the upstream system can confirm.
```
