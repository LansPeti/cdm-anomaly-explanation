# Cross-Validation of LLM-as-Judge for Specification-Recovery Faithfulness

Independent academic re-scoring of all 40 entries in
`experiments/output_recovery/per_run.jsonl` against the existing
`scores.judge_score`. Scoring was done blind: the judge score was inspected
only after all 40 of my own scores were finalized.

## 1. Distribution of my scores (0-100)

| metric | mine | existing judge x100 |
|--------|------|---------------------|
| min    |   5  |   0                 |
| max    |  95  | 100                 |
| median |  92  |  90                 |
| mean   | 84.95|  84.00              |
| stdev  | 20.18|  22.67              |

The two distributions are remarkably close in central tendency and spread.

## 2. Per-class means

| class            | n  | my mean | judge mean |
|------------------|----|---------|------------|
| impossible       | 10 |  76.00  |  73.00     |
| pingpong         | 10 |  88.00  |  89.00     |
| skipped_step     | 10 |  86.60  |  84.00     |
| unlikely_branch  | 10 |  89.20  |  90.00     |

`impossible` is the lowest-scoring class on both sides (entries 14 and 17 —
where the recoverer reported "no anomaly" — drag the mean down for both
graders).

## 3. Correlation with existing judge

- Pearson  r = **0.9299**
- Spearman r = **0.3633**

Pearson is very high because both graders correctly identify the two
catastrophic recoveries (entries 14 and 17) and rate everything else high.
Spearman is much lower because most of the remaining 38 scores are bunched
into a narrow high-score band where small differences in *exact* numeric
ranking dominate the rank correlation despite producing essentially the same
qualitative judgement. Pearson is the more meaningful metric for an
interval-scale rating instrument here.

## 4. Top disagreements

| idx | class            | mine | judge | abs diff | reason judge appears to have missed |
|-----|------------------|------|-------|----------|-------------------------------------|
| 1   | skipped_step     |  50  |  20   |   30     | Spec says EXECUTED->CAPTURED but recovered says NEW->CAPTURED — the *wrong* transition was reconstructed; still both describe a CAPTURED-state skip anomaly so 20 is too harsh, but 50 reflects the substantive state-naming error. |
| 36  | unlikely_branch  |  92  |  70   |   22     | Recovered correctly captures NEW->REJECTED as a rare/unlikely path with the right class hint; judge appears to have penalised stylistic differences ("rare occurrence" vs spec's wording) rather than semantic content. |
| 34  | unlikely_branch  |  75  |  90   |   15     | Recovered re-classifies as `skipped_step` even though the spec is `unlikely_branch`; the underlying NEW->REJECTED is correct, but the class re-framing should have been penalised more than the judge did. |
| 38  | unlikely_branch  |  75  |  90   |   15     | Same pattern as 34 — recovered switches class to `skipped_step`; judge ignored the class drift. |
| 8   | skipped_step     |  88  |  100  |   12     | Judge gave a perfect 100 even though the recovered description re-labels the anomaly class as `impossible` rather than `skipped_step`; some penalty for class drift seems warranted. |
| 20  | pingpong         |  80  |  90   |   10     | Recovered names ENRICHED-UNMATCHED_ERROR loop (and a count of 7) but does not explicitly include the AMENDED state from the spec's three-state cycle. |

## 5. Aggregate verdict

**Well-calibrated.**

The existing LLM-as-judge tracks an independent academic rater closely:
overall mean differs by less than one point (84.0 vs 84.95), per-class means
differ by 1-3 points, and Pearson correlation is 0.93. Both raters agree on
the two catastrophic recoveries (entries 14 and 17, where the recoverer
returned "no anomaly") and on the broad mass of high-fidelity recoveries.

The judge does exhibit two minor systematic tendencies worth flagging:

- It is slightly *lenient on class-hint drift* — entries 8, 34, 38 received
  90-100 even though the recoverer placed the anomaly in a different class
  than the spec.
- It is slightly *harsh on surface-form differences* — entry 1 was scored
  20, but a closer reading shows the recovered description, while
  reconstructing the wrong source state, still identifies the same
  CAPTURED-skipping pattern; entry 36 is similarly penalised for stylistic
  wording rather than semantic loss.

These biases roughly cancel in aggregate (mean offset ≈ +1 point in the
judge's direction). For the purposes of measuring round-trip
specification-recovery faithfulness on this dataset, the judge is a
reliable measurement instrument; the verdict is **well-calibrated** with
the caveat that it should be supplemented by an explicit class-hint match
metric when class fidelity is itself a target of measurement.
