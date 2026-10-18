# 01. System Architecture

## One-paragraph summary

A trade lifecycle anomaly **explanation** system that decomposes into
three strictly-separated layers: a **detector** computes probabilistic
signals from state transitions; an **evidence pack** serialises those
signals into a schema-stable JSON document with declarative
`validation_requirements` and `business_context` fields; and an
**LLM narrator** translates the JSON into a four-section markdown
explanation with mandatory field-level citations. The detector is
the only "smart" component on the detection side; the LLM is a
disciplined translator, not a judge.

A single architectural seam — `BusinessContextProvider` — is where
deployment-specific knowledge plugs in (declarative mapping or RAG
over a corpus). The rest of the pipeline is firm-agnostic.

## The three layers

```
┌─────────────────────┐    list[list[str]]    ┌──────────────────────┐
│  data_generator     │ ─────────────────────▶│  EmpiricalMarkov-    │
│  (synthetic traces) │                       │  Detector            │
└─────────────────────┘                       │                      │
                                              │  - Laplace-smoothed  │
                                              │    first-order       │
                                              │  - 5 flags           │
                                              │  - JSON persistence  │
                                              └──────────┬───────────┘
                                                         │ DetectionResult
                                                         │ (transition_flags,
                                                         │  sequence_flag,
                                                         │  is_anomalous,
                                                         │  fired_signals)
                                                         ▼
┌─────────────────────┐                       ┌──────────────────────┐
│  AnomalyExplainer   │ ◀──── pack JSON ──────│  build_evidence_pack │
│  (gpt-4o-mini-2024- │                       │                      │
│   07-18, T=0.2)     │                       │  + validation_       │
│                     │                       │    requirements      │
│  - 7 hard rules     │                       │  + business_context  │
│  - 2 few-shots      │                       │    (Static / RAG)    │
│  - SHA-256 cache    │                       │                      │
└─────────────────────┘                       └──────────────────────┘
```

## Module map

| Module | Status | Purpose |
|---|---|---|
| `src/lifecycle_graph.py` | **Implemented** | State space + legal transitions (12 states, canonical CDM). |
| `src/data_generator.py` | **Implemented** | Synthetic traces: 5000 normal + 4×250 anomalous (defaults). |
| `src/flags.py` | **Implemented** | `TransitionFlag` / `SequenceFlag` / `DetectionResult` with `__post_init__` invariants. |
| `src/detector.py` | **Implemented** | `EmpiricalMarkovDetector` — Laplace-smoothed first-order Markov, 5 flags. |
| `src/evidence_pack.py` | **Implemented** | `DetectionResult` → JSON pack with `validation_requirements` + `business_context`. |
| `src/business_context.py` | **Implemented** | `BusinessContextProvider` ABC + `StaticMappingProvider` + `RagProvider`. |
| `src/llm_explainer.py` | **Implemented** | `AnomalyExplainer` with 7 hard rules, 2 few-shots, SHA-256 cache. |
| `src/specification_recovery.py` | **Implemented** | Specifier / Materializer / Recoverer / Comparator for the recovery framework. |
| `src/hallucination_taxonomy.py` | **Implemented** | Six-category Berti 2026 diagnoser (deterministic regex). |
| `eval/eval_runner.py` | **Implemented** | Field-citation rate + hallucination taxonomy; defaults to `markov` detector. |
| `scripts/run_reports.py` | **Implemented** | Runs the detector and writes `results/detector/anomaly_detection_report.json`. |
| `experiments/run_e2e.py` | **Implemented** | End-to-end pipeline with `check_hallucinations()`. |
| `experiments/recovery_*.py` | **Implemented** | Recovery framework, NEURON-trio, RTC@k, noise-floor, per-mode ablation. |
| `experiments/ablation.py` | **Implemented** | Flag-set ablation (redundancy test). |
| `experiments/confidence_intervals.py` | **Implemented** | Clopper-Pearson 95% CI on detector recall. |
| `experiments/hallucination_taxonomy_apply.py` | **Implemented** | Berti taxonomy applied to existing E2E + recovery outputs. |
| `tests/test_*.py` | **Implemented** | 100 unit tests (test_pipeline + test_flags_invariants + test_evidence_pack + test_lifecycle_graph + test_specification_recovery + test_business_context). |
| `results/` | **Committed** | Outputs of `scripts/` and `experiments/` (summaries, reports) — the numbers cited in the docs. |
| `chatbot/` | **Implemented** | Cloudflare Worker + RAG widget over the project's own docs. |
| `docs/` | **Implemented** | MkDocs Material site, also the corpus indexed by the chatbot. |

## Why this separation

### Detector ↔ pack: the contract is the only handshake

The pack builder consumes a `DetectionResult` dataclass (validated by
`__post_init__` invariants) and never inspects detector internals.
The `EmpiricalMarkovDetector` can be swapped for an MLR / decision
tree / XGBoost without touching anything downstream — see
[03 Detector](03_detector.md) for the trade-offs and current
implementation.

### Pack ↔ LLM: the LLM never inspects the model

The LLM receives a JSON document with named fields and is forbidden by
seven hard rules from:

1. emitting any factual claim without an inline parenthetical citation;
2. inventing state names not present in `trace`;
3. asserting causation;
4. for non-anomalous packs, writing anything other than the literal
   `Normal trace, no findings.`;
5. departing from the four-section markdown layout;
6. (rule 6) generating Section 4 content from its world model — Section
   4 must enumerate `validation_requirements` verbatim;
7. (rule 7) inventing business / regulatory content beyond
   `business_context` field values.

These constraints are enforced by:

1. the system prompt (rules 1–7);
2. the post-hoc `validate_explanation()` function;
3. the `check_hallucinations()` content check (which the recovery
   framework's first iteration empirically motivated);
4. the eval harness in `eval/eval_runner.py`.

This separation is **research-backed**:

- Berti et al. 2024 (arXiv:2407.02310) showed zero-shot LLMs fail at
  trace anomaly **judgment**. Confining the LLM to **narration** is
  the documented mitigation.
- Berti et al. 2026 (Springer ICPM Workshops) catalogue the
  hallucination modes specific to PM tasks. We adopt the taxonomy
  verbatim in `src/hallucination_taxonomy.py`.

### Pack ↔ business knowledge: the seam

`business_context` is filled by a pluggable `BusinessContextProvider`
(see [04 Evidence Pack](04_evidence_pack_spec.md) and the paper's
§7). A firm deploying this architecture replaces one class
(`StaticMappingProvider` → its own RAG over internal runbooks) and
the rest of the pipeline is untouched.

## Quick-start verification

```bash
# 1. Generate data
make data

# 2. Run the unit tests (no API needed)
make test

# 3. Run the detector and write reports
make reports

# 4. Confidence intervals + flag ablation (no API)
make ci
make ablation

# 5. With OPENAI_API_KEY set, end-to-end LLM run on a small sample
make e2e-smoke

# 6. Recovery framework (the methodological contribution)
make recovery
```

See [Quick Start](quickstart.md) for the full walkthrough and
[Reproducibility](09_reproducibility.md) for the make-target
↔ paper-claim cross-reference.
