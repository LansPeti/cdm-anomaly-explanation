# ISDA CDM Trade Lifecycle Anomaly Explanation

A glass-box system that **detects** anomalies in ISDA CDM post-trade
lifecycle traces and **explains** them in plain language — without letting
the LLM make the call.

```
data_generator ──► detector ──► evidence_pack ──► llm_explainer
 synthetic traces   Markov chain   JSON with every     narrates the pack,
 (normal + 4        + 5 flags      signal as a         cites fields,
  anomaly classes)                 citable field       never judges
```

- **Detector** (`src/detector.py`) — a first-order empirical Markov chain
  over the CDM state space. Fires per-transition flags (`PT-impossible`,
  `PT-surprise`, `PT-rank`) and sequence flags (`SQ-low-likelihood`,
  `SQ-edge-repeat`).
- **Evidence pack** (`src/evidence_pack.py`) — deterministic JSON
  serialisation of the detection result; the *only* input the LLM sees.
- **LLM narrator** (`src/llm_explainer.py`) — turns the pack into a
  four-section markdown explanation with mandatory field citations, then a
  post-hoc validator checks it.

The data contract between the layers is `src/flags.py` (detector output)
and `src/evidence_pack.py` (LLM input) — start reading there. Each module
in `src/` has a docstring explaining its role.

---

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/LansPeti/cdm-anomaly-explanation.git
cd cdm-anomaly-explanation

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

make test                        # unit tests, no API key needed
make demo                        # one trace through the whole pipeline (no LLM)
```

No `make` (e.g. on Windows)? Every target is a one-liner — see the
[`Makefile`](Makefile). The equivalents of the above are
`python -m pytest` and `python -m src.demo --no-llm`.

To run the LLM layer, copy `.env.example` to `.env` and set
`OPENAI_API_KEY`. Then for example:

```bash
python -m src.demo --anomaly pingpong      # detect + explain one anomalous trace
```

## Repository layout

| Path | What's there |
| --- | --- |
| `src/` | The pipeline itself — lifecycle graph, data generator, detector, evidence pack, LLM explainer, business-context providers, specification-recovery framework. |
| `tests/` | Unit tests (no network / API key needed). |
| `scripts/` | `run_reports.py` — runs the detector on the generated dataset. |
| `eval/` | `eval_runner.py` — field-citation rate + hallucination checks for LLM output. |
| `experiments/` | Research experiment scripts (ablation, confidence intervals, E2E, recovery faithfulness, …). `rag_corpus/` is the fixed document corpus used by the RAG experiments. |
| `results/` | Committed outputs of `scripts/` and `experiments/`. One folder per experiment. |
| `examples/` | Five reference evidence packs + gold-standard explanations. |

Generated data goes to `data/` (git-ignored); regenerate it with `make data`.

## Common tasks

| Command | What it does | API key? |
| --- | --- | --- |
| `make test` | Unit tests | no |
| `make lint` | `ruff check` | no |
| `make demo` | One trace through the pipeline | no |
| `make data` | Generate synthetic dataset into `data/` | no |
| `make reports` / `ci` / `ablation` | Detector reports, recall CIs, flag ablation → `results/` | no |
| `make all` | All deterministic results above | no |
| `make e2e-smoke` / `e2e-scale` | End-to-end LLM explanations | **yes** |
| `make recovery` | Specification-Recovery Faithfulness | **yes** |

`make help` lists everything.

## Thesis ↔ repository

This repository is the implementation behind the BSc thesis *Adapting Large
Language Models to Financial Domain Knowledge — Faithful, Auditable Narration
of Trade-Lifecycle Anomalies* (Lanszki Péter, PPKE ITK, 2026; supervisor
Dr. Fogarasi Norbert). The detector layer (`src/detector.py`) was implemented
by Hanna Pinke.

Every number in the thesis's evaluation chapter comes from a committed file
in `results/`. LLM runs need `OPENAI_API_KEY` and `make data` first; the
thesis runs used 10 specifications per class (40 total).

| Thesis | Result | Folder in `results/` | Reproduce |
| --- | --- | --- | --- |
| §6.1 | Recall 100 %, 95 % CI, FP 4.62 % | `detector/`, `ci/` | `make reports ci` |
| §6.2 | Signal ablation (7 configs) | `ablation/` | `make ablation` |
| §6.3 | Hallucination taxonomy, 78/100 clean | `e2e/`, `taxonomy/` | `make e2e-scale`, then `python -m experiments.hallucination_taxonomy_apply` |
| §6.4, Table 6.1 | Canonical static config, ρ = 0.75 | `recovery_static/` | `python -m experiments.recovery_faithfulness --mode static --n-per-class 10` |
| Table 6.2 | No-context baseline, ρ = 0.79 / 0.81 | `recovery/`, `recovery_vanilla/` | `... --mode vanilla --n-per-class 10` (`recovery/` was produced by an earlier harness version) |
| §6.4.1 | Noise floor ρ = 0.058 | `recovery_noise_floor/` | `make recovery-noise` |
| §6.4.2 | RTC@3 = 0.70, modal agreement 0.967 | `recovery_rtc_k/` | `python -m experiments.recovery_rtc_k` |
| §6.4.3 | Independent-judge Pearson 0.93 | `recovery_cross_validation/` | Manual blind re-scoring (no script) |
| §6.5 | Static vs. RAG: 0.747 → 0.813 | `recovery_static/`, `recovery_rag/` | `... --mode rag --n-per-class 10` |
| §6.5 | RAG without gold narrations, ρ = 0.810 | **not in repo** | `... --mode rag --rag-exclude-gold --n-per-class 10 --out-dir results/recovery_rag_noleak` |
| §6.6 | NEURON plausibility 0.83, sensemaking 0.86 | `recovery_neuron/` | `python -m experiments.recovery_neuron_trio` |
| §6.7, Table 6.3 | Determinism, 5 reps per class | `determinism/` | `make determinism` |
| §6.8 | Cross-model sweep (RAG): gpt-4o 0.77, gpt-5.4-nano 0.82, gpt-5.4-mini 0.81, gpt-5.4 0.96, gpt-5.5 0.08 | `recovery_rag_gpt4o_b/`, `recovery_rag_gpt54nano/`, `recovery_rag_gpt54mini/`, `recovery_rag_gpt54/`, `recovery_rag_gpt55/` | `... --mode rag` with the `--*-model` flags set (which roles were swapped is not recorded in these older files) |
| §6.8 | gpt-5.5 with raised budget, ρ = 0.75 | **not in repo** | `... --mode rag --explainer-model gpt-5.5 --explainer-max-tokens 8000 --out-dir results/recovery_rag_gpt55_fixed` |

`...` = `python -m experiments.recovery_faithfulness`. The RAG provider
retrieves from `experiments/rag_corpus/` plus `examples/gold_explanations.md`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — branching workflow, tests, and
code conventions.
