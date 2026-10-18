# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
make test
```

Only needed for LLM work: copy `.env.example` to `.env` and set
`OPENAI_API_KEY`. `.env` is git-ignored — never commit keys.

## Branching workflow

`main` is protected by convention: **never push directly to it.**

```bash
git checkout main && git pull
git checkout -b <yourname>/<topic>      # e.g. anna/xgboost-detector
# ... work, commit small and often ...
git push -u origin <yourname>/<topic>
```

Then open a pull request against `main`:

- Describe what changed and why (for model work: which model, which
  thresholds differ from `DEFAULT_THRESHOLDS`, and the resulting numbers).
- CI must be green (lint + tests on Python 3.10–3.12).
- Get at least one review before merging.

Trying several alternatives (e.g. model families)? Make one branch per
alternative off `main` instead of stacking them, so they can be compared
side by side.

## Before you push

```bash
make lint      # ruff check .   (make format to autofix)
make test      # python -m pytest
```

If a test breaks because you changed a contract (e.g. `src/flags.py` or
the evidence-pack schema), discuss it in the PR — don't disable the test.

## Where things go

- **Library code** → `src/`. Run modules from the repo root as
  `python -m src.<module>`.
- **A new experiment** → a script in `experiments/` that writes to
  `results/<experiment_name>/`, plus a `make` target.
- **Generated data** → `data/` (git-ignored). Don't commit datasets or
  bulky per-trace outputs; commit summaries only.
- **A new detector** → subclass `AnomalyDetector` in `src/detector.py`;
  `detect()` must return a `DetectionResult` (`src/flags.py`).

## Commit messages

Short imperative subject line (`Add Laplace smoothing to detector`), with a
body explaining *why* when it isn't obvious. `"updates"` is not a commit
message.
