# Makefile — single entry point for tests and for reproducing every
# published result. Targets are idempotent and self-contained.
# Deterministic targets write into results/; LLM targets need OPENAI_API_KEY.

PYTHON   ?= python3
SEED     ?= 42
NORMAL   ?= 5000
PER_ANOM ?= 250

.PHONY: help install test lint format demo data reports ci ablation \
        e2e-smoke e2e-scale determinism recovery recovery-noise all clean

help:
	@echo "Setup / dev:"
	@echo "  install        - pip install dev requirements"
	@echo "  test           - run the unit tests"
	@echo "  lint           - ruff check"
	@echo "  format         - ruff autofix + format imports"
	@echo "  demo           - run the pipeline on one trace (no LLM)"
	@echo "Deterministic results (no API key):"
	@echo "  data           - generate synthetic dataset into data/"
	@echo "  reports        - detector reports      -> results/detector/"
	@echo "  ci             - recall confidence CIs -> results/ci/"
	@echo "  ablation       - flag ablation study   -> results/ablation/"
	@echo "  all            - test + data + reports + ci + ablation"
	@echo "LLM results [need OPENAI_API_KEY]:"
	@echo "  e2e-smoke      - 4-trace end-to-end run (1 per anomaly class)"
	@echo "  e2e-scale      - 100-trace end-to-end run (25 per class)"
	@echo "  determinism    - same pack 5x, measure drift"
	@echo "  recovery       - specification-recovery faithfulness"
	@echo "  recovery-noise - noise-floor control for recovery"
	@echo "  clean          - remove generated data"

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check . --fix

demo:
	$(PYTHON) -m src.demo --no-llm

data: data/normal.jsonl

data/normal.jsonl:
	$(PYTHON) -m src.data_generator --seed $(SEED) --n-normal $(NORMAL) --n-per-anomaly $(PER_ANOM) --out data/

reports: data
	$(PYTHON) -m scripts.run_reports

ci: reports
	$(PYTHON) -m experiments.confidence_intervals

ablation: data
	$(PYTHON) -m experiments.ablation

e2e-smoke: data
	$(PYTHON) -m experiments.run_e2e --per-class 1

e2e-scale: data
	$(PYTHON) -m experiments.run_e2e --per-class 25

determinism: data
	$(PYTHON) -m experiments.run_determinism --reps 5

recovery: data
	$(PYTHON) -m experiments.recovery_faithfulness --n-per-class 2

recovery-noise:
	$(PYTHON) -m experiments.recovery_noise_floor

all: test data reports ci ablation
	@echo "All deterministic results regenerated. For LLM results run:"
	@echo "  make e2e-scale determinism"

# Only removes regenerable inputs; committed results/ are left alone
# (they are overwritten by re-running the corresponding target).
clean:
	rm -rf data/ .pytest_tmp_runs/
