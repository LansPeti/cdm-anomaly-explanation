"""Noise-floor measurement for the Specification-Recovery framework.

Question: What ρ does the framework produce when the explanation is
DELIBERATELY MISMATCHED — i.e. an explanation produced for a different
trace, fed to the recoverer for a specification it did not produce?

If ρ_noise ≈ ρ_real, the framework is measuring nothing.
If ρ_noise ≪ ρ_real, the framework has signal.

Procedure:
    1. Read results/recovery/per_run.jsonl (the real runs).
    2. For each (spec_i, explanation_i) pair, swap explanation_i with
       explanation_j where j ≠ i, drawn from a different anomaly class
       to maximise mismatch.
    3. Run the recoverer on the swapped explanation.
    4. Compare the recovered spec to spec_i (the original — not the
       one that produced explanation_j).
    5. Report aggregate ρ_noise.

Usage
-----
    python -m experiments.recovery_noise_floor
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from statistics import mean

from src.specification_recovery import (
    AnomalyRecoverer,
    AnomalySpecification,
    SemanticComparator,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "results" / "recovery" / "per_run.jsonl"
OUT_DIR = REPO_ROOT / "results" / "recovery_noise_floor"


def main(argv: list[str] | None = None) -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found.", file=sys.stderr)
        print("       Run experiments/recovery_faithfulness.py first.",
              file=sys.stderr)
        return 2

    print(f"[1/4] Loading runs from {INPUT_PATH.name}...")
    runs: list[dict] = []
    with INPUT_PATH.open() as f:
        for line in f:
            runs.append(json.loads(line))
    if len(runs) < 2:
        print(f"ERROR: need ≥ 2 runs to swap, got {len(runs)}.", file=sys.stderr)
        return 2
    print(f"      Loaded {len(runs)} runs across "
          f"{len(set(r['spec']['class_hint'] for r in runs))} classes.")

    print("[2/4] Pairing each spec with a foreign explanation "
          "(different anomaly class)...")
    rng = random.Random(42)
    pairs: list[tuple[dict, dict]] = []  # (spec_run, foreign_explanation_run)
    for i, run in enumerate(runs):
        own_class = run["spec"]["class_hint"]
        candidates = [
            (j, r) for j, r in enumerate(runs)
            if j != i and r["spec"]["class_hint"] != own_class
        ]
        if not candidates:
            # Fallback: any other run.
            candidates = [(j, r) for j, r in enumerate(runs) if j != i]
        _, foreign = rng.choice(candidates)
        pairs.append((run, foreign))

    print(f"[3/4] Running recoverer + comparator on {len(pairs)} mismatched pairs...")
    recoverer = AnomalyRecoverer()
    comparator = SemanticComparator()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i, (own, foreign) in enumerate(pairs, start=1):
        own_spec = AnomalySpecification(
            nl_description=own["spec"]["nl_description"],
            class_hint=own["spec"]["class_hint"],
            metadata=own["spec"].get("metadata", {}),
        )
        foreign_explanation = foreign["explanation"]
        recovered = recoverer.recover(foreign_explanation)
        scores = comparator.rho(own_spec, recovered)
        results.append({
            "own_class": own["spec"]["class_hint"],
            "foreign_class": foreign["spec"]["class_hint"],
            "own_spec_nl": own["spec"]["nl_description"],
            "foreign_explanation": foreign_explanation[:200] + "...",
            "recovered_class": recovered.class_hint,
            "recovered_nl": recovered.nl_description,
            "scores": scores,
        })
        print(f"  [{i}/{len(pairs)}] own={own['spec']['class_hint']:18s} "
              f"foreign={foreign['spec']['class_hint']:18s} "
              f"recovered={recovered.class_hint!s:18s} "
              f"rho={scores['rho']:.3f}")

    print("[4/4] Aggregating...")
    summary = {
        "n": len(results),
        "noise_floor_rho": round(mean(r["scores"]["rho"] for r in results), 4),
        "noise_floor_judge": round(mean(r["scores"]["judge_score"] for r in results), 4),
        "noise_floor_class_match": round(mean(r["scores"]["class_match"] for r in results), 4),
    }

    (OUT_DIR / "per_run.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")

    print()
    print(f"  Noise-floor ρ̄ = {summary['noise_floor_rho']}  "
          f"(n={summary['n']})")
    print(f"  Noise-floor judge = {summary['noise_floor_judge']}, "
          f"class match = {summary['noise_floor_class_match']}")
    print()
    print(f"Wrote {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
