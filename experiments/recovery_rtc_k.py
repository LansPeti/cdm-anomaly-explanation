"""RTC@k: Round-Trip Correctness with k repetitions.

Per Allamanis et al. (arXiv 2402.08699), running the round-trip k
times and reporting agreement statistics gives a robustness signal
that single-shot rho cannot. We adopt the formal RTC@k metric:

    RTC@k = fraction of round-trips where >= ceil(k/2) of k recoveries
            score above a fixed faithfulness threshold

This script runs the full pipeline k=3 times per spec on the same n=10
specs per anomaly class (40 specs total → 120 round-trips). Reports:

  - per-spec recovery score distribution (k=3 individual scores)
  - per-spec modal recovery (most-frequent class hint across k)
  - aggregate RTC@k at threshold rho >= 0.7
  - variance across repetitions (stability indicator)

Cost: roughly 3x the standard recovery run (~$1).

Usage
-----
    python -m experiments.recovery_rtc_k --k 3 --n-per-class 10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

from src.business_context import StaticMappingProvider
from src.detector import EmpiricalMarkovDetector
from src.evidence_pack import build_evidence_pack
from src.llm_explainer import AnomalyExplainer
from src.specification_recovery import (
    AnomalyRecoverer,
    AnomalySpecifier,
    SemanticComparator,
    TraceMaterializer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "results" / "recovery_rtc_k"

DEFAULT_CLASSES = ["skipped_step", "impossible", "pingpong", "unlikely_branch"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3,
                        help="Number of repetitions per spec.")
    parser.add_argument("--n-per-class", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Faithfulness threshold for RTC@k.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    classes = list(DEFAULT_CLASSES)

    print("[1/4] Loading dataset and fitting detector...")
    if not (DATA_DIR / "normal.jsonl").exists():
        print("ERROR: missing data/normal.jsonl", file=sys.stderr)
        return 2
    normal_traces = [
        json.loads(line)["trace"]
        for line in (DATA_DIR / "normal.jsonl").open()
    ]
    detector = EmpiricalMarkovDetector()
    detector.fit(normal_traces)

    print("[2/4] Building roles (gpt-4o-mini all 4 roles)...")
    specifier = AnomalySpecifier(temperature=0.7)
    materializer = TraceMaterializer()
    explainer = AnomalyExplainer()
    recoverer = AnomalyRecoverer(temperature=0.2)
    comparator = SemanticComparator(temperature=0.0)

    def pipeline(trace):
        result = detector.detect(trace)
        pack = build_evidence_pack(
            "TRD-RTC", trace, result,
            business_context_provider=StaticMappingProvider(),
        )
        explanation = explainer.explain(pack)
        return explanation.text, list(result.fired_signals)

    # The RecoveryFaithfulnessRunner.run_one synthesises a fresh spec
    # each call, but we hold the spec fixed across k repetitions, so
    # we drive the loop manually below.

    print(f"[3/4] Running {args.k}× repetitions per spec on "
          f"{args.n_per_class} specs × {len(classes)} classes "
          f"= {args.k * args.n_per_class * len(classes)} total round-trips...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # We run k repetitions for each spec, holding the spec fixed across reps.
    # The RecoveryFaithfulnessRunner.run_one synthesises a fresh spec each
    # time, so we override that here: generate the spec once, then repeat
    # only (materialize → pipeline → recover → compare).
    import random
    rng = random.Random(args.seed)
    per_spec_records: list[dict] = []
    for cls in classes:
        for _ in range(args.n_per_class):
            spec = specifier.specify(cls, rng=rng)
            trace = materializer.materialize(spec, rng=rng)
            # k repetitions of the explanation+recovery+score
            rep_scores: list[dict] = []
            rep_classes: list[str] = []
            for _rep in range(args.k):
                explanation_text, fired = pipeline(trace)
                recovered = recoverer.recover(explanation_text)
                scores = comparator.rho(spec, recovered)
                rep_scores.append(scores)
                rep_classes.append(str(recovered.class_hint))
            modal_class, modal_count = Counter(rep_classes).most_common(1)[0]
            rho_values = [s["rho"] for s in rep_scores]
            per_spec_records.append({
                "class_hint": cls,
                "spec_nl": spec.nl_description,
                "k": args.k,
                "rep_rho": rho_values,
                "rep_classes": rep_classes,
                "modal_class": modal_class,
                "modal_agreement": modal_count / args.k,
                "rho_mean": round(mean(rho_values), 4),
                "rho_stdev": round(pstdev(rho_values), 4),
                "rho_min": round(min(rho_values), 4),
                "rho_max": round(max(rho_values), 4),
                "passes_threshold": sum(1 for v in rho_values if v >= args.threshold),
            })
            print(f"  {cls:<18s} ρ={rho_values} modal={modal_class} "
                  f"({modal_count}/{args.k})")

    print(f"[4/4] Aggregating RTC@{args.k} statistics...")
    n_total = len(per_spec_records)
    rtc_at_k = sum(
        1 for r in per_spec_records
        if r["passes_threshold"] >= (args.k + 1) // 2
    ) / n_total
    mean_modal_agreement = mean(r["modal_agreement"] for r in per_spec_records)
    mean_within_spec_stdev = mean(r["rho_stdev"] for r in per_spec_records)

    summary = {
        "n_specs": n_total,
        "k": args.k,
        "threshold": args.threshold,
        f"RTC_at_{args.k}": round(rtc_at_k, 4),
        "mean_modal_class_agreement": round(mean_modal_agreement, 4),
        "mean_within_spec_rho_stdev": round(mean_within_spec_stdev, 4),
        "mean_aggregate_rho": round(
            mean(r["rho_mean"] for r in per_spec_records), 4
        ),
    }

    (OUT_DIR / "per_spec.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_spec_records) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")

    print()
    print(f"  RTC@{args.k} (>= ceil(k/2) reps clear ρ ≥ {args.threshold}): "
          f"{summary[f'RTC_at_{args.k}']:.3f}")
    print(f"  Mean modal class agreement: {summary['mean_modal_class_agreement']:.3f}")
    print(f"  Mean within-spec ρ stdev:    {summary['mean_within_spec_rho_stdev']:.3f}")
    print(f"  Mean aggregate ρ:            {summary['mean_aggregate_rho']:.3f}")
    print()
    print(f"Wrote {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
