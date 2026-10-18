"""End-to-end runner for the Specification-Recovery Faithfulness
framework described in experiments/rag_corpus/12_specification_recovery_faithfulness.md.

Wires together:

    AnomalySpecifier  →  TraceMaterializer  →  EmpiricalMarkovDetector
        →  build_evidence_pack  →  AnomalyExplainer  →  AnomalyRecoverer
        →  SemanticComparator  →  ρ score per run

Outputs:
  results/recovery/per_run.jsonl   — one JSON per evaluation
  results/recovery/summary.json    — aggregate metrics

Usage
-----
    python -m experiments.recovery_faithfulness                    # n=2 per class (~$0.20)
    python -m experiments.recovery_faithfulness --n-per-class 5    # ~$0.50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

from src.business_context import (
    BusinessContextProvider,
    RagProvider,
    StaticMappingProvider,
)
from src.detector import EmpiricalMarkovDetector
from src.evidence_pack import build_evidence_pack
from src.llm_explainer import AnomalyExplainer
from src.specification_recovery import (
    AnomalyRecoverer,
    AnomalySpecifier,
    RecoveryFaithfulnessRunner,
    SemanticComparator,
    TraceMaterializer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "recovery"

DEFAULT_CLASSES = ["skipped_step", "impossible", "pingpong", "unlikely_branch"]


def build_pipeline_callable(detector: EmpiricalMarkovDetector,
                            explainer: AnomalyExplainer,
                            business_context_provider: BusinessContextProvider | None = None):
    """Create the (trace -> (explanation_text, fired_signals)) callable
    that the RecoveryFaithfulnessRunner expects.
    """
    def pipeline(trace: list[str]) -> tuple[str, list[str]]:
        result = detector.detect(trace)
        pack = build_evidence_pack(
            "TRD-RECOVERY", trace, result,
            business_context_provider=business_context_provider,
        )
        explanation = explainer.explain(pack)
        return explanation.text, list(result.fired_signals)
    return pipeline


def build_provider(mode: str,
                   exclude_gold: bool = False) -> BusinessContextProvider | None:
    """Resolve a business-context provider from the --mode flag.

    Modes
    -----
    - "vanilla": no business_context field at all (legacy path —
                 build_evidence_pack always adds the field, so we
                 substitute a NullProvider that returns an empty struct
                 the LLM is told to ignore)
    - "static":  StaticMappingProvider (default)
    - "rag":     RagProvider over the in-repo doc corpus
    """
    if mode == "vanilla":
        # We pass a provider that returns the static skeleton with a
        # marker phase so the LLM can be told to ignore the field.
        # Strict apples-to-apples comparison: the same code path runs;
        # only the content differs.
        return _NullBusinessContextProvider()
    if mode == "static":
        return StaticMappingProvider()
    if mode == "rag":
        # Fixed corpus of project write-ups with trade-lifecycle vocabulary
        # (kept in experiments/rag_corpus/ so published results reproduce).
        # Public sources only — no firm-internal data.
        corpus_dir = REPO_ROOT / "experiments" / "rag_corpus"
        corpus_paths = sorted([
            corpus_dir / "01_architecture.md",
            corpus_dir / "10_known_limitations.md",
            corpus_dir / "12_specification_recovery_faithfulness.md",
            REPO_ROOT / "examples" / "gold_explanations.md",
        ])
        if exclude_gold:
            # Leakage control: drop the per-class gold narrations.
            corpus_paths = [p for p in corpus_paths
                            if p.name != "gold_explanations.md"]
        corpus_paths = [p for p in corpus_paths if p.exists()]
        return RagProvider(corpus_paths=corpus_paths, top_k=3)
    raise ValueError(f"Unknown mode: {mode!r}. Use vanilla, static, or rag.")


class _NullBusinessContextProvider(BusinessContextProvider):
    """Returns a context that signals 'no business knowledge'.

    The phase is "unknown" and all other fields are empty. The LLM
    explainer's hard rule 7 says to weave business_context fields when
    present; an empty context yields no business citations, matching
    the pre-extension behaviour for an apples-to-apples ρ comparison.
    """

    def context_for(self, trace, fired_signals):
        from src.business_context import BusinessContext
        return BusinessContext(
            current_lifecycle_phase="unknown",
            phase_purpose="",
            typical_operational_owner="",
            regulatory_relevance=[],
            common_failure_causes=[],
            downstream_dependencies_blocked=[],
            retrieved_excerpts=None,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=2,
                        help="Number of round-trip evaluations per anomaly class.")
    parser.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES),
                        help="Comma-separated class hints.")
    parser.add_argument("--specifier-model", type=str,
                        default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--explainer-model", type=str,
                        default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--recoverer-model", type=str,
                        default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--judge-model", type=str,
                        default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="static",
                        choices=["vanilla", "static", "rag"],
                        help="Business-context provider mode.")
    parser.add_argument("--rag-exclude-gold", action="store_true",
                        help="rag mode only: leave examples/gold_explanations.md "
                             "out of the retrieval corpus (leakage control).")
    parser.add_argument("--explainer-max-tokens", type=int, default=900,
                        help="Narrator completion budget. Reasoning-class models "
                             "(e.g. gpt-5.5) need more, e.g. 8000.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory (defaults to "
                             "results/recovery_<mode>).")
    args = parser.parse_args(argv)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    out_dir = (
        Path(args.out_dir) if args.out_dir
        else REPO_ROOT / "results" / f"recovery_{args.mode}"
    )

    if not (DATA_DIR / "normal.jsonl").exists():
        print(f"ERROR: missing {DATA_DIR / 'normal.jsonl'}", file=sys.stderr)
        print("       Run: python -m src.data_generator --seed 42 "
              "--n-normal 5000 --n-per-anomaly 250", file=sys.stderr)
        return 2

    print("[1/5] Loading normal traces and fitting detector...")
    normal_traces = [
        json.loads(line)["trace"]
        for line in (DATA_DIR / "normal.jsonl").open()
    ]
    detector = EmpiricalMarkovDetector()
    detector.fit(normal_traces)

    print("[2/5] Building four LLM roles + business-context provider "
          f"(mode={args.mode}, "
          f"specifier={args.specifier_model}, "
          f"explainer={args.explainer_model}, "
          f"recoverer={args.recoverer_model}, "
          f"judge={args.judge_model})...")
    specifier = AnomalySpecifier(model=args.specifier_model, temperature=0.7)
    materializer = TraceMaterializer()
    explainer = AnomalyExplainer(model=args.explainer_model,
                                 max_tokens=args.explainer_max_tokens)
    recoverer = AnomalyRecoverer(model=args.recoverer_model, temperature=0.2)
    comparator = SemanticComparator(model=args.judge_model, temperature=0.0)
    bc_provider = build_provider(args.mode, exclude_gold=args.rag_exclude_gold)

    runner = RecoveryFaithfulnessRunner(
        specifier=specifier,
        materializer=materializer,
        pipeline=build_pipeline_callable(detector, explainer, bc_provider),
        recoverer=recoverer,
        comparator=comparator,
    )

    print(f"[3/5] Running {args.n_per_class} evaluation(s) per class "
          f"on {classes}...")
    results = runner.run_many(classes, args.n_per_class, seed=args.seed)

    print("[4/5] Writing per-run records and summary...")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = out_dir / "per_run.jsonl"
    with per_run_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "spec": r.spec.to_dict(),
                "trace": r.trace,
                "explanation": r.explanation,
                "recovered": r.recovered.to_dict(),
                "fired_signals": r.fired_signals,
                "scores": r.scores,
            }) + "\n")

    # Aggregate
    summary: dict = {"n": len(results), "mode": args.mode,
                     "rag_exclude_gold": args.rag_exclude_gold,
                     "explainer_max_tokens": args.explainer_max_tokens,
                     "models": {"specifier": args.specifier_model,
                                "explainer": args.explainer_model,
                                "recoverer": args.recoverer_model,
                                "judge": args.judge_model},
                     "by_class": {}, "aggregate": {}}
    by_class: dict[str, list[dict]] = {}
    for r in results:
        by_class.setdefault(r.spec.class_hint or "free_form", []).append(r.scores)
    for cls, scores_list in sorted(by_class.items()):
        summary["by_class"][cls] = {
            "n": len(scores_list),
            "mean_judge": round(mean(s["judge_score"] for s in scores_list), 4),
            "mean_class_match": round(mean(s["class_match"] for s in scores_list), 4),
            "mean_rho": round(mean(s["rho"] for s in scores_list), 4),
        }
    all_scores = [r.scores for r in results]
    summary["aggregate"] = {
        "n": len(all_scores),
        "mean_judge": round(mean(s["judge_score"] for s in all_scores), 4) if all_scores else None,
        "mean_class_match": round(mean(s["class_match"] for s in all_scores), 4) if all_scores else None,
        "mean_rho": round(mean(s["rho"] for s in all_scores), 4) if all_scores else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")

    print("[5/5] Done. Summary:\n")
    print(f"  Aggregate ρ̄ = {summary['aggregate']['mean_rho']}  "
          f"(n={summary['aggregate']['n']})")
    print(f"  Aggregate judge = {summary['aggregate']['mean_judge']}, "
          f"class match = {summary['aggregate']['mean_class_match']}")
    print()
    print(f"  {'Class':<20s} {'n':>4s} {'judge':>8s} {'class':>8s} {'rho':>8s}")
    for cls, m in sorted(summary["by_class"].items()):
        print(f"  {cls:<20s} {m['n']:>4d} {m['mean_judge']:>8.3f} "
              f"{m['mean_class_match']:>8.3f} {m['mean_rho']:>8.3f}")
    print()
    print(f"Wrote {per_run_path}")
    print(f"Wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
