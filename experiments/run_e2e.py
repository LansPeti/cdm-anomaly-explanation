"""End-to-end pipeline runner.

Wires together the production layers without modifying any of them:

    data_generator outputs (data/normal.jsonl, data/anomalies.jsonl)
        → EmpiricalMarkovDetector (src/detector.py)
            → build_evidence_pack (src/evidence_pack.py)
                → AnomalyExplainer (src/llm_explainer.py)

Outputs one markdown file per explained trace under results/e2e/,
plus a summary JSON (results/e2e/summary.json) listing every run.

Usage
-----
    python -m experiments.run_e2e                         # default 1 per class
    python -m experiments.run_e2e --per-class 3           # 3 per anomaly class
    python -m experiments.run_e2e --classes pingpong      # only one class
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from eval.eval_runner import check_hallucinations
from src.detector import EmpiricalMarkovDetector
from src.evidence_pack import build_evidence_pack
from src.llm_explainer import AnomalyExplainer, validate_explanation

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "results" / "e2e"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def select_records_per_class(records: list[dict], per_class: int,
                             classes: set[str] | None) -> list[dict]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        label = r.get("label", "unknown")
        if classes is not None and label not in classes:
            continue
        by_class[label].append(r)
    selected: list[dict] = []
    for label, recs in sorted(by_class.items()):
        selected.extend(recs[:per_class])
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=1,
                        help="How many anomalous traces to explain per anomaly class.")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated label filter, e.g. 'pingpong,impossible'.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18",
                        help="OpenAI chat model id (pinned by default).")
    args = parser.parse_args(argv)

    classes = (
        set(s.strip() for s in args.classes.split(",")) if args.classes else None
    )

    normal_path = DATA_DIR / "normal.jsonl"
    anomaly_path = DATA_DIR / "anomalies.jsonl"
    if not normal_path.exists() or not anomaly_path.exists():
        print(f"ERROR: missing dataset under {DATA_DIR}/. Run:", file=sys.stderr)
        print("  python -m src.data_generator --seed 42 --n-normal 5000 "
              "--n-per-anomaly 250", file=sys.stderr)
        return 2

    print(f"[1/4] Loading {normal_path.name}...", flush=True)
    normal_records = load_jsonl(normal_path)
    normal_traces = [r["trace"] for r in normal_records]

    print(f"[2/4] Fitting EmpiricalMarkovDetector on {len(normal_traces)} normal traces...",
          flush=True)
    detector = EmpiricalMarkovDetector()
    detector.fit(normal_traces)

    print(f"[3/4] Selecting traces (per_class={args.per_class})...", flush=True)
    anomaly_records = load_jsonl(anomaly_path)
    selected = select_records_per_class(anomaly_records, args.per_class, classes)
    print(f"      Selected {len(selected)} traces.", flush=True)

    print(f"[4/4] Detect → evidence pack → LLM explain (model={args.model})...",
          flush=True)
    explainer = AnomalyExplainer(model=args.model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    for i, record in enumerate(selected, start=1):
        trade_id = record.get("trade_id", f"TRD-{i:06d}")
        label = record.get("label", "unknown")
        trace = record["trace"]

        result = detector.detect(trace)
        model_context = detector.export_trace_transition_context(trace)
        pack = build_evidence_pack(
            trade_id,
            trace,
            result,
            model_context=model_context,
        )
        explanation = explainer.explain(pack)
        validation = validate_explanation(explanation.text, pack)
        hallucination = check_hallucinations(explanation.text, pack)

        out_md = OUT_DIR / f"{label}_{trade_id}.md"
        out_md.write_text(
            f"# {trade_id} — {label}\n\n"
            f"**Fired signals:** {', '.join(result.fired_signals) or '(none)'}\n\n"
            f"**Trace:** `{' → '.join(trace)}`\n\n"
            f"---\n\n{explanation.text}\n",
            encoding="utf-8",
        )
        out_pack = OUT_DIR / f"{label}_{trade_id}.pack.json"
        out_pack.write_text(json.dumps(pack, indent=2), encoding="utf-8")

        summary.append({
            "trade_id": trade_id,
            "label": label,
            "fired_signals": result.fired_signals,
            "n_transition_findings": len(pack["transition_findings"]),
            "validation": validation,
            "hallucination": hallucination,
            "usage": explanation.usage,
            "explanation_path": str(out_md.relative_to(REPO_ROOT)),
        })
        valid = "OK" if validation.get("ok") else "FAIL"
        hall = "clean" if hallucination.get("ok") else f"FAIL({len(hallucination.get('issues', []))})"
        print(f"  [{i}/{len(selected)}] {label:18s} {trade_id:30s} "
              f"signals={result.fired_signals!s:60s} valid={valid} hall={hall}",
              flush=True)

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {len(selected)} explanations and summary to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
