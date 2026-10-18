"""Flag ablation study.

Empirically tests the redundancy hypothesis (internal design review): the reduced
flag set {PT-impossible, SQ-low-likelihood, SQ-edge-repeat} achieves
identical recall to the full 5-flag set on the four synthetic anomaly
classes, with marginally lower aggregate FP. This is the empirical
basis for demoting PT-rank and PT-surprise to evidence-pack metadata
rather than detection signals.

How it works
------------
The detector is *instantiated* with different threshold dicts that
effectively disable individual flags by setting them to values that
cannot fire. We do NOT modify the detector code. We only pass
different `thresholds=` kwargs to the constructor.

  - "PT-surprise off"  → set PT_SURPRISE_NATS to a huge value
  - "PT-rank off"      → set PT_RANK_MIN to a huge value
  - etc.

Outputs
-------
  results/ablation/summary.json  — per-config recall + FP
  results/ablation/report.md     — markdown table

Usage
-----
    python -m experiments.ablation
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from src.detector import EmpiricalMarkovDetector

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "results" / "ablation"

# A "huge" value used to disable a flag without touching detector code.
DISABLE = 1e9


CONFIGURATIONS = {
    "full_5_flag": {
        # Detector defaults — all five flags active.
    },
    "no_PT_surprise": {
        "PT_SURPRISE_NATS": DISABLE,
    },
    "no_PT_rank": {
        "PT_RANK_MIN": DISABLE,
    },
    "no_PT_surprise_and_rank": {
        "PT_SURPRISE_NATS": DISABLE,
        "PT_RANK_MIN": DISABLE,
    },
    # the predicted minimum: keep the structurally meaningful flag
    # (PT-impossible) and the two sequence-level flags.
    "minimal_PT_imp_plus_seq": {
        "PT_SURPRISE_NATS": DISABLE,
        "PT_RANK_MIN": DISABLE,
    },
    "no_SQ_low_likelihood": {
        "SQ_NORM_LL_Z_MAX": -DISABLE,
    },
    "no_SQ_edge_repeat": {
        "SQ_EDGE_REPEAT_MIN": DISABLE,
    },
}


def run_one(label: str, thresholds: dict, normal_traces: list[list[str]],
            anomaly_records: list[dict]) -> dict:
    det = EmpiricalMarkovDetector(thresholds=thresholds)
    det.fit(normal_traces)

    # Recall on anomalies, by class
    by_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "detected": 0}
    )
    for rec in anomaly_records:
        cls = rec["label"]
        result = det.detect(rec["trace"])
        by_class[cls]["n"] += 1
        if result.is_anomalous:
            by_class[cls]["detected"] += 1

    # FP on normal traces (we already have them in normal_traces)
    fp = 0
    for trace in normal_traces:
        result = det.detect(trace)
        if result.is_anomalous:
            fp += 1

    return {
        "label": label,
        "thresholds": {k: v for k, v in thresholds.items() if v != DISABLE},
        "by_class": {
            cls: {
                "n": stats["n"],
                "detected": stats["detected"],
                "recall": stats["detected"] / stats["n"] if stats["n"] else 0.0,
            }
            for cls, stats in by_class.items()
        },
        "fp_normal": {
            "n": len(normal_traces),
            "fp": fp,
            "rate": fp / len(normal_traces) if normal_traces else 0.0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    normal_path = DATA_DIR / "normal.jsonl"
    anomaly_path = DATA_DIR / "anomalies.jsonl"
    if not normal_path.exists() or not anomaly_path.exists():
        print(f"ERROR: data files missing in {DATA_DIR}/.", file=sys.stderr)
        print("       Run: python -m src.data_generator --seed 42 "
              "--n-normal 5000 --n-per-anomaly 250", file=sys.stderr)
        return 2

    print("Loading datasets...")
    with normal_path.open() as f:
        normal_records = [json.loads(line) for line in f]
    with anomaly_path.open() as f:
        anomaly_records = [json.loads(line) for line in f]
    normal_traces = [r["trace"] for r in normal_records]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(CONFIGURATIONS)} configurations on "
          f"{len(normal_traces)} normal + {len(anomaly_records)} anomalous traces...\n")

    runs: list[dict] = []
    for label, thresh in CONFIGURATIONS.items():
        run = run_one(label, thresh, normal_traces, anomaly_records)
        runs.append(run)
        per_class = " ".join(
            f"{cls[:5]}:{stats['recall'] * 100:.1f}%"
            for cls, stats in sorted(run["by_class"].items())
        )
        print(f"  {label:30s}  FP={run['fp_normal']['rate'] * 100:5.2f}%   "
              f"{per_class}")

    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")

    # Human-readable markdown table
    md_lines = ["# Ablation Study: Flag-Set Comparison\n"]
    md_lines.append(
        "Each configuration disables a subset of the detector's flags via "
        "constructor `thresholds=` overrides. The detector code "
        "(src/detector.py) is not modified.\n\n"
    )
    classes = sorted(runs[0]["by_class"].keys())
    header = "| Configuration | FP rate | " + " | ".join(
        f"{c} recall" for c in classes
    ) + " |"
    sep = "|" + "---|" * (len(classes) + 2)
    md_lines.append(header)
    md_lines.append(sep)
    for run in runs:
        recalls = [
            f"{run['by_class'][c]['recall'] * 100:.1f}%"
            for c in classes
        ]
        md_lines.append(
            f"| {run['label']} | "
            f"{run['fp_normal']['rate'] * 100:.2f}% | "
            + " | ".join(recalls) + " |"
        )
    md_lines.append("\n")
    md_lines.append(
        "## Interpretation\n\nIf `minimal_PT_imp_plus_seq` matches "
        "`full_5_flag` on recall, the redundancy hypothesis is "
        "empirically confirmed and PT-rank / PT-surprise can be demoted "
        "to evidence-pack metadata.\n"
    )
    (OUT_DIR / "report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nWrote {out_path} and {OUT_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
