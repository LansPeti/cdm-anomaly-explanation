"""Determinism measurement.

Calls the LLM N times with the same evidence pack and reports how many
distinct responses come back. At T=0.2 we expect occasional drift; the
question is HOW occasional, and on which packs.

This is the empirical measurement that the review flagged as a MiFID II / DORA reproducibility concern.

Usage
-----
    python -m experiments.run_determinism                  # default 5 reps on 1 pack per class
    python -m experiments.run_determinism --reps 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from src.detector import EmpiricalMarkovDetector
from src.evidence_pack import build_evidence_pack
from src.llm_explainer import AnomalyExplainer

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "results" / "determinism"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5,
                        help="How many times to call the LLM with the same pack.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    args = parser.parse_args(argv)

    if not (DATA_DIR / "anomalies.jsonl").exists():
        print("ERROR: data/anomalies.jsonl missing. Run data_generator first.",
              file=sys.stderr)
        return 2

    print("[1/3] Loading dataset and fitting detector...")
    normal_traces = [json.loads(line)["trace"] for line in (DATA_DIR / "normal.jsonl").open()]
    detector = EmpiricalMarkovDetector()
    detector.fit(normal_traces)

    # Pick one anomalous trace per class (deterministically — first match)
    anomaly_records = [json.loads(line) for line in (DATA_DIR / "anomalies.jsonl").open()]
    seen: dict[str, dict] = {}
    for r in anomaly_records:
        if r["label"] not in seen:
            seen[r["label"]] = r
    targets = list(seen.values())
    print(f"      Selected {len(targets)} pack(s), one per anomaly class.")

    print("[2/3] Building evidence packs...")
    packs = []
    for r in targets:
        result = detector.detect(r["trace"])
        pack = build_evidence_pack(r["trade_id"], r["trace"], result)
        packs.append((r["label"], pack))

    print(f"[3/3] Calling LLM {args.reps}× per pack (model={args.model}, T=0.2)...")
    explainer = AnomalyExplainer(model=args.model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    for label, pack in packs:
        responses: list[str] = []
        hashes: list[str] = []
        for rep in range(args.reps):
            exp = explainer.explain(pack)
            responses.append(exp.text)
            hashes.append(hash_text(exp.text))
        unique = len(set(hashes))
        per_class_dir = OUT_DIR / label
        per_class_dir.mkdir(parents=True, exist_ok=True)
        for idx, (h, txt) in enumerate(zip(hashes, responses)):
            (per_class_dir / f"rep_{idx:02d}_{h}.md").write_text(txt, encoding="utf-8")
        summary.append({
            "label": label,
            "trade_id": pack["trade_id"],
            "n_reps": args.reps,
            "n_unique": unique,
            "is_deterministic": unique == 1,
            "hashes": hashes,
        })
        marker = "DETERMINISTIC" if unique == 1 else f"DRIFT ({unique} unique)"
        print(f"  {label:18s} {pack['trade_id']:30s} {marker}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    n_det = sum(1 for s in summary if s["is_deterministic"])
    print(f"\nResult: {n_det}/{len(summary)} packs deterministic at T=0.2 across "
          f"{args.reps} repetitions.")
    print(f"Output: {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
