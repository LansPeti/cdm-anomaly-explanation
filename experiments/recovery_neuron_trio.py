"""NEURON-trio measurement: faithfulness + plausibility + sensemaking.

Adopts the three-dimensional explainability evaluation from NEURON
(arXiv 2605.01189, 2026), originally for clinical-domain grounded
narration. Faithfulness is already measured by the recovery framework
(rho); this script adds:

  - plausibility_score:  competent-operator believability of the
                         explanation, independent of truth
  - sensemaking_score:   does the explanation enable the analyst to
                         act?

We grade the EXISTING 40-round-trip RAG output (gpt-4o-mini), so this
adds two new LLM-as-judge passes per row. Total ~80 new LLM calls.

Also adds embedding similarity (sigma_embed) as a second similarity
signal next to the existing LLM judge.

Outputs:
  results/recovery_neuron/per_run.jsonl
  results/recovery_neuron/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

from src.specification_recovery import (
    AnomalySpecification,
    SemanticComparator,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "results" / "recovery_rag" / "per_run.jsonl"
OUT_DIR = REPO_ROOT / "results" / "recovery_neuron"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--judge-model", type=str,
                        default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--include-embedding", action="store_true",
                        help="Also compute sigma_embed via OpenAI embeddings.")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.", file=sys.stderr)
        return 2

    print(f"[1/3] Loading runs from {input_path.name}...")
    runs: list[dict] = []
    with input_path.open() as f:
        for line in f:
            runs.append(json.loads(line))
    print(f"      Loaded {len(runs)} runs.")

    comparator = SemanticComparator(model=args.judge_model, temperature=0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[2/3] Grading plausibility + sensemaking + (optional) embedding "
          f"on {len(runs)} entries...")
    out: list[dict] = []
    for i, row in enumerate(runs, start=1):
        plaus_score, plaus_reason = comparator.plausibility_score(row["explanation"])
        sense_score, sense_reason = comparator.sensemaking_score(row["explanation"])
        record = {
            "trade_label": row["spec"]["class_hint"],
            "judge_score": row["scores"]["judge_score"],  # existing faithfulness
            "class_match": row["scores"]["class_match"],
            "rho": row["scores"]["rho"],
            "plausibility": plaus_score,
            "plausibility_reasoning": plaus_reason,
            "sensemaking": sense_score,
            "sensemaking_reasoning": sense_reason,
        }
        if args.include_embedding:
            spec_a = AnomalySpecification(**{
                k: v for k, v in row["spec"].items()
                if k in ("nl_description", "class_hint", "metadata")
            })
            spec_b = AnomalySpecification(**{
                k: v for k, v in row["recovered"].items()
                if k in ("nl_description", "class_hint", "metadata")
            })
            record["embedding_similarity"] = comparator.embedding_similarity(
                spec_a, spec_b
            )
        out.append(record)
        marker = (
            f"plaus={plaus_score:.2f} sense={sense_score:.2f}"
            + (f" emb={record.get('embedding_similarity', 0):.2f}"
               if args.include_embedding else "")
        )
        print(f"  [{i}/{len(runs)}] {row['spec']['class_hint']:<18s} "
              f"rho={row['scores']['rho']:.2f} {marker}")

    print("[3/3] Aggregating...")
    summary = {
        "n": len(out),
        "mean_faithfulness_rho": round(mean(r["rho"] for r in out), 4),
        "mean_judge": round(mean(r["judge_score"] for r in out), 4),
        "mean_plausibility": round(mean(r["plausibility"] for r in out), 4),
        "mean_sensemaking": round(mean(r["sensemaking"] for r in out), 4),
    }
    if args.include_embedding:
        summary["mean_embedding_similarity"] = round(
            mean(r["embedding_similarity"] for r in out), 4
        )

    (OUT_DIR / "per_run.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")

    print()
    print(f"  faithfulness ρ̄ = {summary['mean_faithfulness_rho']}")
    print(f"  plausibility   = {summary['mean_plausibility']}")
    print(f"  sensemaking    = {summary['mean_sensemaking']}")
    if args.include_embedding:
        print(f"  σ_embed        = {summary['mean_embedding_similarity']}")
    print()
    print(f"Wrote {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
