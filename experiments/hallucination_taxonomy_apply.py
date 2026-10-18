"""Apply the Berti 2026-style hallucination taxonomy to existing
explanations and report per-category breakdown.

Inputs (existing artefacts):
  results/e2e/per_run_files (the e2e-scale 100-trace run)
  results/e2e/summary.json
  results/recovery_rag/per_run.jsonl

Output:
  results/taxonomy/summary.json
  results/taxonomy/per_run.jsonl

No API calls — everything is regex / pack-traversal heuristic.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

from src.hallucination_taxonomy import diagnose

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "taxonomy"


def _iter_e2e_outputs() -> list[tuple[str, str, dict]]:
    """Read the e2e 100-trace output: (label, explanation_text, pack)."""
    out_dir = REPO_ROOT / "results" / "e2e"
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[tuple[str, str, dict]] = []
    for entry in summary:
        explanation_path = REPO_ROOT / entry["explanation_path"]
        pack_path = explanation_path.with_suffix(".pack.json").parent / (
            explanation_path.stem + ".pack.json"
        )
        if not explanation_path.exists() or not pack_path.exists():
            continue
        text = explanation_path.read_text(encoding="utf-8")
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        rows.append((entry["label"], text, pack))
    return rows


def _iter_recovery_outputs() -> list[tuple[str, str, dict]]:
    """Read the rag-mode 40-round-trip output."""
    path = REPO_ROOT / "results" / "recovery_rag" / "per_run.jsonl"
    if not path.exists():
        return []
    rows: list[tuple[str, str, dict]] = []
    for line in path.open():
        rec = json.loads(line)
        # Reconstruct a minimal pack (the RAG run did not save the pack
        # alongside, so we synthesise the fields the diagnoser needs).
        pack = {
            "trace": rec["trace"],
            "verdict": {"fired_signals": rec["fired_signals"]},
            "business_context": {},  # not preserved in this output
        }
        rows.append((rec["spec"]["class_hint"], rec["explanation"], pack))
    return rows


def main(argv: list[str] | None = None) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    e2e_rows = _iter_e2e_outputs()
    rag_rows = _iter_recovery_outputs()
    print(f"E2E rows: {len(e2e_rows)}, RAG-recovery rows: {len(rag_rows)}")

    all_results: list[dict] = []
    for source, rows in (("e2e_100", e2e_rows), ("rag_recovery", rag_rows)):
        for label, text, pack in rows:
            d = diagnose(text, pack).to_dict()
            d["label"] = label
            d["source"] = source
            all_results.append(d)

    if not all_results:
        print("No data to analyse.", file=sys.stderr)
        return 2

    (OUT_DIR / "per_run.jsonl").write_text(
        "\n".join(json.dumps(r) for r in all_results) + "\n",
        encoding="utf-8",
    )

    # Aggregate per-source
    summary: dict = {}
    for source in ("e2e_100", "rag_recovery"):
        rows_for_source = [r for r in all_results if r["source"] == source]
        if not rows_for_source:
            continue
        cat_counts = Counter()
        for r in rows_for_source:
            for cat in (
                "causal_prescription", "invented_state", "invented_flag",
                "unfaithful_elaboration", "entity_confusion", "omission",
            ):
                if r[cat]:
                    cat_counts[cat] += 1
        clean_count = sum(1 for r in rows_for_source if r["ok"])
        summary[source] = {
            "n": len(rows_for_source),
            "clean": clean_count,
            "clean_rate": round(clean_count / len(rows_for_source), 4),
            "category_incidence": dict(cat_counts),
            "category_incidence_pct": {
                cat: round(count / len(rows_for_source), 4)
                for cat, count in cat_counts.items()
            },
            "mean_total_findings": round(
                mean(r["total"] for r in rows_for_source), 3
            ),
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")

    print()
    for source, stats in summary.items():
        print(f"=== {source} ===")
        print(f"  n: {stats['n']}, clean: {stats['clean']} "
              f"({stats['clean_rate']*100:.1f}%)")
        print("  Category incidence:")
        for cat, pct in sorted(stats["category_incidence_pct"].items(),
                               key=lambda x: -x[1]):
            print(f"    {cat:<28s} {pct*100:5.1f}%")
        print()
    print(f"Wrote {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
