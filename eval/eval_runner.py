"""
eval_runner.py — Quality evaluation for the LLM explanation layer.

WHAT WE MEASURE
===============

Two complementary metrics, both mechanical (no human judging required):

  1. **Field-citation rate**: every claim in the explanation should cite a
     field from the evidence pack. We extract parenthetical citations and
     check they resolve to a field path in the pack.

  2. **Hallucination taxonomy**: based on Berti et al. and the Springer
     2025 PM-hallucination taxonomy, we look for specific failure modes:
       - Invented states (UPPER_SNAKE tokens not in the pack's `trace`)
       - Invented flag names (anything matching `(PT|SQ)-\\w+` not in
         the pack's fired_signals)
       - Causal language ("caused", "led to", "made the trade …")
       - Missing required sections
       - Section order violations

We deliberately do NOT use an LLM-as-judge because Berti et al. showed
LLM judgment is unreliable on trace anomaly tasks. All checks here are
deterministic.

USAGE
=====

    # Run on N anomaly traces with the EmpiricalMarkovDetector
    python -m eval.eval_runner --n 25 --data data/anomalies.jsonl

    # Or against a single trace given as JSON:
    python -m eval.eval_runner --pack examples/pack_pingpong.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.evidence_pack import build_evidence_pack
from src.lifecycle_graph import STATES as _LIFECYCLE_STATES
from src.llm_explainer import (
    REQUIRED_SECTIONS,
    AnomalyExplainer,
    validate_explanation,
)

# ---------------------------------------------------------------------------
# Citation extractor + checker
# ---------------------------------------------------------------------------

def extract_citations(text: str) -> list[str]:
    """Pull parenthetical text that looks like a field reference."""
    # Conservative: match parens whose content includes a dot or [ or known
    # top-level field name.
    candidates = re.findall(r"\(([^()]+)\)", text)
    out = []
    for c in candidates:
        if any(k in c for k in (
            ".", "[",
            "trace", "verdict", "sequence_level", "transition_findings",
            "model_context",
            "edge_count_distribution", "fired", "p_observed", "rank",
            "training_count", "norm_ll", "expected_top3",
        )):
            out.append(c.strip())
    return out


def citation_resolves(cite: str, pack: dict) -> bool:
    """Loose: does any token in the citation appear as a field in the pack?"""
    pack_str = json.dumps(pack)
    base = re.split(r"[\.\[]", cite)[0].strip()
    if not base:
        return False
    return base in pack_str


# ---------------------------------------------------------------------------
# Hallucination checks
# ---------------------------------------------------------------------------

CAUSAL_PHRASES = (
    " caused ", " caused.", " caused,",
    " led to ",
    " resulted in ",
    " because of the ",  # weak heuristic; often legitimate, so flagged with low weight
    " made the trade ",
)

# Derived from lifecycle_graph.STATES (single source of truth). Adding a
# state to the graph propagates here automatically; previously this was a
# hand-maintained duplicate that could silently drift.
VALID_STATES = frozenset(_LIFECYCLE_STATES)

VALID_FLAG_NAMES = frozenset({
    "PT-impossible", "PT-surprise", "PT-rank",
    "SQ-low-likelihood", "SQ-edge-repeat",
})


def check_hallucinations(text: str, pack: dict) -> dict:
    issues: list[str] = []

    # Invented states
    upper_tokens = set(re.findall(r"\b[A-Z][A-Z_]{2,}\b", text))
    invented_states = upper_tokens - VALID_STATES
    # Filter known section / flag tokens
    invented_states = {t for t in invented_states if not t.startswith(("PT", "SQ"))}
    invented_states.discard("LLM")
    invented_states.discard("CDM")
    invented_states.discard("ISDA")
    invented_states.discard("JSON")
    if invented_states:
        issues.append(f"Invented state-like tokens: {sorted(invented_states)}")

    # Invented flags
    flag_mentions = set(re.findall(r"\b(?:PT|SQ)-[a-z\-]+\b", text))
    invented_flags = flag_mentions - VALID_FLAG_NAMES
    if invented_flags:
        issues.append(f"Invented flag names: {sorted(invented_flags)}")

    # Causal language
    lc = text.lower()
    for phrase in CAUSAL_PHRASES:
        if phrase in lc:
            issues.append(f"Causal phrasing: {phrase.strip()!r}")
            break

    # Sections
    pos = -1
    for s in REQUIRED_SECTIONS:
        new_pos = text.find(s)
        if new_pos < 0:
            issues.append(f"Missing required section: {s!r}")
        elif new_pos < pos:
            issues.append(f"Section out of order: {s!r}")
        else:
            pos = new_pos

    return {"hallucination_ok": not issues, "issues": issues}


# ---------------------------------------------------------------------------
# Per-trace scoring
# ---------------------------------------------------------------------------

def score_one(pack: dict, explanation: str) -> dict:
    citations = extract_citations(explanation)
    resolved = sum(1 for c in citations if citation_resolves(c, pack))
    citation_rate = (resolved / len(citations)) if citations else 0.0

    halluc = check_hallucinations(explanation, pack)
    structural = validate_explanation(explanation, pack)

    return {
        "trade_id": pack.get("trade_id"),
        "is_anomalous": pack.get("verdict", {}).get("is_anomalous"),
        "n_citations": len(citations),
        "n_resolved": resolved,
        "citation_resolution_rate": round(citation_rate, 3),
        "hallucination_issues": halluc["issues"],
        "structural_issues": structural["issues"],
        "passed": (
            citation_rate >= 0.8
            and not halluc["issues"]
            and not structural["issues"]
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_against_dataset(jsonl_path: Path, n: int,
                        detector_name: str = "markov") -> list[dict]:
    """Run end-to-end against a JSONL dataset.

    Trains the named detector on `data/normal.jsonl`, runs it on the first
    `n` records of `jsonl_path`, and scores each LLM explanation for
    citation rate, hallucination, and structural conformance. Raises if
    `OPENAI_API_KEY` is unavailable — there is no longer a silent fallback
    that produces zero-result output with exit code 0.
    """
    if detector_name == "markov":
        from src.detector import EmpiricalMarkovDetector

        normal_path = Path("data/normal.jsonl")
        if not normal_path.exists():
            raise FileNotFoundError(
                "Missing data/normal.jsonl. Run: "
                "python -m src.data_generator --seed 42 --n-normal 5000 --n-per-anomaly 250"
            )

        with normal_path.open() as f:
            normal_traces = [json.loads(line)["trace"] for line in f]

        det = EmpiricalMarkovDetector()
        det.fit(normal_traces)

    else:
        raise NotImplementedError(
            f"Detector '{detector_name}' not wired."
        )

    try:
        explainer = AnomalyExplainer()
    except RuntimeError as exc:
        # Was: silent skip + scores=[] + exit 0. That produced output
        # indistinguishable from a legitimate zero-result run, which is
        # disqualifying for any audit-adjacent use. Surface the error.
        raise RuntimeError(
            "eval_runner cannot evaluate the LLM layer without an "
            "OPENAI_API_KEY. Set it in .env, export it, or pass --pack "
            "to score a single pre-built pack non-interactively. "
            f"Original error: {exc}"
        ) from exc

    scores = []
    with jsonl_path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            trace = rec["trace"]
            detection = det.detect(trace)
            model_context = det.export_trace_transition_context(trace)
            pack = build_evidence_pack(
                rec.get("trade_id", f"TRD-EVAL-{i}"),
                trace,
                detection,
                model_context=model_context,
            )
            result = explainer.explain(pack)
            scores.append(score_one(pack, result.text))
    return scores


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LLM explanation eval runner")
    ap.add_argument("--data", default="data/anomalies.jsonl",
                    help="JSONL file produced by data_generator")
    ap.add_argument("--n", type=int, default=25,
                    help="Number of traces to evaluate")
    ap.add_argument("--pack", help="Run against a single pre-built pack JSON")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args(argv)

    if args.pack:
        # One-shot mode
        with open(args.pack) as f:
            pack = json.load(f)
        try:
            explainer = AnomalyExplainer()
            result = explainer.explain(pack)
            score = score_one(pack, result.text)
            print(json.dumps({"explanation": result.text,
                              "score": score}, indent=2))
        except RuntimeError as e:
            print(f"Cannot run: {e}", file=sys.stderr)
            return 1
        return 0

    scores = run_against_dataset(Path(args.data), args.n, detector_name="markov")
    summary = {
        "n_evaluated": len(scores),
        "n_passed": sum(s["passed"] for s in scores),
        "avg_citation_rate": round(
            sum(s["citation_resolution_rate"] for s in scores) /
            max(len(scores), 1), 3),
        "scores": scores,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"Evaluated {summary['n_evaluated']}, passed {summary['n_passed']}, "
          f"avg citation rate {summary['avg_citation_rate']}")
    print(f"Detailed scores -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
