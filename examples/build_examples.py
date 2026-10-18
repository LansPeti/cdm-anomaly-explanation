"""
build_examples.py — Generate 5 example evidence packs for documentation.

Builds one example pack per anomaly class plus one normal pack, using
HAND-CRAFTED but realistic DetectionResult numbers (derived from the
empirical experiments in results/).

These examples serve two purposes:
  1. Show what numeric values a detector should produce.
  2. Show the LLM (via few-shot) what packs look like in production.

Run:
    python -m examples.build_examples
"""
from __future__ import annotations

import json
from pathlib import Path

from src.evidence_pack import build_evidence_pack
from src.flags import DetectionResult, SequenceFlag, TransitionFlag

# ---------------------------------------------------------------------------
# 1. NORMAL TRACE
# ---------------------------------------------------------------------------

def _normal_pack():
    trace = ["NEW", "EXECUTED", "CAPTURED", "ENRICHED",
             "MATCHED", "CLEARED", "SETTLED"]
    seq = SequenceFlag(
        norm_log_likelihood=-0.061,
        norm_ll_z_score=0.37,
        max_edge_repeat=1,
        most_repeated_edge=("NEW", "EXECUTED"),
        fired_flags=[],
    )
    det = DetectionResult(
        transition_flags=[],
        sequence_flag=seq,
        is_anomalous=False,
        fired_signals=[],
    )
    return trace, det, "TRD-EXAMPLE-NORMAL-001"


# ---------------------------------------------------------------------------
# 2. SKIPPED STEP — CAPTURED -> MATCHED (skips ENRICHED)
# ---------------------------------------------------------------------------

def _skipped_step_pack():
    trace = ["NEW", "EXECUTED", "CAPTURED", "MATCHED", "CLEARED", "SETTLED"]
    tf = TransitionFlag(
        step_index=2,
        from_state="CAPTURED",
        to_state="MATCHED",
        p_observed=0.0001,
        surprise_nats=9.17,
        rank=7,
        n_candidates=12,
        gap_from_top1=0.956,
        expected_top3=[("ENRICHED", 0.9561),
                       ("CANCELLED", 0.0428),
                       ("EXECUTED", 0.0001)],
        training_count=0,
        impossible=True,
        fired_flags=["PT-impossible", "PT-surprise", "PT-rank"],
    )
    seq = SequenceFlag(
        norm_log_likelihood=-1.867,
        norm_ll_z_score=-2.45,
        max_edge_repeat=1,
        most_repeated_edge=("NEW", "EXECUTED"),
        fired_flags=["SQ-low-likelihood"],
    )
    det = DetectionResult(
        transition_flags=[tf], sequence_flag=seq,
        is_anomalous=True,
        fired_signals=["PT-impossible", "PT-surprise", "PT-rank",
                       "SQ-low-likelihood"],
    )
    return trace, det, "TRD-EXAMPLE-SKIPPED-001"


# ---------------------------------------------------------------------------
# 3. IMPOSSIBLE TRANSITION — NEW -> FAILED_SETTLEMENT
# ---------------------------------------------------------------------------

def _impossible_pack():
    trace = ["NEW", "FAILED_SETTLEMENT", "CAPTURED", "ENRICHED",
             "MATCHED", "CLEARED", "SETTLED"]
    tf = TransitionFlag(
        step_index=0,
        from_state="NEW",
        to_state="FAILED_SETTLEMENT",
        p_observed=0.0001,
        surprise_nats=9.21,
        rank=9,
        n_candidates=12,
        gap_from_top1=0.971,
        expected_top3=[("EXECUTED", 0.9711),
                       ("REJECTED", 0.0279),
                       ("CLEARED", 0.0001)],
        training_count=0,
        impossible=True,
        fired_flags=["PT-impossible", "PT-surprise", "PT-rank"],
    )
    seq = SequenceFlag(
        norm_log_likelihood=-2.003,
        norm_ll_z_score=-2.67,
        max_edge_repeat=1,
        most_repeated_edge=("NEW", "FAILED_SETTLEMENT"),
        fired_flags=["SQ-low-likelihood"],
    )
    det = DetectionResult(
        transition_flags=[tf], sequence_flag=seq,
        is_anomalous=True,
        fired_signals=["PT-impossible", "PT-surprise", "PT-rank",
                       "SQ-low-likelihood"],
    )
    return trace, det, "TRD-EXAMPLE-IMPOSSIBLE-001"


# ---------------------------------------------------------------------------
# 4. PINGPONG — 8 cycles of ENRICHED <-> UNMATCHED_ERROR <-> AMENDED
# ---------------------------------------------------------------------------

def _pingpong_pack():
    trace = (["NEW", "EXECUTED", "CAPTURED", "ENRICHED"]
             + ["UNMATCHED_ERROR", "AMENDED", "ENRICHED"] * 8
             + ["MATCHED", "CLEARED", "SETTLED"])
    seq = SequenceFlag(
        norm_log_likelihood=-0.528,
        norm_ll_z_score=-0.36,            # <-- IMPORTANT: NOT flagged by likelihood
        max_edge_repeat=8,
        most_repeated_edge=("ENRICHED", "UNMATCHED_ERROR"),
        fired_flags=["SQ-edge-repeat"],
    )
    # No per-transition flags fire — every individual step is high probability.
    det = DetectionResult(
        transition_flags=[],
        sequence_flag=seq,
        is_anomalous=True,
        fired_signals=["SQ-edge-repeat"],
    )
    return trace, det, "TRD-EXAMPLE-PINGPONG-001"


# ---------------------------------------------------------------------------
# 5. UNLIKELY BRANCH — early REJECTED
# ---------------------------------------------------------------------------

def _unlikely_branch_pack():
    trace = ["NEW", "REJECTED"]
    seq = SequenceFlag(
        norm_log_likelihood=-3.580,
        norm_ll_z_score=-5.13,
        max_edge_repeat=1,
        most_repeated_edge=("NEW", "REJECTED"),
        fired_flags=["SQ-low-likelihood"],
    )
    # Per-transition: NEW -> REJECTED has p ≈ 0.029, surprise ≈ 3.5
    # which is BELOW the PT-surprise threshold of 4.0. So PT-flags don't fire.
    det = DetectionResult(
        transition_flags=[],
        sequence_flag=seq,
        is_anomalous=True,
        fired_signals=["SQ-low-likelihood"],
    )
    return trace, det, "TRD-EXAMPLE-UNLIKELY-001"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

EXAMPLES = {
    "pack_normal.json":           _normal_pack,
    "pack_skipped_step.json":     _skipped_step_pack,
    "pack_impossible.json":       _impossible_pack,
    "pack_pingpong.json":         _pingpong_pack,
    "pack_unlikely_branch.json":  _unlikely_branch_pack,
}


def main():
    out_dir = Path(__file__).parent
    for filename, builder in EXAMPLES.items():
        trace, det, trade_id = builder()
        pack = build_evidence_pack(trade_id, trace, det)
        path = out_dir / filename
        path.write_text(json.dumps(pack, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"  Wrote {path.relative_to(out_dir.parent)}")


if __name__ == "__main__":
    main()
