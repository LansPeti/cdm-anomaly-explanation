"""Tests for the evidence-pack builder.

Particularly tests the new validation_requirements field that closes
the narrator-not-judge boundary leak in Section 4.
"""
from __future__ import annotations

from src.detector import EmpiricalMarkovDetector
from src.evidence_pack import (
    FLAG_VALIDATION_REQUIREMENTS,
    _build_validation_requirements,
    build_evidence_pack,
)
from src.flags import DetectionResult, SequenceFlag, TransitionFlag


def _seq(*, fired_flags=None, **overrides) -> SequenceFlag:
    base = dict(
        norm_log_likelihood=-0.5,
        norm_ll_z_score=-0.4,
        max_edge_repeat=1,
        most_repeated_edge=("NEW", "EXECUTED"),
        fired_flags=fired_flags or [],
    )
    base.update(overrides)
    return SequenceFlag(**base)


def _tf(**overrides) -> TransitionFlag:
    base = dict(
        step_index=2,
        from_state="CAPTURED",
        to_state="MATCHED",
        p_observed=0.0001,
        surprise_nats=9.17,
        rank=7,
        n_candidates=12,
        gap_from_top1=0.95,
        expected_top3=[("ENRICHED", 0.96), ("CANCELLED", 0.04), ("EXECUTED", 0.0001)],
        training_count=0,
        impossible=True,
        fired_flags=["PT-impossible", "PT-surprise", "PT-rank"],
    )
    base.update(overrides)
    return TransitionFlag(**base)


class TestValidationRequirements:

    def test_empty_signals_yields_empty(self):
        assert _build_validation_requirements([]) == []

    def test_single_signal_yields_its_items(self):
        out = _build_validation_requirements(["SQ-edge-repeat"])
        assert out == FLAG_VALIDATION_REQUIREMENTS["SQ-edge-repeat"]

    def test_unknown_signal_silently_ignored(self):
        out = _build_validation_requirements(["NOT-A-REAL-FLAG"])
        assert out == []

    def test_signals_dedup_preserving_order(self):
        # Adding the same signal twice yields a single occurrence.
        out = _build_validation_requirements(["PT-impossible", "PT-impossible"])
        assert out == FLAG_VALIDATION_REQUIREMENTS["PT-impossible"]

    def test_multiple_signals_compose_in_order(self):
        out = _build_validation_requirements(
            ["PT-impossible", "SQ-edge-repeat"]
        )
        # PT-impossible items come first (in their original order), then
        # SQ-edge-repeat items.
        pt = FLAG_VALIDATION_REQUIREMENTS["PT-impossible"]
        sq = FLAG_VALIDATION_REQUIREMENTS["SQ-edge-repeat"]
        assert out == pt + sq


class TestBuildEvidencePack:

    def test_normal_trace_pack_has_empty_validation_requirements(self):
        det = DetectionResult(
            transition_flags=[],
            sequence_flag=_seq(),
            is_anomalous=False,
            fired_signals=[],
        )
        pack = build_evidence_pack("TRD-N-1", ["NEW", "EXECUTED"], det)
        assert pack["validation_requirements"] == []
        assert pack["verdict"]["is_anomalous"] is False

    def test_anomalous_pack_has_populated_validation_requirements(self):
        det = DetectionResult(
            transition_flags=[_tf()],
            sequence_flag=_seq(fired_flags=["SQ-low-likelihood"],
                              norm_ll_z_score=-2.5),
            is_anomalous=True,
            fired_signals=["PT-impossible", "PT-surprise", "PT-rank",
                           "SQ-low-likelihood"],
        )
        pack = build_evidence_pack(
            "TRD-A-1",
            ["NEW", "EXECUTED", "CAPTURED", "MATCHED"],
            det,
        )
        assert len(pack["validation_requirements"]) > 0
        # Must include items from each fired signal at least once
        assert any(
            "intermediate state" in r for r in pack["validation_requirements"]
        )

    def test_pack_round_trips_through_json(self):
        import json
        det = DetectionResult(
            transition_flags=[_tf()],
            sequence_flag=_seq(),
            is_anomalous=True,
            fired_signals=["PT-impossible"],
        )
        pack = build_evidence_pack("TRD-A-2", ["NEW", "EXECUTED"], det)
        serialised = json.dumps(pack)
        restored = json.loads(serialised)
        assert restored["validation_requirements"] == pack["validation_requirements"]

    def test_unflagged_transitions_filtered_by_default(self):
        det = DetectionResult(
            transition_flags=[],  # No flagged transitions
            sequence_flag=_seq(fired_flags=["SQ-edge-repeat"], max_edge_repeat=4),
            is_anomalous=True,
            fired_signals=["SQ-edge-repeat"],
        )
        pack = build_evidence_pack(
            "TRD-PP-1",
            ["NEW", "EXECUTED", "CAPTURED", "ENRICHED"],
            det,
        )
        assert pack["transition_findings"] == []
        assert pack["validation_requirements"]  # SQ-edge-repeat items
        assert pack["sequence_level"]["fired"][0]["name"] == "SQ-edge-repeat"

    def test_pack_can_include_trace_local_model_context(self):
        detector = EmpiricalMarkovDetector()
        detector.fit([
            ["NEW", "EXECUTED", "CAPTURED", "ENRICHED"],
            ["NEW", "EXECUTED", "CAPTURED", "CANCELLED"],
            ["NEW", "REJECTED"],
        ])
        trace = ["NEW", "EXECUTED", "CAPTURED", "MATCHED"]
        model_context = detector.export_trace_transition_context(trace)

        det = DetectionResult(
            transition_flags=[],
            sequence_flag=_seq(fired_flags=["SQ-low-likelihood"]),
            is_anomalous=True,
            fired_signals=["SQ-low-likelihood"],
        )
        pack = build_evidence_pack(
            "TRD-MC-1",
            trace,
            det,
            model_context=model_context,
        )

        assert pack["model_context"]["source"] == "fitted_normal_transition_matrix"
        learned = pack["model_context"]["learned_transition_top_k"]
        assert set(learned) == {"NEW", "EXECUTED", "CAPTURED"}
        assert learned["NEW"][0]["to_state"] == "EXECUTED"
        assert "MATCHED" not in learned
