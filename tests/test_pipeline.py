"""
test_pipeline.py — fast tests for the data side, the contract, and the pack.

Tests in this file do NOT call the OpenAI API. The LLM layer is exercised
via `validate_explanation` on hand-crafted strings, plus a smoke test of
the prompt assembly path.

Run:
    pytest tests/
"""
from __future__ import annotations

import json
import random

import pytest

from src.data_generator import (
    generate_dataset,
    generate_normal_trace,
    inject_impossible_transition,
    inject_pingpong,
    inject_skipped_step,
)
from src.detector import DEFAULT_THRESHOLDS, MockDetector
from src.evidence_pack import FLAG_INTERPRETATIONS, build_evidence_pack
from src.flags import DetectionResult, SequenceFlag, TransitionFlag
from src.lifecycle_graph import (
    TERMINAL_STATES,
    is_legal_transition,
    legal_successors,
    shortest_legal_path,
)
from src.llm_explainer import validate_explanation

# ---------------------------------------------------------------------------
# lifecycle_graph
# ---------------------------------------------------------------------------

class TestLifecycleGraph:
    def test_happy_path_legal(self):
        path = ["NEW", "EXECUTED", "CAPTURED", "ENRICHED",
                "MATCHED", "CLEARED", "SETTLED"]
        for a, b in zip(path, path[1:]):
            assert is_legal_transition(a, b), f"{a}->{b} should be legal"

    def test_skipped_step_illegal(self):
        # NEW->SETTLED is not directly legal
        assert not is_legal_transition("NEW", "SETTLED")
        # CAPTURED->MATCHED skips ENRICHED
        assert not is_legal_transition("CAPTURED", "MATCHED")

    def test_terminal_no_outgoing(self):
        for t in TERMINAL_STATES:
            assert legal_successors(t) == []

    def test_shortest_path_exists(self):
        p = shortest_legal_path("NEW", "SETTLED")
        assert p is not None
        assert p[0] == "NEW" and p[-1] == "SETTLED"
        # All edges legal
        for a, b in zip(p, p[1:]):
            assert is_legal_transition(a, b)


# ---------------------------------------------------------------------------
# data_generator
# ---------------------------------------------------------------------------

class TestDataGenerator:
    @pytest.fixture
    def rng(self):
        return random.Random(42)

    def test_normal_trace_starts_at_new(self, rng):
        for _ in range(20):
            trace = generate_normal_trace(rng)
            assert trace[0] == "NEW"
            assert trace[-1] in TERMINAL_STATES

    def test_normal_trace_only_legal_transitions(self, rng):
        for _ in range(50):
            trace = generate_normal_trace(rng)
            for a, b in zip(trace, trace[1:]):
                assert is_legal_transition(a, b), \
                    f"Generated illegal transition {a}->{b}"

    def test_inject_skipped_step_creates_illegal_edge(self, rng):
        # Find a base trace long enough
        for _ in range(20):
            base = generate_normal_trace(rng)
            new_trace, meta = inject_skipped_step(base, rng)
            if new_trace is not None:
                idx = meta["at_index"]
                # The transition that bridges the gap is illegal
                assert not is_legal_transition(
                    new_trace[idx-1], new_trace[idx]
                )
                return
        pytest.fail("Could not produce a skipped_step injection in 20 tries")

    def test_inject_impossible_transition(self, rng):
        for _ in range(20):
            base = generate_normal_trace(rng)
            new_trace, meta = inject_impossible_transition(base, rng)
            if new_trace is not None:
                idx = meta["at_index"]
                assert not is_legal_transition(
                    new_trace[idx-1], new_trace[idx]
                )
                return
        pytest.fail("Could not inject impossible_transition in 20 tries")

    def test_inject_pingpong_increases_length(self, rng):
        for _ in range(20):
            base = generate_normal_trace(rng)
            if "ENRICHED" not in base:
                continue
            new_trace, meta = inject_pingpong(base, rng)
            if new_trace is not None:
                assert len(new_trace) > len(base)
                assert meta["n_cycles"] >= 5
                # Edge ENRICHED->UNMATCHED_ERROR appears multiple times
                count = sum(
                    1 for i in range(len(new_trace) - 1)
                    if new_trace[i] == "ENRICHED" and
                       new_trace[i+1] == "UNMATCHED_ERROR"
                )
                assert count >= 5
                return
        pytest.fail("Could not inject pingpong in 20 tries")

    def test_dataset_assembly_balanced(self):
        normal, anomalies = generate_dataset(n_normal=50, n_per_anomaly=10)
        assert len(normal) == 50
        # 4 anomaly classes; allow some retries to fail (rare)
        assert 30 <= len(anomalies) <= 40
        labels = {r["label"] for r in anomalies}
        assert labels >= {"skipped_step", "impossible",
                          "pingpong", "unlikely_branch"}

    def test_record_schema(self):
        normal, _ = generate_dataset(n_normal=5, n_per_anomaly=1)
        for r in normal:
            assert {"trade_id", "trace", "events", "label",
                    "is_anomalous", "metadata"} <= r.keys()
            assert r["is_anomalous"] is False
            assert len(r["events"]) == len(r["trace"]) - 1


# ---------------------------------------------------------------------------
# flags + detector contract
# ---------------------------------------------------------------------------

class TestDetectorContract:
    def test_mock_detector_clean_result(self):
        d = MockDetector()
        d.fit([])
        result = d.detect(["NEW", "EXECUTED", "CAPTURED", "ENRICHED",
                           "MATCHED", "CLEARED", "SETTLED"])
        assert result.is_anomalous is False
        assert result.fired_signals == []
        assert result.transition_flags == []
        assert isinstance(result.sequence_flag, SequenceFlag)

    def test_default_thresholds_present(self):
        for k in ("PT_SURPRISE_NATS", "PT_RANK_MIN",
                  "SQ_NORM_LL_Z_MAX", "SQ_EDGE_REPEAT_MIN"):
            assert k in DEFAULT_THRESHOLDS


# ---------------------------------------------------------------------------
# evidence_pack
# ---------------------------------------------------------------------------

def _fake_anomalous_detection() -> DetectionResult:
    """Construct a hand-built DetectionResult for a CAPTURED->MATCHED skip."""
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
    sq = SequenceFlag(
        norm_log_likelihood=-1.87,
        norm_ll_z_score=-2.45,
        max_edge_repeat=1,
        most_repeated_edge=("NEW", "EXECUTED"),
        fired_flags=["SQ-low-likelihood"],
    )
    return DetectionResult(
        transition_flags=[tf], sequence_flag=sq,
        is_anomalous=True,
        fired_signals=["PT-impossible", "PT-surprise", "PT-rank",
                       "SQ-low-likelihood"],
    )


class TestEvidencePack:
    def test_normal_pack_has_empty_findings(self):
        d = MockDetector()
        result = d.detect(["NEW", "EXECUTED", "CAPTURED", "ENRICHED",
                           "MATCHED", "CLEARED", "SETTLED"])
        pack = build_evidence_pack("TRD-T-001",
                                   ["NEW", "EXECUTED", "CAPTURED", "ENRICHED",
                                    "MATCHED", "CLEARED", "SETTLED"],
                                   result)
        assert pack["transition_findings"] == []
        assert pack["verdict"]["is_anomalous"] is False
        assert pack["sequence_level"]["fired"] == []

    def test_anomalous_pack_includes_finding(self):
        trace = ["NEW", "EXECUTED", "CAPTURED", "MATCHED", "CLEARED", "SETTLED"]
        det = _fake_anomalous_detection()
        pack = build_evidence_pack("TRD-T-002", trace, det)
        assert pack["verdict"]["is_anomalous"] is True
        assert pack["verdict"]["fired_signals"] == det.fired_signals
        assert len(pack["transition_findings"]) == 1
        f = pack["transition_findings"][0]
        assert f["observed_transition"] == {"from": "CAPTURED", "to": "MATCHED"}
        assert f["expected_top3"][0]["state"] == "ENRICHED"

    def test_pack_is_json_serializable(self):
        trace = ["NEW", "EXECUTED", "CAPTURED", "MATCHED", "CLEARED", "SETTLED"]
        det = _fake_anomalous_detection()
        pack = build_evidence_pack("TRD-T-003", trace, det)
        # Must round-trip
        assert json.loads(json.dumps(pack)) == pack

    def test_flag_interpretations_cover_all_flag_names(self):
        for name in ("PT-impossible", "PT-surprise", "PT-rank",
                     "SQ-low-likelihood", "SQ-edge-repeat"):
            assert name in FLAG_INTERPRETATIONS


# ---------------------------------------------------------------------------
# llm_explainer (no API calls)
# ---------------------------------------------------------------------------

class TestExplanationValidator:
    def test_normal_pack_must_have_literal_response(self):
        pack = {"verdict": {"is_anomalous": False}, "trace": ["NEW"]}
        good = "Normal trace, no findings."
        bad = "## 1. Observed Trace\n..."
        assert validate_explanation(good, pack)["ok"] is True
        assert validate_explanation(bad, pack)["ok"] is False

    def test_anomalous_pack_requires_all_sections(self):
        trace = ["NEW", "EXECUTED", "CAPTURED", "MATCHED",
                 "CLEARED", "SETTLED"]
        det = _fake_anomalous_detection()
        pack = build_evidence_pack("TRD-T-004", trace, det)
        good = "\n\n".join([
            "## 1. Observed Trace\nThe trade went NEW -> EXECUTED ... (trace).",
            "## 2. Why It Is Flagged\nFlagged (verdict.fired_signals).",
            "## 3. What Normal Would Look Like\nExpected ENRICHED "
            "(transition_findings[0].expected_top3[0]).",
            "## 4. Recommended Validation Step\nVerify (verdict).",
        ])
        v = validate_explanation(good, pack)
        assert v["ok"], v["issues"]

    def test_validator_catches_missing_section(self):
        trace = ["NEW", "EXECUTED"]
        det = _fake_anomalous_detection()
        pack = build_evidence_pack("TRD-T-005", trace, det)
        bad = "## 1. Observed Trace\nx (trace)."  # missing 2,3,4
        v = validate_explanation(bad, pack)
        assert not v["ok"]
        assert any("Missing required section" in i for i in v["issues"])
