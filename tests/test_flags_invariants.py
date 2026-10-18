"""Adversarial tests for the dataclass __post_init__ invariants in flags.py.

These guard the detector→evidence-pack→LLM contract boundary against the
silent-corruption failure modes catalogued in the internal design review
.
"""
from __future__ import annotations

import pytest

from src.flags import DetectionResult, SequenceFlag, TransitionFlag

# ---------------------------------------------------------------------------
# Reusable valid kwargs (so each test only changes one field)
# ---------------------------------------------------------------------------

def _valid_tf(**overrides) -> dict:
    base = dict(
        step_index=0,
        from_state="MATCHED",
        to_state="CLEARED",
        p_observed=0.9715,
        surprise_nats=0.0289,
        rank=1,
        n_candidates=12,
        gap_from_top1=0.0,
        expected_top3=[("CLEARED", 0.9715), ("CANCELLED", 0.0184), ("AMENDED", 0.0001)],
        training_count=4_812,
        impossible=False,
        fired_flags=[],
    )
    base.update(overrides)
    return base


def _valid_sf(**overrides) -> dict:
    base = dict(
        norm_log_likelihood=-0.32,
        norm_ll_z_score=-0.4,
        max_edge_repeat=1,
        most_repeated_edge=("NEW", "EXECUTED"),
        fired_flags=[],
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TransitionFlag invariants
# ---------------------------------------------------------------------------

class TestTransitionFlagInvariants:

    def test_baseline_valid_kwargs_construct(self):
        tf = TransitionFlag(**_valid_tf())
        assert tf.p_observed == 0.9715

    @pytest.mark.parametrize("p", [0.0, -0.0001, -1.0])
    def test_p_observed_zero_or_negative_rejected(self, p):
        with pytest.raises(ValueError, match="p_observed"):
            TransitionFlag(**_valid_tf(p_observed=p))

    @pytest.mark.parametrize("p", [1.0001, 2.0])
    def test_p_observed_above_one_rejected(self, p):
        with pytest.raises(ValueError, match="p_observed"):
            TransitionFlag(**_valid_tf(p_observed=p))

    def test_surprise_negative_rejected(self):
        with pytest.raises(ValueError, match="surprise_nats"):
            TransitionFlag(**_valid_tf(surprise_nats=-0.5))

    def test_surprise_zero_accepted(self):
        # surprise == -log(1.0) == 0; legal at the boundary
        tf = TransitionFlag(**_valid_tf(p_observed=1.0, surprise_nats=0.0))
        assert tf.surprise_nats == 0.0

    @pytest.mark.parametrize("rank", [0, -1])
    def test_rank_below_one_rejected(self, rank):
        with pytest.raises(ValueError, match="rank"):
            TransitionFlag(**_valid_tf(rank=rank))

    def test_negative_training_count_rejected(self):
        with pytest.raises(ValueError, match="training_count"):
            TransitionFlag(**_valid_tf(training_count=-1, impossible=False))

    def test_impossible_inconsistency_with_training_count_rejected(self):
        # impossible=True but training_count != 0 — the failure mode that
        # caused the LLM to narrate "never seen" while training_count was 5
        with pytest.raises(ValueError, match="impossible"):
            TransitionFlag(**_valid_tf(training_count=5, impossible=True))

    def test_impossible_consistency_with_zero_training_count(self):
        tf = TransitionFlag(**_valid_tf(
            training_count=0, impossible=True, p_observed=0.0001,
            fired_flags=["PT-impossible"],
        ))
        assert tf.impossible is True

    def test_impossible_false_with_zero_training_count_rejected(self):
        # Inverse: training_count=0 must imply impossible=True.
        with pytest.raises(ValueError, match="impossible"):
            TransitionFlag(**_valid_tf(training_count=0, impossible=False))

    def test_negative_gap_from_top1_rejected(self):
        with pytest.raises(ValueError, match="gap_from_top1"):
            TransitionFlag(**_valid_tf(gap_from_top1=-0.5))

    def test_expected_top3_ascending_rejected(self):
        # The failure mode that would cause the LLM to narrate the
        # least-likely state as the most probable expected continuation.
        ascending = [("CANCELLED", 0.01), ("CLEARED", 0.95), ("AMENDED", 0.97)]
        with pytest.raises(ValueError, match="expected_top3"):
            TransitionFlag(**_valid_tf(expected_top3=ascending))

    def test_expected_top3_unsorted_in_middle_rejected(self):
        unsorted = [("A", 0.9), ("B", 0.3), ("C", 0.5)]
        with pytest.raises(ValueError, match="expected_top3"):
            TransitionFlag(**_valid_tf(expected_top3=unsorted))

    def test_expected_top3_two_items_descending_accepted(self):
        # Some detectors may have a small vocabulary
        tf = TransitionFlag(**_valid_tf(
            expected_top3=[("X", 0.7), ("Y", 0.3)],
        ))
        assert len(tf.expected_top3) == 2

    def test_expected_top3_equal_probabilities_accepted(self):
        # Tied probabilities are not strict descending, but valid (no inversion).
        tf = TransitionFlag(**_valid_tf(
            expected_top3=[("A", 0.5), ("B", 0.5), ("C", 0.0)],
        ))
        assert tf.expected_top3[0][1] == tf.expected_top3[1][1]


# ---------------------------------------------------------------------------
# SequenceFlag invariants
# ---------------------------------------------------------------------------

class TestSequenceFlagInvariants:

    def test_baseline_valid_kwargs_construct(self):
        sf = SequenceFlag(**_valid_sf())
        assert sf.max_edge_repeat == 1

    def test_negative_max_edge_repeat_rejected(self):
        with pytest.raises(ValueError, match="max_edge_repeat"):
            SequenceFlag(**_valid_sf(max_edge_repeat=-1))

    def test_zero_max_edge_repeat_accepted(self):
        # Trace of length < 2 has no edges
        sf = SequenceFlag(**_valid_sf(max_edge_repeat=0, most_repeated_edge=None))
        assert sf.max_edge_repeat == 0


# ---------------------------------------------------------------------------
# DetectionResult composition (verifies child invariants propagate)
# ---------------------------------------------------------------------------

class TestDetectionResultComposition:

    def test_construction_with_valid_children(self):
        tf = TransitionFlag(**_valid_tf())
        sf = SequenceFlag(**_valid_sf())
        dr = DetectionResult(
            transition_flags=[tf],
            sequence_flag=sf,
            is_anomalous=False,
            fired_signals=[],
        )
        assert dr.is_anomalous is False

    def test_corrupt_transition_flag_propagates_failure(self):
        # If a detector tries to construct a corrupt TransitionFlag, the
        # error surfaces at the dataclass boundary, not later in JSON
        # serialisation or LLM narration.
        with pytest.raises(ValueError):
            TransitionFlag(**_valid_tf(p_observed=0.0))
