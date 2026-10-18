"""Structural integrity tests for the lifecycle state machine.

These guard against schema drift between lifecycle_graph.STATES and the
data generator / detector / evidence pack consumers.
"""
from __future__ import annotations

import pytest

from src import lifecycle_graph as lg


class TestStateSpace:

    def test_states_is_a_list_of_strings(self):
        assert isinstance(lg.STATES, list)
        assert all(isinstance(s, str) for s in lg.STATES)

    def test_states_are_unique(self):
        assert len(lg.STATES) == len(set(lg.STATES))

    def test_canonical_states_present(self):
        # Canonical happy path states (per Bakshi et al. 2021)
        for s in ("NEW", "EXECUTED", "CAPTURED", "ENRICHED", "MATCHED",
                  "CLEARED", "SETTLED"):
            assert s in lg.STATES, f"Canonical state {s!r} missing from STATES"

    def test_terminal_states_subset_of_states(self):
        assert lg.TERMINAL_STATES.issubset(set(lg.STATES))

    def test_terminal_states_match_documented_three(self):
        # The pipeline assumes exactly these terminal states.
        assert lg.TERMINAL_STATES == {"SETTLED", "REJECTED", "CANCELLED"}


class TestLegalSuccessors:

    def test_legal_successors_function_exists(self):
        # Either as a function or a dict — both are acceptable APIs.
        legal = getattr(lg, "legal_successors", None) or getattr(
            lg, "LEGAL_SUCCESSORS", None
        )
        assert legal is not None, (
            "lifecycle_graph must expose legal_successors() or LEGAL_SUCCESSORS"
        )

    def test_terminal_states_have_no_outgoing_transitions(self):
        # Terminal states should not have any legal successors.
        legal = getattr(lg, "legal_successors", None)
        if legal is None:
            pytest.skip("legal_successors() not available")
        for s in lg.TERMINAL_STATES:
            successors = legal(s)
            assert successors in (None, [], {}, set()), (
                f"Terminal state {s!r} should have no successors, got {successors}"
            )


class TestSafeCapsConsistency:
    """The single-source-of-truth contract: anything that enumerates
    valid state names in the LLM layer or eval layer must derive from
    lifecycle_graph.STATES.
    """

    def test_llm_explainer_safe_caps_matches_states(self):
        from src.llm_explainer import _LIFECYCLE_STATES
        # The LLM explainer's safe_caps now derives from STATES, so this is
        # a tautology by construction — we test that the import succeeded.
        assert set(_LIFECYCLE_STATES) == set(lg.STATES)
