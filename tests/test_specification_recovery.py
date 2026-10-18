"""Offline tests for the Specification-Recovery Faithfulness module.

These tests do NOT call the OpenAI API. The LLM-backed components
(AnomalySpecifier, AnomalyRecoverer, SemanticComparator) are exercised
through the standard unittest mock patterns; the deterministic parts
(AnomalySpecification, TraceMaterializer, RecoveryFaithfulnessRunner
wiring) are tested directly.
"""
from __future__ import annotations

import json
import random
from unittest.mock import MagicMock

import pytest

from src.specification_recovery import (
    AnomalyRecoverer,
    AnomalySpecification,
    AnomalySpecifier,
    RecoveryFaithfulnessRunner,
    SemanticComparator,
    TraceMaterializer,
)

# ---------------------------------------------------------------------------
# Test fixtures: a fake OpenAI client whose responses we control.
# ---------------------------------------------------------------------------

def _fake_response(content: str) -> MagicMock:
    """Build a MagicMock that mimics the structure of an OpenAI completion."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _fake_client_returning(content: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(content)
    return client


# ---------------------------------------------------------------------------
# AnomalySpecification dataclass
# ---------------------------------------------------------------------------

class TestAnomalySpecification:

    def test_round_trip_through_to_dict(self):
        spec = AnomalySpecification(
            nl_description="An ENRICHED state is followed directly by SETTLED.",
            class_hint="impossible",
            metadata={"observed_jump": ["ENRICHED", "SETTLED"]},
        )
        d = spec.to_dict()
        assert d["nl_description"].startswith("An ENRICHED")
        assert d["class_hint"] == "impossible"
        assert d["metadata"] == {"observed_jump": ["ENRICHED", "SETTLED"]}

    def test_default_metadata_is_independent_per_instance(self):
        a = AnomalySpecification(nl_description="x")
        b = AnomalySpecification(nl_description="y")
        a.metadata["k"] = 1
        assert "k" not in b.metadata


# ---------------------------------------------------------------------------
# TraceMaterializer
# ---------------------------------------------------------------------------

class TestTraceMaterializer:

    @pytest.mark.parametrize("class_hint", [
        "skipped_step", "impossible", "pingpong", "unlikely_branch",
    ])
    def test_known_class_hint_yields_a_trace(self, class_hint):
        # Without metadata, materialize() falls back to the random injectors.
        spec = AnomalySpecification(
            nl_description="dummy",
            class_hint=class_hint,
        )
        rng = random.Random(42)
        m = TraceMaterializer()
        trace = m.materialize(spec, rng)
        assert isinstance(trace, list)
        assert len(trace) >= 2
        assert all(isinstance(s, str) for s in trace)

    def test_spec_aware_skipped_step_omits_named_state(self):
        spec = AnomalySpecification(
            nl_description="x",
            class_hint="skipped_step",
            metadata={"removed_state": "ENRICHED"},
        )
        m = TraceMaterializer()
        trace = m.materialize(spec, random.Random(0))
        assert "ENRICHED" not in trace
        assert trace == ["NEW", "EXECUTED", "CAPTURED",
                         "MATCHED", "CLEARED", "SETTLED"]

    def test_spec_aware_impossible_inserts_named_jump(self):
        spec = AnomalySpecification(
            nl_description="x",
            class_hint="impossible",
            metadata={"after": "MATCHED", "injected": "EXECUTED"},
        )
        m = TraceMaterializer()
        trace = m.materialize(spec, random.Random(0))
        assert trace[-2:] == ["MATCHED", "EXECUTED"]

    def test_spec_aware_pingpong_uses_n_cycles(self):
        spec = AnomalySpecification(
            nl_description="x",
            class_hint="pingpong",
            metadata={"n_cycles": 7},
        )
        m = TraceMaterializer()
        trace = m.materialize(spec, random.Random(0))
        # 7 cycles of ENRICHED -> UNMATCHED_ERROR
        n_unmatched = sum(1 for s in trace if s == "UNMATCHED_ERROR")
        assert n_unmatched == 7

    def test_spec_aware_pingpong_clamps_n_cycles(self):
        spec = AnomalySpecification(
            nl_description="x",
            class_hint="pingpong",
            metadata={"n_cycles": 100},
        )
        m = TraceMaterializer()
        trace = m.materialize(spec, random.Random(0))
        n_unmatched = sum(1 for s in trace if s == "UNMATCHED_ERROR")
        assert n_unmatched == 10  # clamped

    def test_spec_aware_unlikely_branch_after_NEW(self):
        spec = AnomalySpecification(
            nl_description="x",
            class_hint="unlikely_branch",
            metadata={"after": "NEW"},
        )
        m = TraceMaterializer()
        trace = m.materialize(spec, random.Random(0))
        assert trace == ["NEW", "REJECTED"]

    def test_spec_aware_unlikely_branch_after_EXECUTED(self):
        spec = AnomalySpecification(
            nl_description="x",
            class_hint="unlikely_branch",
            metadata={"after": "EXECUTED"},
        )
        m = TraceMaterializer()
        trace = m.materialize(spec, random.Random(0))
        assert trace == ["NEW", "EXECUTED", "REJECTED"]

    def test_invalid_metadata_falls_back_to_random_injector(self):
        # If removed_state is missing/invalid, the spec-aware path returns
        # None and we fall through to the random injector.
        spec = AnomalySpecification(
            nl_description="x",
            class_hint="skipped_step",
            metadata={"removed_state": "NOT_A_REAL_STATE"},
        )
        m = TraceMaterializer()
        trace = m.materialize(spec, random.Random(42))
        # Some valid trace was produced (not the spec-aware one).
        assert isinstance(trace, list) and len(trace) >= 2

    def test_free_form_without_fallback_raises(self):
        spec = AnomalySpecification(nl_description="something weird",
                                    class_hint=None)
        m = TraceMaterializer()
        with pytest.raises(NotImplementedError):
            m.materialize(spec, random.Random(0))

    def test_free_form_with_fallback_used(self):
        called = {"n": 0}

        def fallback(spec, rng):
            called["n"] += 1
            return ["NEW", "EXECUTED"]

        m = TraceMaterializer(free_form_fallback=fallback)
        spec = AnomalySpecification(nl_description="x", class_hint=None)
        trace = m.materialize(spec, random.Random(0))
        assert trace == ["NEW", "EXECUTED"]
        assert called["n"] == 1


# ---------------------------------------------------------------------------
# AnomalySpecifier (with mocked OpenAI client)
# ---------------------------------------------------------------------------

class TestAnomalySpecifier:

    def test_specify_parses_returned_json(self):
        client = _fake_client_returning(json.dumps({
            "nl_description": "An impossible MATCHED -> EXECUTED transition.",
            "class_hint": "impossible",
            "metadata": {"jump": ["MATCHED", "EXECUTED"]},
        }))
        s = AnomalySpecifier(client=client)
        spec = s.specify("impossible")
        assert spec.class_hint == "impossible"
        assert "MATCHED" in spec.nl_description
        assert spec.metadata["jump"] == ["MATCHED", "EXECUTED"]

    def test_specify_falls_back_to_class_hint_if_missing(self):
        # If the LLM forgets to echo class_hint, we keep the requested one.
        client = _fake_client_returning(json.dumps({
            "nl_description": "Trade is missing a step.",
            "metadata": {},
        }))
        s = AnomalySpecifier(client=client)
        spec = s.specify("skipped_step")
        assert spec.class_hint == "skipped_step"


# ---------------------------------------------------------------------------
# AnomalyRecoverer (with mocked OpenAI client)
# ---------------------------------------------------------------------------

class TestAnomalyRecoverer:

    def test_recover_parses_returned_json(self):
        client = _fake_client_returning(json.dumps({
            "nl_description": "ENRICHED follows itself in a loop.",
            "class_hint": "pingpong",
            "metadata": {"loop_states": ["ENRICHED", "UNMATCHED_ERROR"]},
        }))
        r = AnomalyRecoverer(client=client)
        recovered = r.recover("explanation text here")
        assert recovered.class_hint == "pingpong"
        assert recovered.metadata["loop_states"] == ["ENRICHED", "UNMATCHED_ERROR"]


# ---------------------------------------------------------------------------
# SemanticComparator
# ---------------------------------------------------------------------------

class TestSemanticComparator:

    def test_class_match_is_one_when_hints_equal(self):
        a = AnomalySpecification(nl_description="x", class_hint="pingpong")
        b = AnomalySpecification(nl_description="y", class_hint="pingpong")
        assert SemanticComparator.class_match(a, b) == 1

    def test_class_match_is_zero_when_hints_differ(self):
        a = AnomalySpecification(nl_description="x", class_hint="pingpong")
        b = AnomalySpecification(nl_description="y", class_hint="impossible")
        assert SemanticComparator.class_match(a, b) == 0

    def test_judge_score_normalises_to_zero_one(self):
        client = _fake_client_returning(json.dumps({
            "score": 85,
            "reasoning": "Both describe a repeating ENRICHED loop.",
        }))
        c = SemanticComparator(client=client)
        score, reasoning = c.judge_score(
            AnomalySpecification(nl_description="A"),
            AnomalySpecification(nl_description="B"),
        )
        assert 0.84 < score < 0.86
        assert "ENRICHED" in reasoning

    def test_judge_score_clamped_to_zero_one(self):
        # Defensive: even if the judge returns a wonky score we clamp.
        client = _fake_client_returning(json.dumps({
            "score": 200, "reasoning": "out of range",
        }))
        c = SemanticComparator(client=client)
        score, _ = c.judge_score(
            AnomalySpecification(nl_description="A"),
            AnomalySpecification(nl_description="B"),
        )
        assert score == 1.0

    def test_rho_combines_judge_and_class_match(self):
        client = _fake_client_returning(json.dumps({
            "score": 100, "reasoning": "Identical.",
        }))
        c = SemanticComparator(client=client)
        a = AnomalySpecification(nl_description="x", class_hint="pingpong")
        b = AnomalySpecification(nl_description="y", class_hint="pingpong")
        out = c.rho(a, b)
        # 0.7 * 1.0 (judge) + 0.3 * 1 (match) = 1.0
        assert out["judge_score"] == 1.0
        assert out["class_match"] == 1
        assert abs(out["rho"] - 1.0) < 1e-9

    def test_rho_class_mismatch_caps_score(self):
        client = _fake_client_returning(json.dumps({
            "score": 100, "reasoning": "Same wording.",
        }))
        c = SemanticComparator(client=client)
        a = AnomalySpecification(nl_description="x", class_hint="pingpong")
        b = AnomalySpecification(nl_description="y", class_hint="impossible")
        out = c.rho(a, b)
        # Even with perfect judge agreement, missing class match drops ρ.
        assert abs(out["rho"] - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# RecoveryFaithfulnessRunner (full wiring with all four LLMs mocked)
# ---------------------------------------------------------------------------

class TestRecoveryFaithfulnessRunner:

    def test_run_one_threads_through_all_four_stages(self):
        spec_payload = {
            "nl_description": "An impossible MATCHED -> EXECUTED jump.",
            "class_hint": "impossible",
            "metadata": {},
        }
        recover_payload = {
            "nl_description": "MATCHED transitioned back to EXECUTED.",
            "class_hint": "impossible",
            "metadata": {},
        }
        judge_payload = {"score": 90, "reasoning": "Same anomaly."}

        specifier = AnomalySpecifier(
            client=_fake_client_returning(json.dumps(spec_payload))
        )
        recoverer = AnomalyRecoverer(
            client=_fake_client_returning(json.dumps(recover_payload))
        )
        comparator = SemanticComparator(
            client=_fake_client_returning(json.dumps(judge_payload))
        )

        def fake_pipeline(trace):
            return ("explanation text", ["PT-impossible"])

        runner = RecoveryFaithfulnessRunner(
            specifier=specifier,
            materializer=TraceMaterializer(),
            pipeline=fake_pipeline,
            recoverer=recoverer,
            comparator=comparator,
        )
        result = runner.run_one("impossible", random.Random(42))

        assert result.spec.class_hint == "impossible"
        assert result.recovered.class_hint == "impossible"
        assert result.fired_signals == ["PT-impossible"]
        assert 0.85 < result.scores["judge_score"] < 0.95
        assert result.scores["class_match"] == 1
        # 0.7 * 0.9 + 0.3 * 1 = 0.93
        assert abs(result.scores["rho"] - 0.93) < 1e-2
