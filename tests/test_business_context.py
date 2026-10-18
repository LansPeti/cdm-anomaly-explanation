"""Tests for the pluggable BusinessContextProvider abstraction.

Covers the static mapping (deterministic) and the RAG provider (mocked
so no API call is made).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.business_context import (
    PHASE_BUSINESS_SEMANTICS,
    STATE_TO_PHASE,
    BusinessContextProvider,
    RagProvider,
    StaticMappingProvider,
    default_provider,
)

# ---------------------------------------------------------------------------
# Mapping integrity
# ---------------------------------------------------------------------------

class TestMappingIntegrity:

    def test_every_state_in_state_to_phase_resolves_to_known_phase(self):
        for state, phase in STATE_TO_PHASE.items():
            assert phase in PHASE_BUSINESS_SEMANTICS, (
                f"State {state!r} maps to {phase!r}, but {phase!r} has no "
                "entry in PHASE_BUSINESS_SEMANTICS"
            )

    def test_every_phase_has_full_schema(self):
        required = {
            "phase_purpose",
            "typical_operational_owner",
            "regulatory_relevance",
            "common_failure_causes",
            "downstream_dependencies_blocked",
        }
        for phase, sem in PHASE_BUSINESS_SEMANTICS.items():
            missing = required - set(sem)
            assert not missing, f"Phase {phase!r} missing fields: {missing}"

    def test_every_lifecycle_state_is_covered(self):
        # If lifecycle_graph adds a new state, this test catches it.
        from src.lifecycle_graph import STATES
        missing = set(STATES) - set(STATE_TO_PHASE)
        assert not missing, (
            f"States in lifecycle_graph but not in STATE_TO_PHASE: {missing}"
        )


# ---------------------------------------------------------------------------
# StaticMappingProvider
# ---------------------------------------------------------------------------

class TestStaticMappingProvider:

    def test_default_provider_returns_static(self):
        p = default_provider()
        assert isinstance(p, StaticMappingProvider)

    def test_phase_inference_from_last_non_terminal_state(self):
        p = StaticMappingProvider()
        ctx = p.context_for(
            ["NEW", "EXECUTED", "CAPTURED", "MATCHED"],
            ["PT-impossible"],
        )
        assert ctx.current_lifecycle_phase == "matching"
        # Sanity: the phase purpose is non-empty and substantive.
        assert len(ctx.phase_purpose) > 30
        assert "Middle office" in ctx.typical_operational_owner

    def test_terminal_only_trace_returns_terminal_phase(self):
        p = StaticMappingProvider()
        ctx = p.context_for(["NEW", "REJECTED"], [])
        # "matching" appears nowhere here; both NEW (pre_execution) and
        # REJECTED (terminal_failure) are present, so we get the last
        # non-terminal phase = pre_execution.
        assert ctx.current_lifecycle_phase == "pre_execution"

    def test_pingpong_trace_resolves_to_repair_or_matching(self):
        # Pingpong trace ends ENRICHED -> ... -> SETTLED (in our
        # synthetic generator). The last informative state is SETTLED
        # → settlement phase.
        p = StaticMappingProvider()
        trace = ["NEW", "EXECUTED", "CAPTURED", "ENRICHED",
                 "UNMATCHED_ERROR", "AMENDED", "ENRICHED",
                 "MATCHED", "CLEARED", "SETTLED"]
        ctx = p.context_for(trace, ["SQ-edge-repeat"])
        assert ctx.current_lifecycle_phase == "settlement"

    def test_empty_trace_returns_pre_execution(self):
        p = StaticMappingProvider()
        ctx = p.context_for([], [])
        assert ctx.current_lifecycle_phase == "pre_execution"

    def test_to_dict_returns_json_serializable(self):
        p = StaticMappingProvider()
        ctx = p.context_for(["NEW", "EXECUTED", "MATCHED"], ["PT-rank"])
        d = ctx.to_dict()
        # Round-trip
        s = json.dumps(d)
        restored = json.loads(s)
        assert restored["current_lifecycle_phase"] == "matching"
        assert isinstance(restored["regulatory_relevance"], list)
        # No retrieved_excerpts field for static provider.
        assert "retrieved_excerpts" not in d

    def test_provider_inherits_abstract_base(self):
        # Mypy / runtime check: anyone subclassing must implement context_for.
        class IncompleteProvider(BusinessContextProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# RagProvider (mocked OpenAI client)
# ---------------------------------------------------------------------------

class _FakeEmbeddings:
    """A fake OpenAI embeddings client that returns deterministic vectors
    derived from the input text — no API call.
    """

    def __init__(self):
        self.calls: list[str] = []

    def create(self, model: str, input: str):
        self.calls.append(input)
        # Deterministic 8-dim vector based on character codes.
        codes = [ord(c) for c in input[:50]]
        vec = [(c % 7) / 7.0 for c in codes] + [0.0] * 8
        vec = vec[:8]
        # Wrap in OpenAI-shaped response.
        data_item = MagicMock()
        data_item.embedding = vec
        response = MagicMock()
        response.data = [data_item]
        return response


def _fake_openai_client() -> MagicMock:
    client = MagicMock()
    client.embeddings = _FakeEmbeddings()
    return client


class TestRagProvider:

    def test_indexes_and_retrieves_from_corpus(self, tmp_path):
        # Two short corpus files.
        (tmp_path / "matching.md").write_text(
            "Matching phase: counterparties confirm trade economics. "
            "Common failures include SSI mismatch and trade size differences.",
            encoding="utf-8",
        )
        (tmp_path / "settlement.md").write_text(
            "Settlement phase: securities and cash are exchanged on the "
            "agreed date. Failures stem from insufficient funds.",
            encoding="utf-8",
        )
        cache_dir = tmp_path / "embed_cache"
        provider = RagProvider(
            corpus_paths=[tmp_path / "matching.md", tmp_path / "settlement.md"],
            client=_fake_openai_client(),
            cache_dir=cache_dir,
            top_k=2,
            chunk_size=200,
            chunk_overlap=20,
        )
        ctx = provider.context_for(
            ["NEW", "EXECUTED", "CAPTURED", "MATCHED"],
            ["PT-impossible"],
        )
        # Retains the static-mapping skeleton fields.
        assert ctx.current_lifecycle_phase == "matching"
        # Adds retrieved excerpts.
        assert ctx.retrieved_excerpts is not None
        assert len(ctx.retrieved_excerpts) <= 2
        for excerpt in ctx.retrieved_excerpts:
            assert "source" in excerpt
            assert "score" in excerpt
            assert "text" in excerpt

    def test_embedding_cache_round_trip(self, tmp_path):
        (tmp_path / "doc.md").write_text("Some content here", encoding="utf-8")
        cache_dir = tmp_path / "cache"
        # First instantiation embeds and writes cache.
        RagProvider(
            corpus_paths=[tmp_path / "doc.md"],
            client=_fake_openai_client(),
            cache_dir=cache_dir,
            chunk_size=200,
        )
        # The cache directory now has at least one cache file.
        assert any(cache_dir.iterdir())
        # Second instantiation should not need to re-embed.
        client2 = _fake_openai_client()
        # Use a fresh client and verify it does NOT get called.
        RagProvider(
            corpus_paths=[tmp_path / "doc.md"],
            client=client2,
            cache_dir=cache_dir,
            chunk_size=200,
        )
        # client2.embeddings.create() should not have been invoked,
        # because the embedding came from cache.
        assert client2.embeddings.calls == []

    def test_to_dict_includes_retrieved_excerpts(self, tmp_path):
        (tmp_path / "doc.md").write_text(
            "Some content about the matching phase.", encoding="utf-8"
        )
        provider = RagProvider(
            corpus_paths=[tmp_path / "doc.md"],
            client=_fake_openai_client(),
            cache_dir=tmp_path / "cache",
            chunk_size=200,
        )
        ctx = provider.context_for(["NEW", "MATCHED"], [])
        d = ctx.to_dict()
        assert "retrieved_excerpts" in d
        assert isinstance(d["retrieved_excerpts"], list)
