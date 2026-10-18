"""
evidence_pack.py — Convert a DetectionResult into a JSON pack for the LLM.

The pack is the SINGLE SOURCE OF TRUTH the LLM reads. It is intentionally
schema-stable so the LLM prompt and the few-shot examples never have to
change when the detector changes (Markov / MLR / XGB / …).

DESIGN PRINCIPLES (from 2024–2026 research)
==========================================

  - ProM-Ex (IEEE 2025): detector outputs are INPUTS; LLM is ONLY narrator.
  - Berti et al. (arXiv 2407.02310): zero-shot LLMs FAIL at trace anomaly
    judgment; never let the LLM judge.
  - Chain of Evidences (ACL 2025): every prose claim must cite a field.
  - Counterfactual prompting (Liu et al. arXiv 2502.18156): include top-k
    expected continuations alongside the observed one.

CONSEQUENCES FOR THE PACK
=========================

  1. Every numeric signal is a top-level field. The LLM cites by name.
  2. The verdict (`is_anomalous`, `fired_signals`) is computed by the
     DETECTOR, never derived by the LLM.
  3. Each transition_finding includes `expected_top3` so the LLM can write
     "the model expected X with p=0.85" without inventing alternatives.
  4. Sequence-level signals are first-class even when no per-transition
     flags fire (the pingpong case).
"""
from __future__ import annotations

from collections import defaultdict

from src.business_context import (
    BusinessContextProvider,
    StaticMappingProvider,
)
from src.flags import DetectionResult


def _edge_count_distribution(trace: list[str]) -> dict[str, int]:
    """Pre-compute edge counts in a JSON-friendly form ("FROM->TO": count)."""
    counts: dict[str, int] = defaultdict(int)
    for i in range(len(trace) - 1):
        counts[f"{trace[i]}->{trace[i+1]}"] += 1
    return dict(counts)


# Human-readable interpretations of each fired flag. The LLM is allowed to
# rephrase but must keep the meaning. Kept here so it stays close to the
# pack schema and not buried in the prompt.
FLAG_INTERPRETATIONS: dict[str, str] = {
    "PT-impossible":
        "The transition was never observed in normal training data — "
        "structurally invalid under the fitted model.",
    "PT-surprise":
        "The transition has very low probability (high information content) "
        "under the fitted model.",
    "PT-rank":
        "The observed next-state was not among the top expected "
        "continuations from the source state.",
    "SQ-low-likelihood":
        "The trace's overall normalized log-likelihood is unusually low "
        "compared to normal traces.",
    "SQ-edge-repeat":
        "An edge in the trace repeats far more often than is normal — "
        "consistent with a cyclic or stuck-loop pattern.",
}


# Per-flag declarative validation requirements. Section 4 of the LLM output
# enumerates the items that apply to the trace's fired_signals — verbatim,
# without LLM-generated causal hypothesis. This closes the "narrator not
# judge" boundary leak identified in an internal design review:
# Section 4 was previously LLM-generated free-form prescription with no
# corresponding pack field. It is now declarative pack content.
FLAG_VALIDATION_REQUIREMENTS: dict[str, list[str]] = {
    "PT-impossible": [
        "Verify whether the source-system event log records an intermediate "
        "state that was discarded by the audit pipeline before reaching the "
        "lifecycle store.",
        "Confirm that the recorded state names are not the result of a "
        "renaming or schema mismatch with the reference state machine.",
    ],
    "PT-surprise": [
        "Confirm whether the recorded transition reflects a legitimate but "
        "rare workflow variant by checking the source-system audit trail.",
    ],
    "PT-rank": [
        "Cross-check whether the higher-ranked expected continuations were "
        "considered before the observed one was committed.",
    ],
    "SQ-low-likelihood": [
        "Investigate whether the trace path is a known but rare workflow "
        "variant absent from the training distribution rather than a "
        "control failure.",
    ],
    "SQ-edge-repeat": [
        "Inspect whether the repeated edge represents a stuck repair loop "
        "that should escalate to manual review under the operational "
        "service-level policy.",
    ],
}


def _build_validation_requirements(fired_signals: list[str]) -> list[str]:
    """Compose Section 4 content deterministically from fired_signals.

    Order: keep the order of fired_signals; deduplicate while preserving
    first occurrence. The result is the verbatim content of Section 4 —
    no LLM-generated content allowed in that slot.
    """
    seen: set[str] = set()
    items: list[str] = []
    for sig in fired_signals:
        for req in FLAG_VALIDATION_REQUIREMENTS.get(sig, []):
            if req not in seen:
                seen.add(req)
                items.append(req)
    return items


def build_evidence_pack(
    trade_id: str,
    trace: list[str],
    detection: DetectionResult,
    *,
    include_unflagged_transitions: bool = False,
    model_context: dict | None = None,
    business_context_provider: BusinessContextProvider | None = None,
) -> dict:
    """Build a JSON-serializable evidence pack from a DetectionResult.

    Parameters
    ----------
    trade_id
        Identifier of the trade being explained. Used in the LLM output.
    trace
        The observed state sequence.
    detection
        Result produced by the detector. Must be already populated with
        thresholds applied (i.e. fired_flags lists are filled).
    include_unflagged_transitions
        If True, transition_findings includes EVERY transition; if False
        (default), only those with at least one fired flag.
    model_context
        Optional fitted-model context, typically the trace-local slice of
        the learned transition matrix. This is explanatory context only;
        the detector verdict remains the source of truth.
    business_context_provider
        Pluggable provider that fills the `business_context` field. If
        None, defaults to StaticMappingProvider (in-repo declarative
        semantics by lifecycle phase). Pass a different provider — e.g.
        RagProvider — to wire in firm-internal documents without changing
        the rest of the pipeline.

    Returns
    -------
    dict
        Pack ready to be JSON-serialized and passed to the LLM.
    """
    pack: dict = {
        "trade_id": trade_id,
        "trace": list(trace),
        "trace_length": len(trace),
        "edge_count_distribution": _edge_count_distribution(trace),
    }

    # Sequence-level block — always present
    seq = detection.sequence_flag
    seq_block = {
        "norm_log_likelihood": round(seq.norm_log_likelihood, 4),
        "norm_ll_z_score":     round(seq.norm_ll_z_score, 3),
        "max_edge_repeat":     int(seq.max_edge_repeat),
        "most_repeated_edge":  list(seq.most_repeated_edge)
                               if seq.most_repeated_edge else None,
        "fired": [
            {
                "name": flag,
                "interpretation": FLAG_INTERPRETATIONS.get(flag, ""),
            }
            for flag in seq.fired_flags
        ],
    }
    pack["sequence_level"] = seq_block

    # Per-transition findings (filtered)
    findings = []
    for tf in detection.transition_flags:
        if not include_unflagged_transitions and not tf.fired_flags:
            continue
        findings.append({
            "step": tf.step_index,
            "observed_transition": {
                "from": tf.from_state,
                "to":   tf.to_state,
            },
            "p_observed":          round(tf.p_observed, 6),
            "surprise_nats":       round(tf.surprise_nats, 3),
            "rank_among_candidates": tf.rank,
            "n_candidates":        tf.n_candidates,
            "gap_from_top1":       round(tf.gap_from_top1, 4),
            "expected_top3": [
                {"state": s, "probability": round(p, 6)}
                for s, p in tf.expected_top3
            ],
            "training_count":      tf.training_count,
            "fired_flags":         list(tf.fired_flags),
            "fired_interpretations": [
                FLAG_INTERPRETATIONS.get(f, "") for f in tf.fired_flags
            ],
        })
    pack["transition_findings"] = findings

    if model_context is not None:
        pack["model_context"] = model_context

    # Verdict (computed by detector, NOT by the LLM)
    pack["verdict"] = {
        "is_anomalous":   bool(detection.is_anomalous),
        "fired_signals":  list(detection.fired_signals),
    }

    # Validation requirements: declarative Section 4 content. The LLM is
    # required to enumerate these verbatim and may NOT generate Section 4
    # content from its world model. Closes the narrator-not-judge gap.
    pack["validation_requirements"] = _build_validation_requirements(
        list(detection.fired_signals)
    )

    # Business context: pluggable provider fills phase, owner,
    # regulatory relevance, etc. Single architectural seam where firm-
    # internal knowledge plugs in (replace the provider; nothing else
    # changes).
    provider = business_context_provider or StaticMappingProvider()
    pack["business_context"] = provider.context_for(
        list(trace), list(detection.fired_signals)
    ).to_dict()

    return pack
