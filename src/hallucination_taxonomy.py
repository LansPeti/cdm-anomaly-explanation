"""Hallucination taxonomy for constrained-narration LLM output.

Adopts the categorisation from Berti et al. (2026, Springer ICPM
Workshops, "Diagnosing LLM Hallucinations in Process Mining Tasks: a
Taxonomy and a Benchmark") with one trade-lifecycle-specific
extension. The categories are deliberately *narrow* so that a regex /
heuristic implementation is reliable enough to cite in a paper without
LLM-as-judge dependency.

Categories
----------
1. CAUSAL_PRESCRIPTION
   Phrases that assert causation or prescribe action without a
   citation to a pack field. Examples: "led to", "caused", "must",
   "should". This was the dominant failure in our pre-fix iteration.

2. INVENTED_STATE
   UPPER_SNAKE tokens that are not in lifecycle_graph.STATES nor in
   the small allow-list (LLM, CDM, ISDA, JSON, etc.). Catches
   hallucinated state names.

3. INVENTED_FLAG
   PT-* / SQ-* tokens not in VALID_FLAG_NAMES. Catches LLM-invented
   flag names.

4. UNFAITHFUL_ELABORATION
   Sentences that mention a regulatory or operational concept (e.g.
   MiFID II, EMIR, T+2 settlement) without a corresponding citation
   path in the pack. Surfaces extra "knowledge" the LLM brings
   without grounding.

5. ENTITY_CONFUSION
   The narrative refers to a state or transition that does not
   appear in pack.trace or pack.transition_findings. The recoverer
   would reconstruct a different anomaly than the one in the pack.

6. OMISSION
   The narrative does NOT cite at least one fired_signal — i.e. the
   anomaly's primary identification is left out of the explanation.

These six categories cover the failure modes Berti et al. (2026)
catalogue plus the entity-confusion case specific to traced
process-mining output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.lifecycle_graph import STATES

# ---------------------------------------------------------------------------
# Patterns (per-category)
# ---------------------------------------------------------------------------

# Causal phrases: not exhaustive, drawn from an internal design review
# and Berti 2026 cause-attribution failure mode.
_CAUSAL_PHRASES = (
    " led to ",
    " caused ",
    " resulted in ",
    " because of the ",
    " due to the ",
    " made the trade ",
    " forces the system ",
    " requires that ",
)

# Allow-list of UPPER tokens that appear in legitimate prose without
# being state names (acronyms, etc.).
_SAFE_UPPER_TOKENS = {
    "LLM", "CDM", "ISDA", "JSON", "API", "SQL", "URL", "IT",
    "SSI", "SLA", "FX", "BPI", "PM", "CCP", "MIFID", "EMIR",
    "DTCC", "DORA", "T2S", "ISO", "MTok", "GPT",
    # Identifier tokens that can appear in trade_id and class labels.
    "TRD", "IMPOSSIBLE", "PINGPONG", "SKIPPED_STEP",
    "UNLIKELY_BRANCH", "NORMAL", "RECOVERY",
}

# Regulatory / operational concepts that should always be paired with a
# citation if mentioned. We treat any of these as a flag for
# unfaithful_elaboration if the sentence containing them lacks
# parenthetical citations.
_REGULATORY_CONCEPTS = (
    "mifid",
    "emir",
    "dora",
    "iso 20022",
    "t+1 settlement",
    "t+2 settlement",
    "ccp clearing",
    "novation",
    "bilateral clearing",
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class HallucinationDiagnosis:
    causal_prescription: list[str]
    invented_state: list[str]
    invented_flag: list[str]
    unfaithful_elaboration: list[str]
    entity_confusion: list[str]
    omission: list[str]

    def total_count(self) -> int:
        return sum(
            len(getattr(self, f))
            for f in (
                "causal_prescription",
                "invented_state",
                "invented_flag",
                "unfaithful_elaboration",
                "entity_confusion",
                "omission",
            )
        )

    def is_clean(self) -> bool:
        return self.total_count() == 0

    def to_dict(self) -> dict:
        return {
            "causal_prescription": list(self.causal_prescription),
            "invented_state": list(self.invented_state),
            "invented_flag": list(self.invented_flag),
            "unfaithful_elaboration": list(self.unfaithful_elaboration),
            "entity_confusion": list(self.entity_confusion),
            "omission": list(self.omission),
            "total": self.total_count(),
            "ok": self.is_clean(),
        }


# ---------------------------------------------------------------------------
# Per-category checks
# ---------------------------------------------------------------------------

VALID_FLAG_NAMES = frozenset({
    "PT-impossible", "PT-surprise", "PT-rank",
    "SQ-low-likelihood", "SQ-edge-repeat",
})


def _split_sentences(text: str) -> list[str]:
    # Crude sentence split adequate for our short, well-punctuated output.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _check_causal_prescription(text: str) -> list[str]:
    lc = " " + text.lower() + " "
    return [p.strip() for p in _CAUSAL_PHRASES if p in lc]


def _check_invented_state(text: str) -> list[str]:
    valid = set(STATES) | _SAFE_UPPER_TOKENS
    tokens = set(re.findall(r"\b[A-Z][A-Z_]{2,}\b", text))
    invented = sorted(t for t in tokens if t not in valid and not t.startswith(("PT", "SQ")))
    return invented


def _check_invented_flag(text: str) -> list[str]:
    flags_in_text = set(re.findall(r"\b(?:PT|SQ)-[a-z\-]+\b", text))
    return sorted(f for f in flags_in_text if f not in VALID_FLAG_NAMES)


def _check_unfaithful_elaboration(text: str, pack: dict) -> list[str]:
    """A sentence mentioning a regulatory / operational concept must
    contain at least one parenthetical citation. Otherwise the concept
    is unfaithful elaboration.

    Pack is consulted to allow concepts that DO appear under
    business_context.regulatory_relevance — those are legitimate.
    """
    pack_concepts: set[str] = set()
    bc = pack.get("business_context", {})
    for entry in bc.get("regulatory_relevance", []):
        pack_concepts.add(entry.lower())
    issues: list[str] = []
    for sentence in _split_sentences(text):
        lc = sentence.lower()
        for concept in _REGULATORY_CONCEPTS:
            if concept not in lc:
                continue
            # If the concept matches a pack-listed regulatory entry,
            # treat the citation as implicit and skip.
            if any(concept in pc for pc in pack_concepts):
                continue
            if "(" not in sentence:
                issues.append(concept)
                break  # one finding per sentence is enough
    return issues


def _check_entity_confusion(text: str, pack: dict) -> list[str]:
    """Find UPPER_SNAKE tokens in the text that ARE valid lifecycle
    states, do NOT appear in pack.trace, AND are not in the legitimate
    alternative-state pool the pack supplies.

    Legitimate alternative-state pool:
      - transition_findings[*].expected_top3[*].state  (counterfactuals)
      - business_context.* (regulatory / phase-relevant states)

    Anything in the text outside these and outside the trace IS the
    LLM inventing a state attribution.
    """
    trace_states = set(pack.get("trace", []))
    # Counterfactuals from expected_top3
    expected_states: set[str] = set()
    for tf in pack.get("transition_findings", []):
        for e in tf.get("expected_top3", []):
            if isinstance(e, dict) and "state" in e:
                expected_states.add(e["state"])
    # Business-context-mentioned states (downstream_dependencies_blocked etc.)
    bc = pack.get("business_context", {})
    bc_text = json.dumps(bc) if bc else ""
    bc_states = set(re.findall(r"\b[A-Z][A-Z_]{2,}\b", bc_text))
    legitimate = trace_states | expected_states | bc_states

    valid_states = set(STATES)
    tokens = set(re.findall(r"\b[A-Z][A-Z_]{2,}\b", text))
    confused = sorted(
        t for t in tokens
        if t in valid_states and t not in legitimate
    )
    return confused


def _check_omission(text: str, pack: dict) -> list[str]:
    """Every fired_signal must be cited at least once in the text."""
    fired = list(pack.get("verdict", {}).get("fired_signals", []))
    missing = [f for f in fired if f not in text]
    return missing


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def diagnose(text: str, pack: dict) -> HallucinationDiagnosis:
    """Apply all six checks and return a structured diagnosis."""
    return HallucinationDiagnosis(
        causal_prescription=_check_causal_prescription(text),
        invented_state=_check_invented_state(text),
        invented_flag=_check_invented_flag(text),
        unfaithful_elaboration=_check_unfaithful_elaboration(text, pack),
        entity_confusion=_check_entity_confusion(text, pack),
        omission=_check_omission(text, pack),
    )
