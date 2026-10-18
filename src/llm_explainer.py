"""
llm_explainer.py — LLM narration layer.

The LLM converts a JSON evidence pack into a four-section markdown
explanation for an auditor. It is NOT a detector, NOT a judge, NOT a
re-ranker. It is a strict translator from numeric evidence into prose,
with mandatory field-level citations.

WHY NO TOOLS, NO FINE-TUNING
============================

  - No tools: every fact the LLM might want to look up (top-3 expected
    continuations, training counts, fired-flag interpretations) is already
    in the evidence pack. Tools would add latency and a hallucination
    surface for no gain.
  - No fine-tuning: the "pattern" the model needs to learn is "translate
    these field names to prose." That's a few-shot prompting problem,
    not a fine-tuning problem. Berti et al. (arXiv 2407.02310) showed
    zero-shot LLMs FAIL at trace anomaly *judgment*; we mitigate by
    forbidding judgment and confining the LLM to narration.

OUTPUT CONTRACT
===============

The output is markdown with EXACTLY four sections, in order:

    ## 1. Observed Trace
    ## 2. Why It Is Flagged
    ## 3. What Normal Would Look Like
    ## 4. Recommended Validation Step

For non-anomalous packs (verdict.is_anomalous = false), the output is the
single literal line:

    Normal trace, no findings.

DEPENDENCIES
============

  - openai>=1.50 (chat completions API)
  - python-dotenv (loads OPENAI_API_KEY from .env)

CONFIGURATION
=============

  - Model name: configurable via constructor; default
    "gpt-4o-mini-2024-07-18" (pinned). The undated alias "gpt-4o-mini"
    receives silent provider updates and would make explanations for
    the same evidence pack drift across model releases, breaking the
    audit reproducibility claim under MiFID II Article 25 / DORA.
  - Temperature: 0.2 (low — we want consistent narration).
  - max_tokens:  900 (typical pack → 300–500 tokens; cap protects budget).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.lifecycle_graph import STATES as _LIFECYCLE_STATES
from src.lifecycle_graph import graph_summary

load_dotenv()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_GRAPH = graph_summary()
_HAPPY_PATH = " -> ".join(_GRAPH["happy_path"])

SYSTEM_PROMPT = f"""You are an ISDA CDM trade-lifecycle audit narrator.

You will receive a JSON evidence pack containing a trade trace and
detector findings. The verdict has ALREADY been made by a probabilistic
detector. Your job is NOT to judge whether the trade is anomalous — your
job is ONLY to translate the structured evidence into a human-readable
explanation an auditor can act on.

HARD RULES (any violation makes the output unusable):

1. EVERY claim must cite a specific field from the evidence pack using
   inline parentheses. Examples:
     "(verdict.fired_signals)"
     "(transition_findings[0].p_observed)"
     "(sequence_level.norm_ll_z_score)"
2. NEVER invent state names. Use only states that appear in the `trace`
   field of the pack.
3. NEVER claim causation. Use: "consistent with", "suggests",
   "indicates a pattern of".
4. If pack.verdict.is_anomalous is false, output EXACTLY this and nothing
   else:
       Normal trace, no findings.
5. Output is markdown with these four sections, in order, no other
   sections, no preamble:

       ## 1. Observed Trace
       ## 2. Why It Is Flagged
       ## 3. What Normal Would Look Like
       ## 4. Recommended Validation Step

6. Section 4 MUST be a verbatim bulleted enumeration of every item in
   `pack.validation_requirements` (do not paraphrase, do not add items,
   do not draw inferences about causes). Cite the field name once at the
   end: "(validation_requirements)". If the array is empty, output the
   single line:
       No domain-defined validation steps apply for this signal set
       (validation_requirements).
   This is the narrator-not-judge boundary: Section 4 is declarative
   pack content, never LLM-generated prescription.

7. When `pack.business_context` is present, weave its phase, purpose,
   owner, regulatory relevance, common failure causes, and downstream
   dependencies into Section 2 ("Why It Is Flagged") and Section 3
   ("What Normal Would Look Like") wherever they add operational or
   compliance value. Cite each by full path, e.g.
   "(business_context.phase_purpose)",
   "(business_context.typical_operational_owner)",
   "(business_context.regulatory_relevance)". Do NOT invent any
   business content beyond what these fields contain. If
   `pack.business_context.retrieved_excerpts` is present (RAG mode),
   you MAY quote a specific excerpt only with its `source` attribution,
   e.g. "(business_context.retrieved_excerpts[0].source)".

8. When `pack.model_context` is present, you MAY use
   `model_context.learned_transition_top_k` to describe what the fitted
   normal-transition matrix learned as typical next states. Use it only
   as explanatory context, never to override `verdict.is_anomalous`.
   Cite it by full path, e.g.
   "(model_context.learned_transition_top_k.CAPTURED[0])".

DOMAIN GLOSSARY (use as reference only, do not infer beyond it):

  - Canonical happy path: {_HAPPY_PATH}
  - Error states: UNMATCHED_ERROR (matching failure),
                  FAILED_SETTLEMENT (settlement failure).
  - Repair: AMENDED -> ENRICHED (re-enter normal flow).
  - Terminal: SETTLED, REJECTED, CANCELLED.

FLAG SEMANTICS:
  - PT-impossible / PT-surprise: a single transition that should not
    have happened.
  - PT-rank: the observed transition was not among the top expected
    continuations.
  - SQ-low-likelihood: the trace as a whole has unusually low
    probability.
  - SQ-edge-repeat: an edge repeats far more often than is normal,
    suggesting a cycle or a stuck loop.
"""


# ---------------------------------------------------------------------------
# Few-shot examples (compact — the prompt's other rules carry most of the work)
# ---------------------------------------------------------------------------

_FEW_SHOT_INPUT_1 = {
    "trade_id": "TRD-EXAMPLE-IMPOSSIBLE",
    "trace": ["NEW", "EXECUTED", "CAPTURED", "MATCHED", "CLEARED", "SETTLED"],
    "trace_length": 6,
    "edge_count_distribution": {
        "NEW->EXECUTED": 1, "EXECUTED->CAPTURED": 1,
        "CAPTURED->MATCHED": 1, "MATCHED->CLEARED": 1,
        "CLEARED->SETTLED": 1,
    },
    "sequence_level": {
        "norm_log_likelihood": -1.87, "norm_ll_z_score": -2.45,
        "max_edge_repeat": 1, "most_repeated_edge": ["NEW", "EXECUTED"],
        "fired": [{"name": "SQ-low-likelihood",
                   "interpretation": "The trace's overall ..."}],
    },
    "transition_findings": [{
        "step": 2,
        "observed_transition": {"from": "CAPTURED", "to": "MATCHED"},
        "p_observed": 0.0001,
        "surprise_nats": 9.17,
        "rank_among_candidates": 7,
        "n_candidates": 12,
        "gap_from_top1": 0.956,
        "expected_top3": [
            {"state": "ENRICHED", "probability": 0.9561},
            {"state": "CANCELLED", "probability": 0.0428},
            {"state": "EXECUTED", "probability": 0.0001},
        ],
        "training_count": 0,
        "fired_flags": ["PT-impossible", "PT-surprise", "PT-rank"],
        "fired_interpretations": ["...", "...", "..."],
    }],
    "verdict": {
        "is_anomalous": True,
        "fired_signals": ["PT-impossible", "PT-surprise", "PT-rank",
                          "SQ-low-likelihood"],
    },
    "validation_requirements": [
        "Verify whether the source-system event log records an intermediate "
        "state that was discarded by the audit pipeline before reaching the "
        "lifecycle store.",
        "Confirm that the recorded state names are not the result of a "
        "renaming or schema mismatch with the reference state machine.",
        "Confirm whether the recorded transition reflects a legitimate but "
        "rare workflow variant by checking the source-system audit trail.",
        "Cross-check whether the higher-ranked expected continuations were "
        "considered before the observed one was committed.",
        "Investigate whether the trace path is a known but rare workflow "
        "variant absent from the training distribution rather than a "
        "control failure.",
    ],
}

_FEW_SHOT_OUTPUT_1 = """## 1. Observed Trace
The trade followed: NEW -> EXECUTED -> CAPTURED -> MATCHED -> CLEARED -> SETTLED (trace).
At step 2, CAPTURED -> MATCHED was recorded
(transition_findings[0].observed_transition).

## 2. Why It Is Flagged
- The transition CAPTURED -> MATCHED was never observed in normal training data
  (transition_findings[0].training_count = 0, fired_flags includes
  PT-impossible).
- Its model probability is 0.0001
  (transition_findings[0].p_observed), consistent with a structurally
  invalid step.
- The trace's overall log-likelihood is 2.45 standard deviations below
  the normal distribution (sequence_level.norm_ll_z_score),
  reinforcing the per-transition signal.

## 3. What Normal Would Look Like
The model's expected continuation from CAPTURED was ENRICHED with
probability 0.96 (transition_findings[0].expected_top3[0]).
A normal trace would have inserted ENRICHED between CAPTURED and MATCHED.

## 4. Recommended Validation Step
- Verify whether the source-system event log records an intermediate state that was discarded by the audit pipeline before reaching the lifecycle store.
- Confirm that the recorded state names are not the result of a renaming or schema mismatch with the reference state machine.
- Confirm whether the recorded transition reflects a legitimate but rare workflow variant by checking the source-system audit trail.
- Cross-check whether the higher-ranked expected continuations were considered before the observed one was committed.
- Investigate whether the trace path is a known but rare workflow variant absent from the training distribution rather than a control failure.

(validation_requirements)"""


# ---------------------------------------------------------------------------
# Few-shot example #2 — sequence-only anomaly (pingpong / cyclic loop)
#
# Why this exists: the empty `transition_findings` case is structurally
# different from the per-transition case above. Without a worked example,
# the model has no in-context demonstration of how to narrate a trace
# whose anomaly is purely sequence-level (citing sequence_level.*,
# edge_count_distribution, and optional model_context, never
# transition_findings[*]).
# ---------------------------------------------------------------------------

_FEW_SHOT_INPUT_2 = {
    "trade_id": "TRD-EXAMPLE-PINGPONG",
    "trace": [
        "NEW", "EXECUTED", "CAPTURED",
        "ENRICHED", "UNMATCHED_ERROR", "AMENDED",
        "ENRICHED", "UNMATCHED_ERROR", "AMENDED",
        "ENRICHED", "UNMATCHED_ERROR", "AMENDED",
        "ENRICHED", "UNMATCHED_ERROR", "AMENDED",
        "ENRICHED", "MATCHED", "CLEARED", "SETTLED",
    ],
    "trace_length": 19,
    "edge_count_distribution": {
        "NEW->EXECUTED": 1, "EXECUTED->CAPTURED": 1,
        "CAPTURED->ENRICHED": 1,
        "ENRICHED->UNMATCHED_ERROR": 4, "UNMATCHED_ERROR->AMENDED": 4,
        "AMENDED->ENRICHED": 4,
        "ENRICHED->MATCHED": 1, "MATCHED->CLEARED": 1, "CLEARED->SETTLED": 1,
    },
    "sequence_level": {
        "norm_log_likelihood": -0.51, "norm_ll_z_score": -0.35,
        "max_edge_repeat": 4,
        "most_repeated_edge": ["ENRICHED", "UNMATCHED_ERROR"],
        "fired": [{"name": "SQ-edge-repeat",
                   "interpretation": "An edge repeats far more than is normal..."}],
    },
    "transition_findings": [],
    "model_context": {
        "source": "fitted_normal_transition_matrix",
        "smoothing": {"method": "laplace", "alpha": 0.5},
        "top_k": 3,
        "learned_transition_top_k": {
            "ENRICHED": [
                {"to_state": "MATCHED", "probability": 0.85,
                 "training_count": 4963, "rank": 1},
                {"to_state": "UNMATCHED_ERROR", "probability": 0.15,
                 "training_count": 881, "rank": 2},
            ],
            "UNMATCHED_ERROR": [
                {"to_state": "AMENDED", "probability": 0.99,
                 "training_count": 881, "rank": 1},
            ],
            "AMENDED": [
                {"to_state": "ENRICHED", "probability": 0.99,
                 "training_count": 904, "rank": 1},
            ],
        },
    },
    "verdict": {
        "is_anomalous": True,
        "fired_signals": ["SQ-edge-repeat"],
    },
    "validation_requirements": [
        "Inspect whether the repeated edge represents a stuck repair loop "
        "that should escalate to manual review under the operational "
        "service-level policy.",
    ],
}

_FEW_SHOT_OUTPUT_2 = """## 1. Observed Trace
The trade followed: NEW -> EXECUTED -> CAPTURED -> ENRICHED -> UNMATCHED_ERROR
-> AMENDED -> ENRICHED -> UNMATCHED_ERROR -> AMENDED -> ENRICHED ->
UNMATCHED_ERROR -> AMENDED -> ENRICHED -> UNMATCHED_ERROR -> AMENDED ->
ENRICHED -> MATCHED -> CLEARED -> SETTLED (trace).
No single transition crossed the per-transition thresholds
(transition_findings is empty).

## 2. Why It Is Flagged
- The edge ENRICHED -> UNMATCHED_ERROR repeats 4 times in this trace
  (sequence_level.max_edge_repeat = 4,
  sequence_level.most_repeated_edge), exceeding the SQ-edge-repeat
  threshold (sequence_level.fired includes SQ-edge-repeat).
- Each individual edge along the loop is a high-probability transition
  on its own, which is why no PT-* flag fired
  (transition_findings is empty,
  model_context.learned_transition_top_k.ENRICHED[1],
  model_context.learned_transition_top_k.UNMATCHED_ERROR[0],
  model_context.learned_transition_top_k.AMENDED[0]).
- The overall normalized log-likelihood is -0.51 with a z-score of -0.35
  (sequence_level.norm_log_likelihood, sequence_level.norm_ll_z_score),
  i.e. the loop is composed of common edges; the anomaly is in the
  repetition, not in any one step.

## 3. What Normal Would Look Like
A normal trace passes through ENRICHED at most once before reaching
MATCHED. Repeated cycling between ENRICHED, UNMATCHED_ERROR, and AMENDED
is consistent with a stuck repair loop rather than the canonical
straight-through path implied by sequence_level.max_edge_repeat = 1
in normal traces.

## 4. Recommended Validation Step
- Inspect whether the repeated edge represents a stuck repair loop that should escalate to manual review under the operational service-level policy.

(validation_requirements)"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    text: str
    model: str
    pack_summary: dict        # echo of verdict + fired_signals for logging
    usage: dict | None     # token usage if returned by the API


# ---------------------------------------------------------------------------
# The explainer
# ---------------------------------------------------------------------------

class AnomalyExplainer:
    """Wraps the OpenAI chat-completions API with the strict prompt above.

    Parameters
    ----------
    model
        Chat model name. Default: "gpt-4o-mini-2024-07-18" (pinned).
        Use a dated revision so the same evidence pack always yields the
        same explanation across model update cycles. The undated alias
        "gpt-4o-mini" should not be used in audit-adjacent contexts.
    api_key
        If None, picked up from $OPENAI_API_KEY (via .env if present).
    temperature
        Default 0.2 — we want consistent narration, not creativity.
    max_tokens
        Default 900 — typical output is 300–500 tokens.
    """

    def __init__(self,
                 model: str = "gpt-4o-mini-2024-07-18",
                 api_key: str | None = None,
                 temperature: float = 0.2,
                 max_tokens: int = 900,
                 cache_dir: Path | None = None):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "No OPENAI_API_KEY found. Set it in .env or pass api_key="
            )
        self.client = OpenAI(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # SHA-256-keyed response cache. When set, the same evidence pack
        # always yields the same explanation — required for audit
        # reproducibility under MiFID II Article 25 / DORA, given that
        # T=0.2 is non-deterministic. Identified as a compliance gap by
        # an internal design review.
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, evidence_pack: dict) -> str:
        """Stable SHA-256 of (model, temperature, pack JSON)."""
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "pack": evidence_pack,
        }
        # A non-default budget must not replay answers cached under the
        # default one (e.g. truncated reasoning-model outputs). Only added
        # when non-default, so default-config keys stay unchanged.
        if self.max_tokens != 900:
            payload["max_tokens"] = self.max_tokens
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def explain(self, evidence_pack: dict) -> Explanation:
        """Generate a markdown explanation for one evidence pack."""
        # Fast path: skip the API for normal traces.
        if not evidence_pack.get("verdict", {}).get("is_anomalous", False):
            return Explanation(
                text="Normal trace, no findings.",
                model=self.model,
                pack_summary={"is_anomalous": False, "fired_signals": []},
                usage=None,
            )

        # Cache lookup
        cache_key = None
        if self.cache_dir is not None:
            cache_key = self._cache_key(evidence_pack)
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                return Explanation(
                    text=cached["text"],
                    model=cached["model"],
                    pack_summary=cached["pack_summary"],
                    usage={"cache_hit": True},
                )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(_FEW_SHOT_INPUT_1, indent=2)},
            {"role": "assistant", "content": _FEW_SHOT_OUTPUT_1},
            {"role": "user", "content": json.dumps(_FEW_SHOT_INPUT_2, indent=2)},
            {"role": "assistant", "content": _FEW_SHOT_OUTPUT_2},
            {"role": "user", "content": json.dumps(evidence_pack, indent=2)},
        ]

        # Newer model families (gpt-5.x and reasoning models o1/o3/...) require
        # `max_completion_tokens` instead of `max_tokens` and reject the
        # `temperature` parameter. Detect by model name and route accordingly.
        is_gpt5_family = self.model.startswith("gpt-5") or self.model.startswith(("o1", "o3", "o4"))
        if is_gpt5_family:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=self.max_tokens,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        choice = response.choices[0]
        explanation = Explanation(
            text=choice.message.content or "",
            model=self.model,
            pack_summary={
                "is_anomalous": evidence_pack["verdict"]["is_anomalous"],
                "fired_signals": evidence_pack["verdict"]["fired_signals"],
            },
            usage=response.usage.model_dump() if response.usage else None,
        )
        # Cache write
        if self.cache_dir is not None and cache_key is not None:
            (self.cache_dir / f"{cache_key}.json").write_text(
                json.dumps({
                    "text": explanation.text,
                    "model": explanation.model,
                    "pack_summary": explanation.pack_summary,
                }, indent=2),
                encoding="utf-8",
            )
        return explanation

    def explain_batch(self, packs: list[dict]) -> list[Explanation]:
        """Sequentially explain a batch (no concurrency to be polite to the API)."""
        return [self.explain(p) for p in packs]


# ---------------------------------------------------------------------------
# Output validators (cheap, deterministic; run before showing to user)
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = (
    "## 1. Observed Trace",
    "## 2. Why It Is Flagged",
    "## 3. What Normal Would Look Like",
    "## 4. Recommended Validation Step",
)


def validate_explanation(text: str, pack: dict) -> dict:
    """Cheap structural checks before showing the explanation to a user.

    Returns a dict with:
      - ok: bool
      - issues: list[str] of problems found

    Useful in CI / eval loops. Does NOT call any LLM.
    """
    issues: list[str] = []

    if not pack.get("verdict", {}).get("is_anomalous", False):
        if text.strip() != "Normal trace, no findings.":
            issues.append("Non-anomalous pack must produce the literal "
                          "'Normal trace, no findings.' string.")
        return {"ok": not issues, "issues": issues}

    # Section presence + order
    pos = -1
    for s in REQUIRED_SECTIONS:
        new_pos = text.find(s)
        if new_pos < 0:
            issues.append(f"Missing required section: {s!r}")
        elif new_pos < pos:
            issues.append(f"Section out of order: {s!r}")
        else:
            pos = new_pos

    # Citation presence (loose check: at least one parenthetical per section)
    if not text.count("("):
        issues.append("No parenthetical citations found.")

    # Trace state vocabulary check: no invented state names.
    # Single source of truth: lifecycle_graph.STATES.
    safe_caps = set(_LIFECYCLE_STATES)
    import re
    tokens = set(re.findall(r"\b[A-Z][A-Z_]{2,}\b", text))
    invented = tokens - safe_caps
    if invented:
        issues.append(f"Possibly invented state-like tokens: {sorted(invented)}")

    return {"ok": not issues, "issues": issues}
