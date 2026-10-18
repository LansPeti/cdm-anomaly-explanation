"""Specification-Recovery Faithfulness framework.

Implements the round-trip evaluation defined in
experiments/rag_corpus/12_specification_recovery_faithfulness.md:

    AnomalySpecifier  →  TraceMaterializer  →  Pipeline  →  AnomalyRecoverer
        (LLM-A)          (deterministic)      (under test)      (LLM-B)
            │                                                       │
            └────────────────── SemanticComparator ─────────────────┘
                              (ρ ∈ [0,1])

This module is provider-agnostic at interface level. Concrete
implementations use the OpenAI client (the same one the rest of the
project already depends on); future work can swap in alternative
backends (Claude, Llama, etc.) for the cross-LLM ablation.
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.data_generator import (
    ANOMALY_INJECTORS,
    generate_normal_trace,
)

load_dotenv()


def _is_gpt5_family(model: str) -> bool:
    """Detect models that require `max_completion_tokens` and reject `temperature`."""
    return model.startswith("gpt-5") or model.startswith(("o1", "o3", "o4"))


def _create_chat(client, *, model: str, messages: list, temperature: float,
                 response_format: dict | None = None) -> object:
    """Provider-portable chat-completions call.

    Newer model families (gpt-5.x, reasoning models o1/o3/o4) reject
    `temperature` and use `max_completion_tokens`. Older families take
    the conventional `temperature`. We detect by model name.
    """
    kwargs: dict = {"model": model, "messages": messages}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if not _is_gpt5_family(model):
        kwargs["temperature"] = temperature
    return client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Specification dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnomalySpecification:
    """Natural-language description of one anomaly to inject.

    Attributes
    ----------
    nl_description
        A 1-3 sentence English description of the anomaly.
    class_hint
        Optional structured hint (one of the four known classes), or
        None for a free-form / out-of-distribution specification.
    metadata
        Free-form bag of structured details the materializer may use
        (e.g. ``{"repeat_count": 5, "loop_states": ["ENRICHED",
        "UNMATCHED_ERROR"]}`` for a pingpong-style spec).
    """
    nl_description: str
    class_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "nl_description": self.nl_description,
            "class_hint": self.class_hint,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# 1. AnomalySpecifier (LLM-A)
# ---------------------------------------------------------------------------

_SPECIFIER_SYSTEM = """You are a quality-assurance auditor for a
trade-lifecycle monitoring system. Your task is to write natural-language
specifications of anomalies that should appear in trade traces.

The lifecycle has these states, in canonical happy-path order:
NEW → EXECUTED → CAPTURED → ENRICHED → MATCHED → CLEARED → SETTLED.
Error / repair states: UNMATCHED_ERROR, FAILED_SETTLEMENT, AMENDED.
Terminal failures: REJECTED, CANCELLED.

You will be asked to specify anomalies of one of four classes. The
"metadata" field MUST follow the per-class schema below — a downstream
deterministic materializer reads exactly these keys:

- skipped_step:
    metadata = {"removed_state": <one state name from the canonical path
                                   that will be omitted>}
    Choose from: EXECUTED, CAPTURED, ENRICHED, MATCHED, CLEARED.

- impossible:
    metadata = {"after": <state>, "injected": <state that cannot legally
                                                follow `after`>}
    Example: {"after": "MATCHED", "injected": "EXECUTED"}.

- pingpong:
    metadata = {"n_cycles": <int between 5 and 10>}
    The loop is fixed at ENRICHED → UNMATCHED_ERROR → AMENDED → ENRICHED.

- unlikely_branch:
    metadata = {"after": <one of NEW or EXECUTED>}
    A REJECTED state will be appended after `after`.

Output STRICTLY valid JSON with three keys:
  - "nl_description": 1-3 sentences describing the anomaly in plain English,
    without revealing the class label by name.
  - "class_hint": one of {skipped_step, impossible, pingpong, unlikely_branch}.
  - "metadata": dict matching the per-class schema above EXACTLY.

Do NOT include any prose outside the JSON object."""


class AnomalySpecifier:
    """Generates natural-language anomaly specifications via an LLM.

    Independent from the explanation pipeline — uses its own prompt and
    a separate API call.
    """

    def __init__(self,
                 model: str = "gpt-4o-mini-2024-07-18",
                 client: OpenAI | None = None,
                 temperature: float = 0.7):
        self.client = client or OpenAI()
        self.model = model
        self.temperature = temperature

    def specify(self, class_hint: str, rng: random.Random | None = None) -> AnomalySpecification:
        """Produce one specification with the given class hint.

        ``temperature`` is intentionally higher than the explainer's
        (0.7 vs 0.2) — we want diversity across runs rather than
        consistency.
        """
        seed_phrase = ""
        if rng is not None:
            seed_phrase = f"\nDiversity hint: {rng.randint(0, 1_000_000)}"
        user = (
            f"Generate one anomaly specification of class '{class_hint}'. "
            f"The specification should be specific enough that a "
            f"deterministic implementation could materialise it as a "
            f"concrete trace.{seed_phrase}"
        )
        response = _create_chat(
            self.client, model=self.model,
            messages=[
                {"role": "system", "content": _SPECIFIER_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        return AnomalySpecification(
            nl_description=payload.get("nl_description", "").strip(),
            class_hint=payload.get("class_hint", class_hint),
            metadata=payload.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# 2. TraceMaterializer (deterministic)
# ---------------------------------------------------------------------------

class TraceMaterializer:
    """Convert an ``AnomalySpecification`` into a concrete trace.

    Spec-aware: each known class consumes specific keys from
    ``spec.metadata`` deterministically (see ``_SPECIFIER_SYSTEM`` for
    the per-class schema). When the metadata keys are missing or
    invalid, falls back to the random injectors in
    ``data_generator.ANOMALY_INJECTORS`` so the framework remains
    robust to specifier-prompt drift.

    For free-form specs (class_hint is None) the caller must provide
    a fallback hook.
    """

    def __init__(self,
                 free_form_fallback: Callable[[AnomalySpecification, random.Random], list[str]] | None = None):
        self.free_form_fallback = free_form_fallback

    def materialize(self, spec: AnomalySpecification, rng: random.Random,
                    max_attempts: int = 50) -> list[str]:
        # 1. Try the spec-aware deterministic path.
        if spec.class_hint == "skipped_step":
            t = self._inject_skipped_step(spec, rng)
            if t is not None:
                return t
        elif spec.class_hint == "impossible":
            t = self._inject_impossible(spec, rng)
            if t is not None:
                return t
        elif spec.class_hint == "pingpong":
            t = self._inject_pingpong(spec, rng)
            if t is not None:
                return t
        elif spec.class_hint == "unlikely_branch":
            t = self._inject_unlikely_branch(spec, rng)
            if t is not None:
                return t

        # 2. Fall back to the random injectors (back-compat).
        if spec.class_hint in ANOMALY_INJECTORS:
            injector = ANOMALY_INJECTORS[spec.class_hint]
            for _ in range(max_attempts):
                base = generate_normal_trace(rng)
                trace, _meta = injector(base, rng)
                if trace is not None:
                    return trace
            raise RuntimeError(
                f"Could not materialize trace for class_hint={spec.class_hint!r} "
                f"after {max_attempts} attempts."
            )

        # 3. Free-form (no class hint): defer to caller.
        if self.free_form_fallback is not None:
            return self.free_form_fallback(spec, rng)
        raise NotImplementedError(
            "Free-form anomaly specifications require a custom fallback. "
            "Pass free_form_fallback=... to TraceMaterializer or supply a "
            "class_hint in {skipped_step, impossible, pingpong, unlikely_branch}."
        )

    # ---- Spec-aware injectors ----------------------------------------

    @staticmethod
    def _inject_skipped_step(spec: AnomalySpecification,
                             rng: random.Random) -> list[str] | None:
        """Synthesise: NEW → ... → SETTLED with `removed_state` omitted.

        Reads spec.metadata["removed_state"]. Builds the canonical happy
        path and removes the named state. The previous→next edge becomes
        an illegal transition under the lifecycle graph, which the
        detector flags as PT-impossible.
        """
        removed = spec.metadata.get("removed_state")
        canonical = ["NEW", "EXECUTED", "CAPTURED", "ENRICHED",
                     "MATCHED", "CLEARED", "SETTLED"]
        if removed not in canonical or removed in ("NEW", "SETTLED"):
            return None
        return [s for s in canonical if s != removed]

    @staticmethod
    def _inject_impossible(spec: AnomalySpecification,
                           rng: random.Random) -> list[str] | None:
        """Synthesise: canonical path with a single illegal jump.

        Reads spec.metadata["after"] and ["injected"]. Inserts `injected`
        at the position immediately following `after` in a canonical-
        prefix trace.
        """
        after = spec.metadata.get("after")
        injected = spec.metadata.get("injected")
        canonical = ["NEW", "EXECUTED", "CAPTURED", "ENRICHED",
                     "MATCHED", "CLEARED", "SETTLED"]
        if after not in canonical or injected not in (
            "NEW", "EXECUTED", "CAPTURED", "ENRICHED",
            "MATCHED", "CLEARED", "SETTLED",
            "UNMATCHED_ERROR", "FAILED_SETTLEMENT", "AMENDED",
            "REJECTED", "CANCELLED",
        ):
            return None
        idx = canonical.index(after)
        return canonical[: idx + 1] + [injected]

    @staticmethod
    def _inject_pingpong(spec: AnomalySpecification,
                         rng: random.Random) -> list[str] | None:
        """Synthesise: pingpong loop with the requested cycle count.

        Reads spec.metadata["n_cycles"] (clamped to [5, 10]). The loop
        is fixed: ENRICHED → UNMATCHED_ERROR → AMENDED → ENRICHED.
        """
        try:
            n = int(spec.metadata.get("n_cycles", 5))
        except (TypeError, ValueError):
            return None
        n = max(5, min(10, n))
        prefix = ["NEW", "EXECUTED", "CAPTURED", "ENRICHED"]
        cycle = ["UNMATCHED_ERROR", "AMENDED", "ENRICHED"]
        suffix = ["MATCHED", "CLEARED", "SETTLED"]
        return prefix + cycle * n + suffix

    @staticmethod
    def _inject_unlikely_branch(spec: AnomalySpecification,
                                rng: random.Random) -> list[str] | None:
        """Synthesise: short trace ending in REJECTED after the named state.

        Reads spec.metadata["after"] (NEW or EXECUTED).
        """
        after = spec.metadata.get("after")
        if after == "NEW":
            return ["NEW", "REJECTED"]
        if after == "EXECUTED":
            return ["NEW", "EXECUTED", "REJECTED"]
        return None


# ---------------------------------------------------------------------------
# 3. AnomalyRecoverer (LLM-B)
# ---------------------------------------------------------------------------

_RECOVERER_SYSTEM = """You are a quality-assurance auditor reading a
trade-lifecycle anomaly explanation. Your task is to reconstruct the
ORIGINAL natural-language anomaly specification that, if injected into
a normal trace, would have produced the explanation you are reading.

You have NO access to the original specification. You must infer it
purely from the explanation text.

The lifecycle has these states: NEW, EXECUTED, CAPTURED, ENRICHED,
MATCHED, CLEARED, SETTLED, UNMATCHED_ERROR, FAILED_SETTLEMENT,
AMENDED, REJECTED, CANCELLED.

Possible anomaly classes:
- skipped_step: an intermediate state is missing.
- impossible: a transition that cannot legally occur.
- pingpong: an edge repeats far more often than normal.
- unlikely_branch: a structurally legal but very rare path is taken.

Output STRICTLY valid JSON with three keys:
  - "nl_description": 1-3 sentences in plain English that re-state the
    anomaly. Do not paraphrase the explanation; describe the underlying
    anomaly itself.
  - "class_hint": one of the four classes above (your best guess).
  - "metadata": dict of structured details inferred from the explanation
    (e.g. specific states, repeat counts).

Do NOT include any prose outside the JSON object."""


class AnomalyRecoverer:
    """Recovers an ``AnomalySpecification`` from an explanation."""

    def __init__(self,
                 model: str = "gpt-4o-mini-2024-07-18",
                 client: OpenAI | None = None,
                 temperature: float = 0.2):
        self.client = client or OpenAI()
        self.model = model
        self.temperature = temperature

    def recover(self, explanation_text: str) -> AnomalySpecification:
        response = _create_chat(
            self.client, model=self.model,
            messages=[
                {"role": "system", "content": _RECOVERER_SYSTEM},
                {"role": "user", "content": explanation_text},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        return AnomalySpecification(
            nl_description=payload.get("nl_description", "").strip(),
            class_hint=payload.get("class_hint"),
            metadata=payload.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# 4. SemanticComparator (σ functions)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an impartial judge comparing two
natural-language anomaly specifications. Each specification describes
one anomaly that should appear in a trade-lifecycle trace.

Score on a 0-100 scale how well the second specification preserves the
SEMANTIC CONTENT of the first. The two are equivalent (score 100) if
they describe the same underlying anomaly even with different wording.
They are unrelated (score 0) if a competent operator would treat them
as different anomalies.

Output STRICTLY valid JSON with two keys:
  - "score": integer 0-100.
  - "reasoning": one sentence explaining the score.

Do NOT include any prose outside the JSON object."""


_PLAUSIBILITY_SYSTEM = """You are an experienced post-trade operations
analyst. You will be shown a single LLM-generated anomaly explanation
intended for an operations analyst.

Score on a 0-100 scale how PLAUSIBLE the explanation is — i.e. how
much it reads like a competent operator's narrative an experienced
analyst would find credible. Plausibility is independent of whether
the explanation is correct: a clear, well-structured, technically
appropriate-sounding narrative scores high even if the underlying
anomaly is misidentified.

Output STRICTLY valid JSON:
  - "score": integer 0-100.
  - "reasoning": one sentence.

Do NOT include any prose outside the JSON object."""


_SENSEMAKING_SYSTEM = """You are an experienced post-trade operations
analyst. You will be shown an anomaly explanation. Score how well it
ENABLES SENSEMAKING — that is, after reading it, would an analyst
have a clear, actionable mental model of (a) what was unusual,
(b) where in the lifecycle it occurred, and (c) what to investigate
next?

Sensemaking differs from plausibility (does it sound right?) and
faithfulness (does it match the underlying truth?): sensemaking is
about whether the reader leaves the explanation more capable of
acting on it.

Output STRICTLY valid JSON:
  - "score": integer 0-100.
  - "reasoning": one sentence.

Do NOT include any prose outside the JSON object."""


class SemanticComparator:
    """Two-headed similarity scorer.

    - LLM-as-judge (``judge_score``): semantic equivalence with reasoning.
    - Class-hint match (``class_match``): hard 0/1 for the categorical hint.
    - Class-and-judge composite (``rho``): the headline metric ρ in [0,1].

    Embedding-based similarity is left as future work to keep the first
    iteration provider-light (it does not require the embeddings API).
    """

    def __init__(self,
                 model: str = "gpt-4o-mini-2024-07-18",
                 client: OpenAI | None = None,
                 temperature: float = 0.0):
        self.client = client or OpenAI()
        self.model = model
        self.temperature = temperature

    def judge_score(self, spec_a: AnomalySpecification,
                    spec_b: AnomalySpecification) -> tuple[float, str]:
        """LLM-judged 0-100 semantic similarity, normalised to [0, 1].

        Returns (score_in_zero_one, reasoning_string).
        """
        prompt = (
            "Specification A:\n" + spec_a.nl_description +
            "\n\nSpecification B:\n" + spec_b.nl_description
        )
        response = _create_chat(
            self.client, model=self.model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        score = float(payload.get("score", 0)) / 100.0
        reasoning = payload.get("reasoning", "")
        return max(0.0, min(1.0, score)), reasoning

    @staticmethod
    def class_match(spec_a: AnomalySpecification,
                    spec_b: AnomalySpecification) -> int:
        return int(spec_a.class_hint == spec_b.class_hint)

    def embedding_similarity(self, spec_a: AnomalySpecification,
                             spec_b: AnomalySpecification,
                             embedding_model: str = "text-embedding-3-small"
                             ) -> float:
        """Cosine similarity between OpenAI embeddings of the two NL
        descriptions. Provides a non-LLM-judge signal that is harder
        to game by stylistic mimicry.
        """
        import math

        def _embed(text: str) -> list[float]:
            response = self.client.embeddings.create(  # type: ignore[attr-defined]
                model=embedding_model, input=text,
            )
            return list(response.data[0].embedding)

        a = _embed(spec_a.nl_description)
        b = _embed(spec_b.nl_description)
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def plausibility_score(self, explanation_text: str) -> tuple[float, str]:
        """NEURON-style plausibility: would the narrative read as
        competent to an experienced operator? Independent of truth.
        """
        response = _create_chat(
            self.client, model=self.model,
            messages=[
                {"role": "system", "content": _PLAUSIBILITY_SYSTEM},
                {"role": "user", "content": explanation_text},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        score = float(payload.get("score", 0)) / 100.0
        return max(0.0, min(1.0, score)), payload.get("reasoning", "")

    def sensemaking_score(self, explanation_text: str) -> tuple[float, str]:
        """NEURON-style sensemaking: after reading, does the analyst
        know what was unusual, where, and what to do next?
        """
        response = _create_chat(
            self.client, model=self.model,
            messages=[
                {"role": "system", "content": _SENSEMAKING_SYSTEM},
                {"role": "user", "content": explanation_text},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        score = float(payload.get("score", 0)) / 100.0
        return max(0.0, min(1.0, score)), payload.get("reasoning", "")

    def rho(self, spec_a: AnomalySpecification,
            spec_b: AnomalySpecification) -> dict:
        """Composite recovery score for one (spec_a, spec_b) pair.

        Returns a dict with all components so the caller can report
        each separately and recompute composites if desired.
        """
        judge, reasoning = self.judge_score(spec_a, spec_b)
        match = self.class_match(spec_a, spec_b)
        # Composite ρ: 0.7·judge + 0.3·class_match — chosen so that perfect
        # class match alone cannot produce a passing score, and a perfect
        # semantic match without the structured hint still scores 0.7.
        composite = 0.7 * judge + 0.3 * match
        return {
            "judge_score": judge,
            "judge_reasoning": reasoning,
            "class_match": match,
            "rho": composite,
        }


# ---------------------------------------------------------------------------
# 5. End-to-end orchestrator
# ---------------------------------------------------------------------------

@dataclass
class RecoveryRunResult:
    spec: AnomalySpecification
    trace: list[str]
    explanation: str
    recovered: AnomalySpecification
    scores: dict
    fired_signals: list[str]


class RecoveryFaithfulnessRunner:
    """Wires the four stages together for one or many evaluations.

    Decoupled from the pipeline implementation: takes a callable
    ``pipeline`` of type ``trace -> (explanation_text, fired_signals)``.
    The default integration is in ``experiments/recovery_faithfulness.py``.
    """

    def __init__(self,
                 specifier: AnomalySpecifier,
                 materializer: TraceMaterializer,
                 pipeline: Callable[[list[str]], tuple[str, list[str]]],
                 recoverer: AnomalyRecoverer,
                 comparator: SemanticComparator):
        self.specifier = specifier
        self.materializer = materializer
        self.pipeline = pipeline
        self.recoverer = recoverer
        self.comparator = comparator

    def run_one(self, class_hint: str, rng: random.Random) -> RecoveryRunResult:
        spec = self.specifier.specify(class_hint, rng=rng)
        trace = self.materializer.materialize(spec, rng=rng)
        explanation_text, fired_signals = self.pipeline(trace)
        recovered = self.recoverer.recover(explanation_text)
        scores = self.comparator.rho(spec, recovered)
        return RecoveryRunResult(
            spec=spec,
            trace=trace,
            explanation=explanation_text,
            recovered=recovered,
            scores=scores,
            fired_signals=list(fired_signals),
        )

    def run_many(self, classes: list[str], n_per_class: int,
                 seed: int = 42) -> list[RecoveryRunResult]:
        rng = random.Random(seed)
        results: list[RecoveryRunResult] = []
        for cls in classes:
            for _ in range(n_per_class):
                results.append(self.run_one(cls, rng))
        return results
