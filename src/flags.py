"""
flags.py — Data contracts between the detector and the explanation pipeline.

These dataclasses define EXACTLY what the model layer (the detector)
must produce. The evidence-pack builder and the LLM explainer both depend
on this schema, so it is the project's most important interface.

DESIGN PRINCIPLE
================

We deliberately separate the detector's *outputs* (these dataclasses) from
the *internal mechanism* of any specific model. The detector can be
implemented with:
  - an empirical Markov matrix
  - multinomial logistic regression
  - a decision tree
  - a random forest
  - XGBoost / LightGBM / CatBoost
  - or any future probabilistic model

…and the rest of the pipeline does not care, as long as the model produces
these structures faithfully.

FLAG NAMING CONVENTION
======================

  - PT-* (per-transition): fired when a single (s_t, s_{t+1}) pair is
                           anomalous in isolation.
  - SQ-* (sequence-level): fired when the WHOLE TRACE is anomalous in a way
                           that no single transition explains (e.g. cycles,
                           overall low likelihood).

Empirical experiments showed that BOTH families are necessary (see
results/ablation/).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# Per-transition flag result
# ---------------------------------------------------------------------------

@dataclass
class TransitionFlag:
    """Flag values computed for ONE state transition (s_t -> s_{t+1}).

    The detector should produce one TransitionFlag per *flagged* transition.
    Transitions with no fired flags MAY be omitted from the result list, but
    if included must still have all numeric fields populated.

    Attributes
    ----------
    step_index
        Zero-based position of this transition in the trace
        (i.e. the index of s_t).
    from_state, to_state
        The observed transition.
    p_observed
        P(s_{t+1} | s_t [, e_t, x_t]) under the fitted model.
        Must be in (0, 1]; use Laplace smoothing to avoid hard zeros.
    surprise_nats
        −log(p_observed). Higher = rarer.
    rank
        Rank of `to_state` among all candidate next-states, sorted
        descending by probability. 1 = most likely; |S| = least likely.
    n_candidates
        Number of distinct next-states the model considered (typically |S|).
    gap_from_top1
        P(top1) − P(observed). Captures "how strongly the model preferred
        something else." Bounded by [0, 1).
    expected_top3
        The three most likely next states, with their probabilities,
        sorted descending. Used in the LLM evidence pack as a counterfactual.
    training_count
        How many times this exact (from_state -> to_state) was observed
        in training data. 0 means "never seen, only Laplace-smoothing prob".
    impossible
        Convenience flag: True iff training_count == 0.
    fired_flags
        Subset of {"PT-impossible", "PT-surprise", "PT-rank"} indicating
        which thresholds this transition crossed. The detector's
        thresholds are listed in detector.DEFAULT_THRESHOLDS.
    """
    step_index: int
    from_state: str
    to_state: str
    p_observed: float
    surprise_nats: float
    rank: int
    n_candidates: int
    gap_from_top1: float
    expected_top3: list[tuple[str, float]]
    training_count: int
    impossible: bool
    fired_flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 < self.p_observed <= 1.0):
            raise ValueError(
                f"TransitionFlag.p_observed must be in (0, 1]; got {self.p_observed!r}. "
                f"Use Laplace smoothing in the detector to avoid hard zeros."
            )
        if self.surprise_nats < -1e-9:
            raise ValueError(
                f"TransitionFlag.surprise_nats must be >= 0 (it is -log p); "
                f"got {self.surprise_nats!r}."
            )
        if self.rank < 1:
            raise ValueError(
                f"TransitionFlag.rank must be >= 1; got {self.rank!r}."
            )
        if self.training_count < 0:
            raise ValueError(
                f"TransitionFlag.training_count must be >= 0; got {self.training_count!r}."
            )
        if self.impossible != (self.training_count == 0):
            raise ValueError(
                f"TransitionFlag.impossible must equal (training_count == 0); "
                f"got impossible={self.impossible!r}, training_count={self.training_count!r}."
            )
        if self.gap_from_top1 < -1e-9:
            raise ValueError(
                f"TransitionFlag.gap_from_top1 must be >= 0; got {self.gap_from_top1!r}."
            )
        for i in range(len(self.expected_top3) - 1):
            if self.expected_top3[i][1] < self.expected_top3[i + 1][1] - 1e-12:
                raise ValueError(
                    f"TransitionFlag.expected_top3 must be sorted descending by probability; "
                    f"got {self.expected_top3!r}."
                )

    def to_dict(self) -> dict:
        d = asdict(self)
        # Tuples in expected_top3 -> ensure JSON-friendly
        d["expected_top3"] = [
            {"state": s, "probability": round(p, 6)}
            for s, p in self.expected_top3
        ]
        return d


# ---------------------------------------------------------------------------
# Sequence-level (whole-trace) flag result
# ---------------------------------------------------------------------------

@dataclass
class SequenceFlag:
    """Flag values computed for the WHOLE TRACE.

    Two empirically necessary signals:

      - SQ-low-likelihood: the trace's normalized log-likelihood is far below
        the normal distribution. Catches `unlikely_branch`-style anomalies.

      - SQ-edge-repeat: an edge (state-pair) repeats far more than is normal.
        Catches `pingpong`-style cycle anomalies, which slip through
        per-transition flags because each individual step is high-probability.

    Attributes
    ----------
    norm_log_likelihood
        sum(log P(s_{t+1}|s_t)) / (len(trace) - 1). Length-normalized so it
        is comparable across traces of different lengths.
    norm_ll_z_score
        Z-score of `norm_log_likelihood` relative to the distribution of
        normal traces (mean & std measured on the training set).
    max_edge_repeat
        Number of times the most-frequent edge appears in this trace.
        Normal traces have max_edge_repeat == 1 in >98% of cases.
    most_repeated_edge
        The edge (from_state, to_state) achieving max_edge_repeat. None if
        the trace has fewer than 2 states.
    fired_flags
        Subset of {"SQ-low-likelihood", "SQ-edge-repeat"}.
    """
    norm_log_likelihood: float
    norm_ll_z_score: float
    max_edge_repeat: int
    most_repeated_edge: tuple[str, str] | None
    fired_flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_edge_repeat < 0:
            raise ValueError(
                f"SequenceFlag.max_edge_repeat must be >= 0; "
                f"got {self.max_edge_repeat!r}."
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.most_repeated_edge is not None:
            d["most_repeated_edge"] = list(self.most_repeated_edge)
        return d


# ---------------------------------------------------------------------------
# Aggregate detection result
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Top-level result returned by the detector for a single trace.

    Attributes
    ----------
    transition_flags
        One entry per *flagged* transition (those with at least one fired
        flag). Detectors MAY include unflagged transitions too; the evidence
        pack builder filters anyway.
    sequence_flag
        Always exactly one — sequence-level metrics for the whole trace.
    is_anomalous
        Boolean verdict. Equivalent to len(fired_signals) > 0.
        IMPORTANT: this is computed by the DETECTOR, never by the LLM.
    fired_signals
        Union of all fired flags across transition_flags and sequence_flag.
        Used by the LLM as a "what kind of anomaly is this?" routing key.
    """
    transition_flags: list[TransitionFlag]
    sequence_flag: SequenceFlag
    is_anomalous: bool
    fired_signals: list[str]

    def to_dict(self) -> dict:
        return {
            "transition_flags": [tf.to_dict() for tf in self.transition_flags],
            "sequence_flag": self.sequence_flag.to_dict(),
            "is_anomalous": self.is_anomalous,
            "fired_signals": self.fired_signals,
        }
