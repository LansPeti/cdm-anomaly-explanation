"""
detector.py — The anomaly-detection layer.

CONTENTS
========

- `DEFAULT_THRESHOLDS` — cut-points that decide when a flag fires.
- `AnomalyDetector`    — abstract base class (the contract): `fit`,
                         `detect`, `save`, `load`. `detect()` must return a
                         `DetectionResult` as defined in `flags.py`.
- `EmpiricalMarkovDetector` — the production detector: a first-order
                         empirical Markov chain with Laplace smoothing, plus
                         an optional time-gap extension.
- `MockDetector`       — always reports "no anomaly"; used only in tests.

Adding an alternative model (MLR, XGBoost, ...) means subclassing
`AnomalyDetector`; the rest of the pipeline (evidence_pack → llm_explainer)
works unchanged as long as `detect()` returns a valid `DetectionResult`.
"""
import json
import math
import os
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from statistics import mean, stdev

from src.flags import DetectionResult, SequenceFlag, TransitionFlag

# ---------------------------------------------------------------------------
# Default thresholds (empirically tuned)
#
# These are the cut-points the detector should use to set the *fired_flags*
# fields on TransitionFlag and SequenceFlag. They are not magic; tune them
# on real data and update the doc.
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, float | int] = {
    # Per-transition
    "PT_SURPRISE_NATS":    4.0,   # surprise > 4 => fire PT-surprise
                                  #   (p < 0.018)
    "PT_RANK_MIN":         3,     # rank >= 3 => fire PT-rank
    # Per-transition "PT-impossible" is deterministic: training_count == 0.
    "PT_TIME_GAP_Z_MAX": 3.0,
    # Sequence-level
    "SQ_NORM_LL_Z_MAX":    -2.5,  # norm_ll_z < -2 => fire SQ-low-likelihood
    "SQ_EDGE_REPEAT_MIN":  3,     # max_edge_repeat >= 3 => fire SQ-edge-repeat
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AnomalyDetector(ABC):
    """Contract for any anomaly detector that participates in this pipeline.

    The methods are deliberately minimal so any reasonable model
    (probabilistic, tree-based, neural) can satisfy them.

    Lifecycle
    ---------
    1.  Construct: `det = MyDetector(**hparams)`
    2.  Train:     `det.fit(normal_traces)`
    3.  Persist:   `det.save("data/model/")`
    4.  Reload:    `det = MyDetector(); det.load("data/model/")`
    5.  Predict:   `result = det.detect(trace)`  → DetectionResult
    """

    thresholds: dict[str, float | int]

    def __init__(
        self,
        thresholds: dict | None = None,
        include_time_gap_anomalies: bool = False,
    ):
        # Always start from a copy of the defaults so subclasses don't share state
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.include_time_gap_anomalies = bool(include_time_gap_anomalies)

    # -------- training --------

    @abstractmethod
    def fit(self, normal_traces: list[list[str]]) -> None:
        """Fit the model on a list of NORMAL traces only.

        DO NOT include anomalous traces in training — the detector defines
        normality. Anomalies are detected by deviation from that.

        After fitting, the detector should also have computed the baseline
        statistics needed for sequence-level z-scores (mean/std of
        normalized log-likelihood on the training set).
        """
        ...

    # -------- inference --------

    @abstractmethod
    def detect(self, trace: list[str]) -> DetectionResult:
        """Run the full detection pipeline on a single trace.

        Should return a `DetectionResult` whose:
          - transition_flags lists every transition with ≥1 fired flag
          - sequence_flag is always populated (even if no flags fired)
          - is_anomalous = (fired_signals is non-empty)
          - fired_signals = union over transition_flags and sequence_flag
        """
        ...

    # -------- persistence --------

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the fitted model and the baseline statistics to `path`."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore a previously-saved model from `path`."""
        ...


class EmpiricalMarkovDetector(AnomalyDetector):
    def __init__(
        self,
        alpha: float = 0.5,
        thresholds: dict | None = None,
        include_time_gap_anomalies: bool = False,
    ):
        super().__init__(
            thresholds=thresholds,
            include_time_gap_anomalies=include_time_gap_anomalies,
        )
        self.alpha = alpha
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.totals: Counter[str] = Counter()
        self.vocab: set[str] = set()
        self.time_gap_stats: dict[str, dict[str, tuple[float, float]]] = (
            defaultdict(dict)
        )
        self.time_gap_from_state_stats: dict[str, tuple[float, float]] = {}
        self.global_time_gap_stats: tuple[float, float] | None = None
        self.mu_norm_ll = 0.0
        self.sd_norm_ll = 1.0
        self.is_fitted = False

    def fit(self, normal_traces: list[list[str]]) -> None:
        if not normal_traces:
            raise ValueError("fit() requires at least one normal trace.")

        self.counts = defaultdict(Counter)
        self.totals = Counter()
        self.vocab = set()

        for trace in normal_traces:
            if len(trace) < 2:
                continue

            for state in trace:
                self.vocab.add(str(state))

            for from_state, to_state in zip(trace, trace[1:]):
                from_state = str(from_state)
                to_state = str(to_state)
                self.counts[from_state][to_state] += 1
                self.totals[from_state] += 1

        if not self.vocab:
            raise ValueError("No valid states found in normal_traces.")

        norm_lls = [
            self._norm_log_likelihood(trace)
            for trace in normal_traces
            if len(trace) >= 2
        ]

        if norm_lls:
            self.mu_norm_ll = mean(norm_lls)
            self.sd_norm_ll = stdev(norm_lls) if len(norm_lls) > 1 else 1.0
            if self.sd_norm_ll == 0:
                self.sd_norm_ll = 1.0

        self.is_fitted = True

    def fit_time_gaps(self, transition_rows: list[dict]) -> None:
        if not self.include_time_gap_anomalies:
            self.time_gap_stats = defaultdict(dict)
            self.time_gap_from_state_stats = {}
            self.global_time_gap_stats = None
            return

        buckets = defaultdict(list)
        from_state_buckets = defaultdict(list)
        all_gaps = []

        for row in transition_rows:
            from_state = str(row["current_state"])
            to_state = str(row["next_state"])
            gap = row.get("time_gap_seconds")

            if gap is None:
                continue

            gap = float(gap)
            buckets[(from_state, to_state)].append(gap)
            from_state_buckets[from_state].append(gap)
            all_gaps.append(gap)

        self.time_gap_stats = defaultdict(dict)

        for (from_state, to_state), gaps in buckets.items():
            self.time_gap_stats[from_state][to_state] = self._mean_sd(gaps)

        self.time_gap_from_state_stats = {
            from_state: self._mean_sd(gaps)
            for from_state, gaps in from_state_buckets.items()
        }
        self.global_time_gap_stats = self._mean_sd(all_gaps) if all_gaps else None


    def _time_gap_score(
        self,
        from_state: str,
        to_state: str,
        observed_gap: float | None,
    ) -> float | None:
        if observed_gap is None:
            return None

        stats = self._time_gap_baseline(from_state, to_state)
        if stats is None:
            return None

        expected_mean, expected_sd = stats

        if expected_sd == 0:
            return 0.0 if float(observed_gap) == expected_mean else float("inf")

        return abs(float(observed_gap) - expected_mean) / expected_sd


    def build_time_gap_report(self, transition_rows: list[dict]) -> list[dict]:
        report = []

        for row_index, row in enumerate(transition_rows):
            from_state = str(row["current_state"])
            to_state = str(row["next_state"])
            observed_gap = row.get("time_gap_seconds")
            step_index = int(row.get("step_index", row_index))

            score = self._time_gap_score(
                from_state=from_state,
                to_state=to_state,
                observed_gap=observed_gap,
            )

            stats = self._time_gap_baseline(from_state, to_state)
            if stats is None:
                expected_mean = None
                expected_sd = None
            else:
                expected_mean, expected_sd = stats

            is_time_anomaly = False
            if self.include_time_gap_anomalies:
                is_time_anomaly = (
                    score is not None
                    and score > float(self.thresholds["PT_TIME_GAP_Z_MAX"])
                )

            report.append(
                {
                    "trade_id": row.get("trade_id"),
                    "step_index": step_index,
                    "from_state": from_state,
                    "to_state": to_state,
                    "observed_time_gap_seconds": observed_gap,
                    "expected_time_gap_mean": expected_mean,
                    "expected_time_gap_std": expected_sd,
                    "time_gap_score": score,
                    "is_time_anomaly": is_time_anomaly,
                    "injected_time_anomaly": bool(
                        row.get("injected_time_anomaly", False)
                    ),
                }
            )

        return report

    def detect(self, trace: list[str]) -> DetectionResult:
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted or loaded before detect().")

        trace = [str(s) for s in trace]

        transition_flags: list[TransitionFlag] = []

        for step_index, (from_state, to_state) in enumerate(zip(trace, trace[1:])):
            p_observed = self._p(from_state, to_state)
            surprise_nats = -math.log(p_observed)

            ranked = self._ranked_candidates(from_state)
            states_only = [state for state, _ in ranked]

            rank = (
                states_only.index(to_state) + 1
                if to_state in states_only
                else len(states_only) + 1
            )

            expected_top3 = ranked[:3]
            top1_prob = expected_top3[0][1] if expected_top3 else 0.0
            gap_from_top1 = max(0.0, top1_prob - p_observed)

            training_count = int(self.counts.get(from_state, Counter()).get(to_state, 0))
            impossible = training_count == 0

            fired_flags: list[str] = []

            if impossible:
                fired_flags.append("PT-impossible")

            if surprise_nats > float(self.thresholds["PT_SURPRISE_NATS"]):
                fired_flags.append("PT-surprise")

            if rank >= int(self.thresholds["PT_RANK_MIN"]):
                fired_flags.append("PT-rank")

            if fired_flags:
                transition_flags.append(
                    TransitionFlag(
                        step_index=step_index,
                        from_state=from_state,
                        to_state=to_state,
                        p_observed=p_observed,
                        surprise_nats=surprise_nats,
                        rank=rank,
                        n_candidates=len(self.vocab),
                        gap_from_top1=gap_from_top1,
                        expected_top3=expected_top3,
                        training_count=training_count,
                        impossible=impossible,
                        fired_flags=fired_flags,
                    )
                )

        sequence_flag = self._sequence_flag(trace)

        fired_signals = self._unique_preserve_order(
            [
                flag
                for tf in transition_flags
                for flag in tf.fired_flags
            ]
            + sequence_flag.fired_flags
        )

        return DetectionResult(
            transition_flags=transition_flags,
            sequence_flag=sequence_flag,
            is_anomalous=bool(fired_signals),
            fired_signals=fired_signals,
        )
    
    
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        payload = {
            "alpha": self.alpha,
            "thresholds": self.thresholds,
            "include_time_gap_anomalies": self.include_time_gap_anomalies,
            "counts": {
                from_state: dict(counter)
                for from_state, counter in self.counts.items()
            },
            "totals": dict(self.totals),
            "vocab": sorted(self.vocab),
            "mu_norm_ll": self.mu_norm_ll,
            "sd_norm_ll": self.sd_norm_ll,
            "is_fitted": self.is_fitted,
            "time_gap_stats": {
                from_state: {
                    to_state: [mu, sd]
                    for to_state, (mu, sd) in to_states.items()
                }
                for from_state, to_states in self.time_gap_stats.items()
            },
            "time_gap_from_state_stats": {
                from_state: [mu, sd]
                for from_state, (mu, sd) in self.time_gap_from_state_stats.items()
            },
            "global_time_gap_stats": (
                list(self.global_time_gap_stats)
                if self.global_time_gap_stats is not None
                else None
            ),
        }

        with open(os.path.join(path, "empirical_markov_detector.json"), "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, path: str) -> None:
        with open(os.path.join(path, "empirical_markov_detector.json")) as f:
            payload = json.load(f)

        self.alpha = float(payload["alpha"])
        self.thresholds = payload["thresholds"]
        self.include_time_gap_anomalies = bool(
            payload.get("include_time_gap_anomalies", False)
        )

        self.counts = defaultdict(Counter)
        for from_state, counter in payload["counts"].items():
            self.counts[from_state] = Counter(counter)

        self.totals = Counter(payload["totals"])
        self.vocab = set(payload["vocab"])
        self.mu_norm_ll = float(payload["mu_norm_ll"])
        self.sd_norm_ll = float(payload["sd_norm_ll"])
        self.is_fitted = bool(payload["is_fitted"])
        self.time_gap_stats = defaultdict(dict)
        for from_state, to_states in payload.get("time_gap_stats", {}).items():
            for to_state, values in to_states.items():
                self.time_gap_stats[from_state][to_state] = (
                    float(values[0]),
                    float(values[1]),
                )
        self.time_gap_from_state_stats = {
            from_state: (float(values[0]), float(values[1]))
            for from_state, values
            in payload.get("time_gap_from_state_stats", {}).items()
        }
        global_stats = payload.get("global_time_gap_stats")
        self.global_time_gap_stats = (
            (float(global_stats[0]), float(global_stats[1]))
            if global_stats is not None
            else None
        )

    @staticmethod
    def _mean_sd(values: list[float]) -> tuple[float, float]:
        mu = sum(values) / len(values)

        if len(values) > 1:
            variance = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
            sd = variance ** 0.5
        else:
            sd = 0.0

        return mu, sd

    def _time_gap_baseline(
        self,
        from_state: str,
        to_state: str,
    ) -> tuple[float, float] | None:
        return (
            self.time_gap_stats.get(from_state, {}).get(to_state)
            or self.time_gap_from_state_stats.get(from_state)
            or self.global_time_gap_stats
        )

    def _p(self, from_state: str, to_state: str) -> float:
        n_states = max(len(self.vocab), 1)
        denominator = self.totals.get(from_state, 0) + self.alpha * n_states
        numerator = self.counts.get(from_state, Counter()).get(to_state, 0) + self.alpha
        return numerator / denominator

    def _ranked_candidates(self, from_state: str) -> list[tuple[str, float]]:
        candidates = sorted(self.vocab)

        ranked = [
            (candidate, self._p(from_state, candidate))
            for candidate in candidates
        ]

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def export_trace_transition_context(
        self,
        trace: list[str],
        top_k: int = 3,
    ) -> dict:
        """Return the fitted transition-matrix slice relevant to one trace."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted or loaded before export.")

        source_states = self._unique_preserve_order([str(s) for s in trace[:-1]])
        learned_top_k = {}

        for from_state in source_states:
            learned_top_k[from_state] = [
                {
                    "to_state": to_state,
                    "probability": round(probability, 6),
                    "training_count": int(
                        self.counts.get(from_state, Counter()).get(to_state, 0)
                    ),
                    "rank": rank,
                }
                for rank, (to_state, probability)
                in enumerate(self._ranked_candidates(from_state)[:top_k], start=1)
            ]

        return {
            "source": "fitted_normal_transition_matrix",
            "smoothing": {
                "method": "laplace",
                "alpha": self.alpha,
            },
            "top_k": top_k,
            "learned_transition_top_k": learned_top_k,
        }

    def _norm_log_likelihood(self, trace: list[str]) -> float:
        if len(trace) < 2:
            return 0.0

        log_likelihood = 0.0

        for from_state, to_state in zip(trace, trace[1:]):
            p = self._p(str(from_state), str(to_state))
            log_likelihood += math.log(p)

        return log_likelihood / (len(trace) - 1)

    def _sequence_flag(self, trace: list[str]) -> SequenceFlag:
        norm_ll = self._norm_log_likelihood(trace)
        z_score = (norm_ll - self.mu_norm_ll) / self.sd_norm_ll

        edge_counts = Counter(zip(trace, trace[1:]))
        if edge_counts:
            most_repeated_edge, max_edge_repeat = edge_counts.most_common(1)[0]
        else:
            most_repeated_edge = None
            max_edge_repeat = 0

        fired_flags: list[str] = []

        if z_score < float(self.thresholds["SQ_NORM_LL_Z_MAX"]):
            fired_flags.append("SQ-low-likelihood")

        if max_edge_repeat >= int(self.thresholds["SQ_EDGE_REPEAT_MIN"]):
            fired_flags.append("SQ-edge-repeat")

        return SequenceFlag(
            norm_log_likelihood=norm_ll,
            norm_ll_z_score=z_score,
            max_edge_repeat=max_edge_repeat,
            most_repeated_edge=most_repeated_edge,
            fired_flags=fired_flags,
        )

    @staticmethod
    def _unique_preserve_order(items: list[str]) -> list[str]:
        seen = set()
        result = []

        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)

        return result

# ---------------------------------------------------------------------------
# Mock detector (used by demo / tests; never in production)
# ---------------------------------------------------------------------------

class MockDetector(AnomalyDetector):
    """ALWAYS returns "not anomalous". Use only for end-to-end pipeline tests
    when the real detector is not yet implemented.

    The evidence pack builder and LLM explainer should both produce the
    "Normal trace, no findings." path when fed this detector's output.
    """

    def fit(self, normal_traces: list[list[str]]) -> None:
        # No state needed — always clean.
        return

    def detect(self, trace: list[str]) -> DetectionResult:
        seq = SequenceFlag(
            norm_log_likelihood=0.0,
            norm_ll_z_score=0.0,
            max_edge_repeat=1,
            most_repeated_edge=tuple(trace[:2]) if len(trace) >= 2 else None,
            fired_flags=[],
        )
        return DetectionResult(
            transition_flags=[],
            sequence_flag=seq,
            is_anomalous=False,
            fired_signals=[],
        )

    def save(self, path: str) -> None:
        return  # nothing to save

    def load(self, path: str) -> None:
        return  # nothing to load
