"""
data_generator.py — Synthetic ISDA CDM trade lifecycle data factory.

WHAT THIS MODULE PRODUCES
=========================

Three JSONL files in the output directory:
  - normal.jsonl     — N normal traces that follow the canonical flow
  - anomalies.jsonl  — N × 4 traces with one of 4 anomaly classes injected
  - dataset.jsonl    — interleaved combined file, suitable for end-to-end eval

DESIGN NOTES
============

We faithfully follow the project's lifecycle specification (Feb 2026) state space and add
TIMESTAMPS to each event so the data is forward-compatible with future
extensions (e.g. timeout-based anomalies, contextual feature x_t conditioning).

ANOMALY CLASSES (4)
===================

| class            | mechanism                                          | which signal catches it? |
| ---------------- | -------------------------------------------------- | ------------------------ |
| skipped_step     | Remove an intermediate state from a normal trace   | per-transition impossible / surprise |
| impossible       | Insert an illegal state mid-trace                  | per-transition impossible / surprise |
| pingpong         | Repeat ENRICHED→UNMATCHED_ERROR→AMENDED 5–10×      | sequence-level edge_repeat ONLY |
| unlikely_branch  | Force REJECTED early (legal, but rare)             | sequence-level low_likelihood ONLY |

These four classes were chosen to span the FULL signal coverage matrix:
  - skipped_step  + impossible   → require per-transition flags
  - pingpong                     → requires sequence-level cycle flag
  - unlikely_branch              → requires sequence-level likelihood flag

Empirical experiments showed that NO single flag
covers all four classes. The 4-class design is deliberate: it forces the
detector to implement BOTH per-transition and sequence-level signals.

USAGE
=====

    python -m src.data_generator --n-normal 5000 --n-per-anomaly 250 --out data/

OUTPUT SCHEMA (one JSON object per line)
=========================================

    {
      "trade_id": "TRD-2026-NORMAL-000001",
      "trace": ["NEW", "EXECUTED", "CAPTURED", ...],
      "events": [
          {"from": "NEW", "to": "EXECUTED",
           "ts": "2026-01-01T09:00:05Z", "dt_seconds": 5.2},
          ...
      ],
      "label": "normal" | "skipped_step" | "impossible" |
               "pingpong" | "unlikely_branch",
      "is_anomalous": false | true,
      "metadata": { ...injection details for anomalies, empty for normal... }
    }
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from src.lifecycle_graph import (
    STATES,
    TERMINAL_STATES,
    TRANSITIONS,
    legal_successors,
)

# ---------------------------------------------------------------------------
# Realistic per-transition timing distributions (seconds)
# Each transition has a (mean, sd) tuple. Sampled per transition to add a
# forward-compatible time dimension. Not used by detection; included so the
# data is also useful for future timeout / dwell-time anomaly types.
# ---------------------------------------------------------------------------

TIMING_SECONDS: dict[tuple[str, str], tuple[float, float]] = {
    ("NEW", "EXECUTED"):              (5, 2),
    ("NEW", "REJECTED"):              (1, 0.5),
    ("EXECUTED", "CAPTURED"):         (60, 30),
    ("EXECUTED", "REJECTED"):         (10, 5),
    ("CAPTURED", "ENRICHED"):         (300, 120),
    ("CAPTURED", "CANCELLED"):        (300, 200),
    ("ENRICHED", "MATCHED"):          (1800, 600),
    ("ENRICHED", "UNMATCHED_ERROR"):  (3600, 1800),
    ("MATCHED", "CLEARED"):           (3600, 1200),
    ("MATCHED", "CANCELLED"):         (3600, 1200),
    ("CLEARED", "SETTLED"):           (172800, 43200),    # T+2 ± 12 h
    ("CLEARED", "FAILED_SETTLEMENT"): (172800, 43200),
    ("UNMATCHED_ERROR", "AMENDED"):   (7200, 3600),
    ("FAILED_SETTLEMENT", "AMENDED"): (86400, 43200),
    ("AMENDED", "ENRICHED"):          (1800, 900),
}
DEFAULT_TIMING = (60, 30)


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

def _sample_next(state: str, rng: random.Random) -> str | None:
    if state in TERMINAL_STATES or state not in TRANSITIONS:
        return None
    options = TRANSITIONS[state]
    states_, weights_ = zip(*options)
    return rng.choices(states_, weights=weights_, k=1)[0]


def _sample_dt(src: str, dst: str, rng: random.Random) -> float:
    mu, sd = TIMING_SECONDS.get((src, dst), DEFAULT_TIMING)
    return max(rng.gauss(mu, sd), 1.0)


def _build_events(trace: list[str], start_ts: datetime,
                  rng: random.Random) -> list[dict]:
    events = []
    cur = start_ts
    for i in range(len(trace) - 1):
        dt = _sample_dt(trace[i], trace[i+1], rng)
        cur = cur + timedelta(seconds=dt)
        events.append({
            "from": trace[i],
            "to": trace[i+1],
            "ts": cur.isoformat() + "Z",
            "dt_seconds": round(dt, 1),
        })
    return events


def generate_normal_trace(rng: random.Random, max_len: int = 30) -> list[str]:
    """Generate a single normal trace by sampling the transition matrix."""
    trace = ["NEW"]
    for _ in range(max_len):
        nxt = _sample_next(trace[-1], rng)
        if nxt is None:
            break
        trace.append(nxt)
    return trace


# ---------------------------------------------------------------------------
# Anomaly injectors
#
# Each returns (modified_trace, metadata_dict).
# If injection cannot be applied to this base trace, returns (None, {}) and
# the dataset assembler tries with a different base trace.
# ---------------------------------------------------------------------------

def inject_skipped_step(trace: list[str],
                        rng: random.Random) -> tuple[list[str] | None, dict]:
    """Remove an intermediate state — creates an illegal jump.

    The previous state's only edge to the kept-after state will be illegal,
    so this lands in `PT-impossible` reliably.
    """
    if len(trace) < 4:
        return None, {}
    idx = rng.randint(1, len(trace) - 2)
    removed = trace[idx]
    new_trace = trace[:idx] + trace[idx+1:]
    return new_trace, {
        "removed_state": removed,
        "at_index": idx,
        "creates_illegal_edge": f"{trace[idx-1]} -> {trace[idx+1]}",
    }


def inject_impossible_transition(trace: list[str],
                                 rng: random.Random) -> tuple[list[str] | None, dict]:
    """Insert a state that can't legally follow the previous one."""
    if len(trace) < 2:
        return None, {}
    idx = rng.randint(1, len(trace) - 1)
    cur = trace[idx-1]
    legal = set(legal_successors(cur))
    illegal = [s for s in STATES if s != cur and s not in legal]
    if not illegal:
        return None, {}
    bad = rng.choice(illegal)
    new_trace = trace[:idx] + [bad] + trace[idx+1:]
    return new_trace, {
        "injected": bad,
        "after": cur,
        "at_index": idx,
        "would_have_been": trace[idx],
    }


def inject_pingpong(trace: list[str],
                    rng: random.Random) -> tuple[list[str] | None, dict]:
    """Insert ENRICHED→UNMATCHED_ERROR→AMENDED loops.

    Per the lifecycle specification: 5–10 cycles is the canonical
    pingpong pattern. Each cycle adds 3 states.

    Each individual transition in the loop is HIGH probability — so the
    only signal that catches this is sequence-level edge_repeat or the
    raw trace length.
    """
    if "ENRICHED" not in trace:
        return None, {}
    n_cycles = rng.randint(5, 10)
    idx = trace.index("ENRICHED")
    loop = ["UNMATCHED_ERROR", "AMENDED", "ENRICHED"] * n_cycles
    new_trace = trace[:idx+1] + loop + trace[idx+1:]
    return new_trace, {
        "n_cycles": n_cycles,
        "starting_at_index": idx,
        "added_states": len(loop),
    }


def inject_unlikely_branch(trace: list[str],
                           rng: random.Random) -> tuple[list[str] | None, dict]:
    """Force an early REJECTED — legal transition, but very rare in normal data.

    This is THE adversarial case for per-transition flags: REJECTED only
    follows NEW or EXECUTED in the canonical graph, so the transition itself
    isn't impossible. But P(REJECTED | NEW) ≈ 0.03 — which gives surprise
    only ~3.5 nats, just below the threshold. The whole-trace likelihood
    drops sharply because the trace ends abruptly.
    """
    if len(trace) < 2:
        return None, {}
    for i, s in enumerate(trace):
        if s in ("NEW", "EXECUTED") and i + 1 < len(trace):
            return trace[:i+1] + ["REJECTED"], {
                "forced_state": "REJECTED",
                "after": s,
                "at_index": i + 1,
            }
    return None, {}


ANOMALY_INJECTORS = {
    "skipped_step":     inject_skipped_step,
    "impossible":       inject_impossible_transition,
    "pingpong":         inject_pingpong,
    "unlikely_branch":  inject_unlikely_branch,
}


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def make_record(trade_id: str, trace: list[str], label: str,
                metadata: dict, start_ts: datetime,
                rng: random.Random) -> dict:
    return {
        "trade_id": trade_id,
        "trace": trace,
        "events": _build_events(trace, start_ts, rng),
        "label": label,
        "is_anomalous": label != "normal",
        "metadata": metadata,
    }


def generate_dataset(n_normal: int = 5000, n_per_anomaly: int = 250,
                     seed: int = 42, start_date: str = "2026-01-01"):
    rng = random.Random(seed)
    base_ts = datetime.fromisoformat(start_date)

    normal: list[dict] = []
    for i in range(n_normal):
        trace = generate_normal_trace(rng)
        ts = base_ts + timedelta(seconds=rng.randint(0, 86400 * 60))
        normal.append(make_record(
            trade_id=f"TRD-2026-NORMAL-{i:06d}",
            trace=trace, label="normal", metadata={},
            start_ts=ts, rng=rng,
        ))

    anomalies: list[dict] = []
    counter = 0
    for cls, injector in ANOMALY_INJECTORS.items():
        attempts = 0
        produced = 0
        while produced < n_per_anomaly and attempts < n_per_anomaly * 5:
            attempts += 1
            base = generate_normal_trace(rng)
            new_trace, meta = injector(base, rng)
            if new_trace is None:
                continue
            ts = base_ts + timedelta(seconds=rng.randint(0, 86400 * 60))
            anomalies.append(make_record(
                trade_id=f"TRD-2026-{cls.upper()}-{counter:06d}",
                trace=new_trace, label=cls, metadata=meta,
                start_ts=ts, rng=rng,
            ))
            produced += 1
            counter += 1

    return normal, anomalies


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        description="Generate synthetic ISDA CDM trade lifecycle data."
    )
    ap.add_argument("--n-normal", type=int, default=5000,
                    help="Number of normal traces (default 5000)")
    ap.add_argument("--n-per-anomaly", type=int, default=250,
                    help="Number of traces per anomaly class (default 250)")
    ap.add_argument("--out", type=str, default="data",
                    help="Output directory (default ./data)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    print(f"Generating {args.n_normal} normal + "
          f"{args.n_per_anomaly} x {len(ANOMALY_INJECTORS)} anomaly traces...")
    normal, anomalies = generate_dataset(
        n_normal=args.n_normal,
        n_per_anomaly=args.n_per_anomaly,
        seed=args.seed,
    )

    out = Path(args.out)
    write_jsonl(normal, out / "normal.jsonl")
    write_jsonl(anomalies, out / "anomalies.jsonl")
    write_jsonl(normal + anomalies, out / "dataset.jsonl")

    by_class: dict[str, int] = defaultdict(int)
    for r in anomalies:
        by_class[r["label"]] += 1
    print(f"  [OK] Normal:    {len(normal)} -> {out/'normal.jsonl'}")
    print(f"  [OK] Anomalies: {len(anomalies)} -> {out/'anomalies.jsonl'}")
    for cls, n in sorted(by_class.items()):
        print(f"        {cls:18s} {n}")
    print(f"  [OK] Combined:  {len(normal)+len(anomalies)} -> {out/'dataset.jsonl'}")


if __name__ == "__main__":
    main()
