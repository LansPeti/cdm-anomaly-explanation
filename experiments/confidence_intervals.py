"""Compute Clopper-Pearson 95% confidence intervals on the detector's
recall and false-positive rate.

Reads:
  - results/detector/anomaly_detection_report.json   (detector run on anomalous traces)
  - data/anomalies.jsonl                    (ground-truth labels)
  - optionally: a normal-trace detector run for FP rate

Direct response to the review's repeated request for
confidence intervals on every reported point estimate. Currently the
"100% recall, 4% FP" headline carries no statistical reading.

Usage
-----
    python -m experiments.confidence_intervals
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANOMALY_REPORT = REPO_ROOT / "results" / "detector" / "anomaly_detection_report.json"
ANOMALY_LABELS = REPO_ROOT / "data" / "anomalies.jsonl"
NORMAL_LABELS = REPO_ROOT / "data" / "normal.jsonl"
OUT_DIR = REPO_ROOT / "results" / "ci"


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact 95% CI for a binomial proportion.

    Standard library only — uses the relation between the binomial and
    beta distributions. Implemented via scipy if available, else a manual
    bisection on the regularised incomplete beta function (we use the
    bisection for portability — no scipy dependency needed).
    """
    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        # one-sided upper bound only
        upper = 1.0 - (alpha / 2) ** (1.0 / n)
        return (0.0, upper)
    if k == n:
        lower = (alpha / 2) ** (1.0 / n)
        return (lower, 1.0)

    try:
        from scipy.stats import beta as _beta
        lower = _beta.ppf(alpha / 2, k, n - k + 1)
        upper = _beta.ppf(1 - alpha / 2, k + 1, n - k)
        return (float(lower), float(upper))
    except ImportError:
        pass

    # Bisection on incomplete beta (avoids scipy). Accurate to ~1e-6.
    from math import exp, lgamma, log

    def beta_cdf(x: float, a: float, b: float) -> float:
        # Series for regularized incomplete beta — adequate for moderate (a, b).
        # For our use (n up to a few thousand) this is adequate at moderate cost.
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b)
                 + a * log(x) + b * log(1 - x))
        # Use continued-fraction expansion (Press et al. NR §6.4)
        eps = 1e-12
        max_iter = 1000
        if x < (a + 1) / (a + b + 2):
            cf = _betacf(x, a, b, eps, max_iter)
            return bt * cf / a
        else:
            cf = _betacf(1 - x, b, a, eps, max_iter)
            return 1.0 - bt * cf / b

    def _betacf(x: float, a: float, b: float, eps: float, max_iter: int) -> float:
        qab, qap, qam = a + b, a + 1, a - 1
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                return h
        return h

    def bisect(target_p: float, a: float, b: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if beta_cdf(mid, a, b) < target_p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = bisect(alpha / 2, k, n - k + 1)
    upper = bisect(1 - alpha / 2, k + 1, n - k)
    return (lower, upper)


def fmt(p: float) -> str:
    return f"{100 * p:.2f}%"


def main(argv: list[str] | None = None) -> int:
    if not ANOMALY_REPORT.exists():
        print(f"ERROR: {ANOMALY_REPORT} not found.", file=sys.stderr)
        print("       Run scripts/run_reports.py first.",
              file=sys.stderr)
        return 2

    print("Loading detector report and ground-truth labels...")
    with ANOMALY_REPORT.open() as f:
        report = json.load(f)
    label_by_trade_id = {}
    if ANOMALY_LABELS.exists():
        with ANOMALY_LABELS.open() as f:
            for line in f:
                rec = json.loads(line)
                label_by_trade_id[rec["trade_id"]] = rec["label"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {"by_class": {}, "aggregate": {}}

    # ---- Recall: per anomaly class ---------------------------------------
    by_class: dict[str, dict[str, int]] = {}
    total_n = 0
    total_detected = 0
    for row in report:
        label = label_by_trade_id.get(row["trade_id"], "unlabelled")
        if label == "normal":
            continue
        total_n += 1
        if row["is_anomalous"]:
            total_detected += 1
        bucket = by_class.setdefault(label, {"n": 0, "detected": 0})
        bucket["n"] += 1
        if row["is_anomalous"]:
            bucket["detected"] += 1

    print()
    print(f"{'Anomaly class':<20s}  {'detected':>10s}  {'point':>8s}  "
          f"{'95% CI':>22s}")
    print("-" * 70)
    for label, bucket in sorted(by_class.items()):
        n, k = bucket["n"], bucket["detected"]
        point = k / n if n > 0 else 0.0
        lo, hi = clopper_pearson(k, n)
        print(f"{label:<20s}  {k:>4d}/{n:<5d}  {fmt(point):>8s}  "
              f"[{fmt(lo)}, {fmt(hi)}]")
        summary["by_class"][label] = {
            "n": n, "detected": k, "recall": point,
            "ci_low": lo, "ci_high": hi,
        }
    if total_n > 0:
        agg_point = total_detected / total_n
        agg_lo, agg_hi = clopper_pearson(total_detected, total_n)
        print("-" * 70)
        print(f"{'aggregate':<20s}  {total_detected:>4d}/{total_n:<5d}  "
              f"{fmt(agg_point):>8s}  [{fmt(agg_lo)}, {fmt(agg_hi)}]")
        summary["aggregate"]["recall"] = {
            "n": total_n, "detected": total_detected,
            "point": agg_point, "ci_low": agg_lo, "ci_high": agg_hi,
        }

    # ---- FP rate: needs a normal-trace report. The current report runs the
    # detector on anomalous traces only, so FP cannot be derived here.
    print()
    print("False-positive rate: not derivable from this report (it covers")
    print("anomalous traces only). To compute it, run the detector on the")
    print("normal-trace pool and pass the resulting JSON via --normal-report.")

    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
