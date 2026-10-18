from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.detector import EmpiricalMarkovDetector

DATA_DIR = Path("data")
REPORT_DIR = Path("results/detector")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rows_to_traces(records: list[dict[str, Any]]) -> list[list[str]]:
    return [record["trace"] for record in records]


def records_to_time_gap_rows(
    records: list[dict[str, Any]],
    *,
    detector: EmpiricalMarkovDetector | None = None,
    inject_one_time_anomaly_per_record: bool = False,
) -> list[dict[str, Any]]:
    """Convert generated event timestamps into time-gap detector rows.

    When ``inject_one_time_anomaly_per_record`` is true, one transition per
    record is stretched beyond the fitted timing threshold. This creates a
    dedicated time-anomaly stress set instead of pretending that structural
    sequence anomalies are also timing anomalies.
    """
    rows: list[dict[str, Any]] = []

    for rec_index, record in enumerate(records):
        trade_id = record.get("trade_id", f"TRD-{rec_index}")
        events = record.get("events", [])
        record_rows: list[dict[str, Any]] = []

        for step_index, event in enumerate(events):
            record_rows.append(
                {
                    "trade_id": trade_id,
                    "step_index": step_index,
                    "current_state": event["from"],
                    "next_state": event["to"],
                    "time_gap_seconds": event.get("dt_seconds"),
                    "injected_time_anomaly": False,
                }
            )

        if inject_one_time_anomaly_per_record:
            _inject_time_anomaly(record_rows, detector)

        rows.extend(record_rows)

    return rows


def _inject_time_anomaly(
    rows: list[dict[str, Any]],
    detector: EmpiricalMarkovDetector | None,
) -> None:
    if detector is None:
        return

    threshold = float(detector.thresholds["PT_TIME_GAP_Z_MAX"])

    for row in rows:
        from_state = str(row["current_state"])
        to_state = str(row["next_state"])
        stats = detector._time_gap_baseline(from_state, to_state)
        if stats is None:
            continue

        expected_mean, expected_sd = stats
        effective_sd = (
            expected_sd
            if expected_sd > 0
            else max(abs(expected_mean) * 0.1, 1.0)
        )
        row["time_gap_seconds"] = round(
            expected_mean + (threshold + 2.0) * effective_sd,
            1,
        )
        row["injected_time_anomaly"] = True
        return


def build_anomaly_detection_report(
    detector: EmpiricalMarkovDetector,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a findings-only report.

    Keep sequence-only anomalies too: result.is_anomalous is true when
    either transition-level or sequence-level signals fire.
    """
    report = []

    for i, record in enumerate(records):
        trade_id = record.get("trade_id", f"TRD-{i}")
        trace = record["trace"]
        result = detector.detect(trace)

        if not result.is_anomalous:
            continue

        report.append(
            {
                "trade_id": trade_id,
                "trace": trace,
                "is_anomalous": result.is_anomalous,
                "fired_signals": result.fired_signals,
                "sequence_flag": result.sequence_flag.to_dict(),
                "transition_flags": [
                    tf.to_dict()
                    for tf in result.transition_flags
                ],
            }
        )

    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build anomaly and optional time-gap reports."
    )
    time_group = ap.add_mutually_exclusive_group()
    time_group.add_argument(
        "--include-time-anomalies",
        "--include-time-gap-anomalies",
        action="store_true",
        dest="include_time_anomalies",
        help="I want to include time anomalies: also fit and report the time-gap watcher.",
    )
    time_group.add_argument(
        "--exclude-time-anomalies",
        "--exclude-time-gap-anomalies",
        action="store_false",
        dest="include_time_anomalies",
        help="I don't want to include time anomalies: use only probability-based anomaly detection.",
    )
    ap.add_argument(
        "--time-anomaly-source",
        choices=("stress", "observed"),
        default="stress",
        help=(
            "For --include-time-anomalies, use 'stress' to inject one "
            "controlled time anomaly per trace, or 'observed' to score the "
            "raw generated timestamps only."
        ),
    )
    ap.set_defaults(include_time_anomalies=False)
    args = ap.parse_args()

    REPORT_DIR.mkdir(exist_ok=True)

    normal_path = DATA_DIR / "normal.jsonl"
    anomalies_path = DATA_DIR / "anomalies.jsonl"

    if not normal_path.exists() or not anomalies_path.exists():
        raise FileNotFoundError(
            "Missing data files. First run:\n"
            "python -m src.data_generator --seed 42 --n-normal 5000 --n-per-anomaly 250"
        )

    normal_records = load_jsonl(normal_path)
    anomaly_records = load_jsonl(anomalies_path)

    normal_traces = rows_to_traces(normal_records)

    detector = EmpiricalMarkovDetector(
        include_time_gap_anomalies=args.include_time_anomalies
    )

    if args.include_time_anomalies:
        print("Time anomaly mode: I want to include time anomalies.")
        print(f"Time anomaly source: {args.time_anomaly_source}")
    else:
        print("Time anomaly mode: I don't want to include time anomalies (default).")

    # 1. Main Markov anomaly model
    detector.fit(normal_traces)

    # 2. Optional time-gap model
    if args.include_time_anomalies:
        normal_time_gap_rows = records_to_time_gap_rows(records=normal_records)
        detector.fit_time_gaps(normal_time_gap_rows)
    else:
        detector.fit_time_gaps([])

    # 3. Report 1: findings-only anomaly detection report over anomalous data
    anomaly_detection_report = build_anomaly_detection_report(
        detector=detector,
        records=anomaly_records,
    )

    anomaly_report_path = REPORT_DIR / "anomaly_detection_report.json"
    anomaly_report_path.write_text(
        json.dumps(anomaly_detection_report, indent=2),
        encoding="utf-8",
    )

    print("Reports written:")
    print(f"  {anomaly_report_path}")

    if args.include_time_anomalies:
        anomaly_time_gap_rows = records_to_time_gap_rows(
            records=anomaly_records,
            detector=detector,
            inject_one_time_anomaly_per_record=args.time_anomaly_source == "stress",
        )
        raw_time_gap_report = detector.build_time_gap_report(anomaly_time_gap_rows)
        time_gap_report = [
            row for row in raw_time_gap_report
            if row["is_time_anomaly"]
        ]

        time_gap_report_path = REPORT_DIR / "time_gap_report.json"
        time_gap_report_path.write_text(
            json.dumps(time_gap_report, indent=2),
            encoding="utf-8",
        )

        n_time_anomalies = len(time_gap_report)
        n_injected_time_anomalies = sum(
            row.get("injected_time_anomaly", False)
            for row in anomaly_time_gap_rows
        )
        n_detected_injected = sum(
            row["is_time_anomaly"]
            for row in raw_time_gap_report
            if row.get("injected_time_anomaly", False)
        )
        print(f"  {time_gap_report_path}")
    else:
        n_time_anomalies = 0
        n_injected_time_anomalies = 0
        n_detected_injected = 0
        print("  [time-gap report skipped]")

    n_detected = sum(row["is_anomalous"] for row in anomaly_detection_report)

    print()
    print(f"Anomaly detector: {n_detected}/{len(anomaly_records)} traces flagged")
    if args.include_time_anomalies:
        n_time_rows = len(anomaly_time_gap_rows)
        print(f"Time-gap detector: {n_time_anomalies}/{n_time_rows} transitions flagged")
        if args.time_anomaly_source == "stress":
            print(
                "Time-gap stress recall: "
                f"{n_detected_injected}/{n_injected_time_anomalies} injected gaps flagged"
            )
    else:
        print("Time-gap detector: disabled")


if __name__ == "__main__":
    main()
