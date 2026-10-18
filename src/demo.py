"""
demo.py — End-to-end pipeline demo (CLI).

Wires together:
  data_generator -> detector -> evidence_pack -> llm_explainer

The `EmpiricalMarkovDetector` is fitted on freshly generated normal
traces, then run on one trace — a normal one by default, or one with an
injected anomaly via `--anomaly <class>`.

USAGE
=====

    # Normal trace, no LLM call (no API key needed):
    python -m src.demo --no-llm

    # Anomalous trace, with LLM explanation (needs OPENAI_API_KEY):
    python -m src.demo --anomaly pingpong

    # Or pipe a hand-crafted trace as JSON:
    echo '{"trace": ["NEW","EXECUTED","CAPTURED","MATCHED","CLEARED","SETTLED"]}' \
        | python -m src.demo --stdin --no-llm
"""
from __future__ import annotations

import argparse
import json
import random
import sys

from src.data_generator import ANOMALY_INJECTORS, generate_normal_trace
from src.detector import AnomalyDetector, EmpiricalMarkovDetector
from src.evidence_pack import build_evidence_pack
from src.llm_explainer import AnomalyExplainer, validate_explanation


def fit_detector(rng: random.Random, n_train: int = 2000) -> AnomalyDetector:
    """Fit the Markov detector on `n_train` synthetic normal traces."""
    detector = EmpiricalMarkovDetector()
    detector.fit([generate_normal_trace(rng) for _ in range(n_train)])
    return detector


def run_one(trace: list[str], trade_id: str, detector: AnomalyDetector,
            with_llm: bool = True) -> dict:
    """Run the full pipeline on a single trace and return all artifacts."""
    detection = detector.detect(trace)
    pack = build_evidence_pack(trade_id, trace, detection)

    explanation_text = None
    validation = None
    if with_llm:
        try:
            explainer = AnomalyExplainer()
            result = explainer.explain(pack)
            explanation_text = result.text
            validation = validate_explanation(result.text, pack)
        except RuntimeError as e:
            # No API key — keep the demo runnable without one
            explanation_text = f"[LLM skipped — {e}]"
            validation = None

    return {
        "trade_id": trade_id,
        "trace": trace,
        "detection": detection.to_dict(),
        "evidence_pack": pack,
        "explanation": explanation_text,
        "validation": validation,
    }


def _anomalous_trace(cls: str, rng: random.Random) -> list[str]:
    injector = ANOMALY_INJECTORS[cls]
    for _ in range(100):
        new_trace, _meta = injector(generate_normal_trace(rng), rng)
        if new_trace is not None:
            return new_trace
    raise RuntimeError(f"Could not inject anomaly class {cls!r}.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ISDA CDM anomaly pipeline demo")
    ap.add_argument("--trade-id", default="TRD-DEMO-001")
    ap.add_argument("--stdin", action="store_true",
                    help='Read a JSON object {"trace": [...]} from stdin.')
    ap.add_argument("--anomaly", choices=sorted(ANOMALY_INJECTORS),
                    help="Inject this anomaly class into the generated trace.")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip the LLM call; print only the evidence pack.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    detector = fit_detector(rng)

    if args.stdin:
        trace = json.loads(sys.stdin.read())["trace"]
    elif args.anomaly:
        trace = _anomalous_trace(args.anomaly, rng)
    else:
        trace = generate_normal_trace(rng)

    out = run_one(trace, args.trade_id, detector, with_llm=not args.no_llm)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
