"""ISDA CDM Trade Lifecycle Anomaly Explanation System.

Public API surface:

  - data_generator    : synthetic trace factory
  - lifecycle_graph   : state space + legal transitions
  - flags             : DetectionResult / TransitionFlag / SequenceFlag
  - detector          : abstract AnomalyDetector + DEFAULT_THRESHOLDS
                        + EmpiricalMarkovDetector (MockDetector for tests)
  - evidence_pack     : DetectionResult -> JSON pack for LLM
  - llm_explainer     : pack -> markdown explanation
"""
