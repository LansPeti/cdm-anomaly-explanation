"""
lifecycle_graph.py — ISDA CDM trade lifecycle state machine.

Defines the canonical state space and the LEGAL transitions, faithful to
the project's lifecycle specification (Feb 2026) and ISDA CDM literature (Bakshi et al. 2021).

This module is purely declarative: it answers
  - what states exist
  - what transitions are legal (with their canonical normal-flow weights)
  - which states are terminal

It does NOT do any anomaly detection — that's the job of `detector.py` (which
Hanna implements). The probability weights here are the *generative* ground
truth used by `data_generator.py` to produce synthetic normal traces.

Importantly: in the operational pipeline these weights are NEVER referenced
by the detector. The detector learns its own empirical probabilities from
the generated traces. The graph here is just the data-generating story.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# State space (per the project's lifecycle specification, 2026-02-27)
# ---------------------------------------------------------------------------

STATES: list[str] = [
    # Happy path
    "NEW",          # Order received
    "EXECUTED",     # Market execution
    "CAPTURED",     # Internal booking (middle office)
    "ENRICHED",     # SSI / fees / client data added
    "MATCHED",      # Counterparty agreement
    "CLEARED",      # Clearing house acceptance
    "SETTLED",      # Funds & shares exchanged (terminal success)

    # Error states
    "UNMATCHED_ERROR",     # Matching failure
    "FAILED_SETTLEMENT",   # Settlement failure (e.g. insufficient funds)

    # Repair / lifecycle
    "AMENDED",      # Manual correction; routes back to ENRICHED

    # Terminal failures
    "REJECTED",     # Invalid order
    "CANCELLED",    # Voluntary or discrepancy-driven cancellation
]

TERMINAL_STATES: set[str] = {"SETTLED", "REJECTED", "CANCELLED"}


# ---------------------------------------------------------------------------
# Transition matrix — the "ground truth" used to GENERATE normal data.
#
# Each entry: source -> [(target, weight), ...]
# Weights are interpreted as un-normalized probabilities; the data generator
# normalizes per source.
#
# These weights reflect realistic post-trade processing: most trades flow
# straight through, a small fraction hits errors and recovers via AMENDED.
# ---------------------------------------------------------------------------

TRANSITIONS: dict[str, list[tuple[str, float]]] = {
    "NEW":               [("EXECUTED", 0.97), ("REJECTED", 0.03)],
    "EXECUTED":          [("CAPTURED", 0.99), ("REJECTED", 0.01)],
    "CAPTURED":          [("ENRICHED", 0.95), ("CANCELLED", 0.05)],
    "ENRICHED":          [("MATCHED", 0.85), ("UNMATCHED_ERROR", 0.15)],
    "MATCHED":           [("CLEARED", 0.97), ("CANCELLED", 0.03)],
    "CLEARED":           [("SETTLED", 0.92), ("FAILED_SETTLEMENT", 0.08)],
    "UNMATCHED_ERROR":   [("AMENDED", 1.00)],
    "AMENDED":           [("ENRICHED", 1.00)],
    "FAILED_SETTLEMENT": [("AMENDED", 1.00)],
    # Terminal states (SETTLED, REJECTED, CANCELLED) have no outgoing edges.
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES


def legal_successors(state: str) -> list[str]:
    """Returns the list of states that can legally follow `state`."""
    return [s for s, _ in TRANSITIONS.get(state, [])]


def is_legal_transition(src: str, dst: str) -> bool:
    return dst in legal_successors(src)


def transition_weight(src: str, dst: str) -> float | None:
    """Returns the canonical generative weight, or None if illegal."""
    for s, w in TRANSITIONS.get(src, []):
        if s == dst:
            return w
    return None


def shortest_legal_path(src: str, dst: str) -> list[str] | None:
    """BFS for the shortest legal state path from src to dst (None if unreachable)."""
    if src == dst:
        return [src]
    seen = {src}
    queue: list[tuple[str, list[str]]] = [(src, [src])]
    while queue:
        cur, path = queue.pop(0)
        for nxt in legal_successors(cur):
            if nxt in seen:
                continue
            new_path = path + [nxt]
            if nxt == dst:
                return new_path
            seen.add(nxt)
            queue.append((nxt, new_path))
    return None


def graph_summary() -> dict:
    """Diagnostic summary used by the LLM as a domain glossary."""
    return {
        "states": STATES,
        "terminal_states": sorted(TERMINAL_STATES),
        "n_legal_transitions": sum(len(v) for v in TRANSITIONS.values()),
        "happy_path": ["NEW", "EXECUTED", "CAPTURED", "ENRICHED",
                       "MATCHED", "CLEARED", "SETTLED"],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(graph_summary(), indent=2))
    print()
    print("Reachability check (NEW -> SETTLED):",
          shortest_legal_path("NEW", "SETTLED"))
    print("Illegal? NEW -> SETTLED direct:",
          not is_legal_transition("NEW", "SETTLED"))
