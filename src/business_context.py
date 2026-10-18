"""Business-context provision for the evidence pack.

The evidence pack's `business_context` field is filled by a pluggable
provider so that the *content* (mapping table, vector store, LLM, etc.)
can change while the *interface* stays stable. This is the key
abstraction enabling firm-style internal-knowledge integration without
touching the explanation pipeline:

    StaticMappingProvider  → in-repo YAML / Python config (this file)
    RagProvider            → OpenAI embeddings + cosine-similarity
                              retrieval over a document corpus
    CompositeProvider      → static skeleton + RAG-retrieved excerpts

Importantly: every provider returns a *structured* dict (NOT free-form
prose). The LLM narrator may only cite fields, never invent business
content. The "narrator not judge" boundary is preserved at the schema
level.
"""
from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# State → phase mapping (single source of truth)
# ---------------------------------------------------------------------------

# Maps each state in lifecycle_graph.STATES to a business "phase". States
# can belong to one or two phases; we pick the operational primary
# attribution. Repair / error states fall into "repair" or "terminal".
STATE_TO_PHASE: dict[str, str] = {
    "NEW":                "pre_execution",
    "EXECUTED":           "execution",
    "CAPTURED":           "capture",
    "ENRICHED":           "enrichment",
    "MATCHED":            "matching",
    "CLEARED":            "clearing",
    "SETTLED":            "settlement",
    "UNMATCHED_ERROR":    "matching",
    "FAILED_SETTLEMENT":  "settlement",
    "AMENDED":            "repair",
    "REJECTED":           "terminal_failure",
    "CANCELLED":          "terminal_failure",
}


# ---------------------------------------------------------------------------
# Phase → declarative business semantics
# ---------------------------------------------------------------------------

# Each phase has the same schema. Values are the project authors'
# interpretation of generic post-trade operations practice (drawn from
# Bakshi et al. 2021 and the ISDA CDM normative spec); they are NOT
# firm-specific. A deployment would replace this dict with
# its own internal mapping or a RagProvider over its own corpus.
PHASE_BUSINESS_SEMANTICS: dict[str, dict] = {
    "pre_execution": {
        "phase_purpose":
            "The trade has been entered into the system but has not yet "
            "been executed in the market.",
        "typical_operational_owner":
            "Front-office order management",
        "regulatory_relevance": [],
        "common_failure_causes": [
            "Order entry validation failure",
            "Insufficient pre-trade checks",
        ],
        "downstream_dependencies_blocked": [
            "Market execution",
            "All subsequent lifecycle stages",
        ],
    },
    "execution": {
        "phase_purpose":
            "The trade has been executed in the market and a fill has "
            "been received.",
        "typical_operational_owner":
            "Trading desk",
        "regulatory_relevance": [
            "MiFID II Article 25 (record-keeping)",
        ],
        "common_failure_causes": [
            "Venue routing error",
            "Order fill discrepancy",
        ],
        "downstream_dependencies_blocked": [
            "Internal booking",
            "All post-execution stages",
        ],
    },
    "capture": {
        "phase_purpose":
            "The trade has been booked into the firm's middle-office "
            "trade store with all economic terms recorded.",
        "typical_operational_owner":
            "Middle office trade capture team",
        "regulatory_relevance": [
            "MiFID II Article 25 (record-keeping)",
        ],
        "common_failure_causes": [
            "Trade booking field mismatch with execution venue",
            "Static-data lookup failure (counterparty / instrument)",
        ],
        "downstream_dependencies_blocked": [
            "Trade enrichment",
            "Counterparty matching",
        ],
    },
    "enrichment": {
        "phase_purpose":
            "Standing settlement instructions, fees, and counterparty "
            "reference data are appended to the trade record.",
        "typical_operational_owner":
            "Middle office reference-data team",
        "regulatory_relevance": [],
        "common_failure_causes": [
            "Missing or stale standing settlement instruction (SSI)",
            "Counterparty reference-data mismatch",
            "Fee schedule not configured for instrument or venue",
        ],
        "downstream_dependencies_blocked": [
            "Counterparty matching",
            "Clearing",
        ],
    },
    "matching": {
        "phase_purpose":
            "The trade is matched against the counterparty's trade record. "
            "Both sides must agree on economic terms before clearing.",
        "typical_operational_owner":
            "Middle office reconciliation team",
        "regulatory_relevance": [
            "MiFID II Article 25 (transaction record-keeping)",
            "EMIR Article 9 (trade reporting)",
        ],
        "common_failure_causes": [
            "Counterparty SSI mismatch",
            "Trade size or price field discrepancy",
            "Settlement currency or date misalignment",
        ],
        "downstream_dependencies_blocked": [
            "Clearing",
            "Settlement at the intended date",
        ],
    },
    "clearing": {
        "phase_purpose":
            "The trade has been accepted by the clearing house (CCP) and "
            "novated to the CCP as central counterparty.",
        "typical_operational_owner":
            "Clearing operations team",
        "regulatory_relevance": [
            "EMIR Article 4 (clearing obligation)",
        ],
        "common_failure_causes": [
            "CCP eligibility check failure",
            "Margin call on initial deposit",
            "Product not eligible for clearing at this CCP",
        ],
        "downstream_dependencies_blocked": [
            "Settlement",
        ],
    },
    "settlement": {
        "phase_purpose":
            "Funds and securities are exchanged between the counterparties "
            "according to the agreed settlement date.",
        "typical_operational_owner":
            "Settlement operations team",
        "regulatory_relevance": [
            "T+1 / T+2 settlement cycle compliance",
            "EMIR daily reporting on settlement status",
        ],
        "common_failure_causes": [
            "Insufficient funds at settlement bank",
            "Securities not in custodian account at value date",
            "Cross-border settlement-rail failure",
        ],
        "downstream_dependencies_blocked": [
            "Trade lifecycle close-out",
        ],
    },
    "repair": {
        "phase_purpose":
            "A previously failed step has been amended and the trade "
            "re-enters the lifecycle for a second attempt.",
        "typical_operational_owner":
            "Middle office exception management",
        "regulatory_relevance": [
            "MiFID II Article 25 (record-keeping of amendments)",
        ],
        "common_failure_causes": [
            "Repair attempt did not address the underlying mismatch",
            "Amendment introduced a new field discrepancy",
        ],
        "downstream_dependencies_blocked": [
            "Resumption of normal lifecycle progression",
        ],
    },
    "terminal_failure": {
        "phase_purpose":
            "The trade has been terminally rejected or cancelled and "
            "will not progress further.",
        "typical_operational_owner":
            "Compliance / surveillance",
        "regulatory_relevance": [
            "EMIR Article 9 (cancelled-trade reporting)",
        ],
        "common_failure_causes": [
            "Pre-trade compliance check failed",
            "Counterparty cancellation request",
            "Discrepancy could not be reconciled",
        ],
        "downstream_dependencies_blocked": [],
    },
}


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

@dataclass
class BusinessContext:
    """Structured context attached to an evidence pack.

    All fields are *citable* by the LLM under the same hard-rule regime
    as the rest of the pack. The narrator cites e.g.
    `business_context.phase_purpose` rather than inventing prose.
    """
    current_lifecycle_phase: str
    phase_purpose: str
    typical_operational_owner: str
    regulatory_relevance: list[str]
    common_failure_causes: list[str]
    downstream_dependencies_blocked: list[str]
    # RAG-only: free-form excerpts retrieved from a corpus, each with
    # source attribution. Empty for static providers.
    retrieved_excerpts: list[dict] = None  # type: ignore[assignment]

    def to_dict(self) -> dict:
        d = {
            "current_lifecycle_phase": self.current_lifecycle_phase,
            "phase_purpose": self.phase_purpose,
            "typical_operational_owner": self.typical_operational_owner,
            "regulatory_relevance": list(self.regulatory_relevance),
            "common_failure_causes": list(self.common_failure_causes),
            "downstream_dependencies_blocked": list(self.downstream_dependencies_blocked),
        }
        if self.retrieved_excerpts:
            d["retrieved_excerpts"] = list(self.retrieved_excerpts)
        return d


class BusinessContextProvider(ABC):
    """Contract for any provider that fills the pack.business_context field.

    Implementations may use static config, retrieval-augmented generation
    over a document corpus, or any combination thereof. The interface is
    deliberately minimal so firm-internal data can be wired in by replacing
    the provider, with no other code change.
    """

    @abstractmethod
    def context_for(self, trace: list[str],
                    fired_signals: list[str]) -> BusinessContext:
        """Return business context for a trace and its fired signals."""


# ---------------------------------------------------------------------------
# StaticMappingProvider
# ---------------------------------------------------------------------------

class StaticMappingProvider(BusinessContextProvider):
    """Maps the *last informative state* of the trace to a phase, then
    looks up the declarative semantics dict.

    "Last informative state" = the last state in the trace whose phase
    is not terminal. We pick this rather than the literal last state
    because the operational analyst cares about *where the trade got
    stuck*, not about the post-failure terminal label.
    """

    def context_for(self, trace: list[str],
                    fired_signals: list[str]) -> BusinessContext:
        phase = self._infer_phase(trace)
        semantics = PHASE_BUSINESS_SEMANTICS.get(phase, {
            "phase_purpose": "Unknown phase",
            "typical_operational_owner": "Unknown",
            "regulatory_relevance": [],
            "common_failure_causes": [],
            "downstream_dependencies_blocked": [],
        })
        return BusinessContext(
            current_lifecycle_phase=phase,
            phase_purpose=semantics["phase_purpose"],
            typical_operational_owner=semantics["typical_operational_owner"],
            regulatory_relevance=list(semantics["regulatory_relevance"]),
            common_failure_causes=list(semantics["common_failure_causes"]),
            downstream_dependencies_blocked=list(
                semantics["downstream_dependencies_blocked"]
            ),
            retrieved_excerpts=None,
        )

    @staticmethod
    def _infer_phase(trace: list[str]) -> str:
        """Walk the trace from the end and pick the last non-terminal phase.

        Falls back to "terminal_failure" if every state is terminal, and
        to "pre_execution" if the trace is empty.
        """
        if not trace:
            return "pre_execution"
        for state in reversed(trace):
            phase = STATE_TO_PHASE.get(state, "pre_execution")
            if phase != "terminal_failure":
                return phase
        return "terminal_failure"


# ---------------------------------------------------------------------------
# RagProvider — retrieval-augmented business context
# ---------------------------------------------------------------------------

# RagProvider was originally below the static provider; its imports
# have been hoisted to the top of the file.


class RagProvider(BusinessContextProvider):
    """Retrieval-augmented context provider.

    Reads a corpus of plain-text or markdown documents, embeds them
    with the OpenAI embedding API, and at query time retrieves the
    top-K most-similar chunks to the trace+signals query. Returns the
    same `BusinessContext` schema as `StaticMappingProvider` but with
    the `retrieved_excerpts` field populated.

    The structured fields (phase, owner, regulatory_relevance, etc.)
    still come from the static mapping — RAG augments rather than
    replaces them. This is the "composite" pattern in practice: static
    skeleton, RAG body. A deployment would (a) replace the static
    mapping with its own and (b) point the RagProvider at its internal
    document store.

    Design notes
    ------------
    - Embeddings are cached on disk by SHA-256 of the chunk text, so
      repeated calls don't re-embed the same content.
    - Cosine similarity is computed in pure Python (no numpy required).
    - For the workshop paper this is a PoC; a production deployment
      would use FAISS or pgvector.
    """

    def __init__(self,
                 corpus_paths: Iterable[Path],
                 embedding_model: str = "text-embedding-3-small",
                 top_k: int = 3,
                 chunk_size: int = 800,
                 chunk_overlap: int = 100,
                 client: object | None = None,
                 cache_dir: Path | None = None):
        from openai import OpenAI

        self.embedding_model = embedding_model
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = client or OpenAI()
        self.cache_dir = (
            Path(cache_dir) if cache_dir
            else Path(__file__).resolve().parent.parent / ".cache" / "embeddings"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Static fallback for the structured fields.
        self._static = StaticMappingProvider()

        # Build the index.
        self.chunks: list[dict] = []
        for path in corpus_paths:
            text = Path(path).read_text(encoding="utf-8")
            for i, chunk in enumerate(self._split(text)):
                self.chunks.append({
                    "source": Path(path).name,
                    "chunk_index": i,
                    "text": chunk,
                })
        # Embed all chunks (with cache).
        for c in self.chunks:
            c["embedding"] = self._embed(c["text"])

    def _split(self, text: str) -> list[str]:
        """Simple character-window splitter with overlap."""
        out: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            out.append(text[i: i + self.chunk_size])
            i += self.chunk_size - self.chunk_overlap
        return [c.strip() for c in out if c.strip()]

    def _embed(self, text: str) -> list[float]:
        key = hashlib.sha256(
            (self.embedding_model + "\0" + text).encode("utf-8")
        ).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        response = self.client.embeddings.create(  # type: ignore[attr-defined]
            model=self.embedding_model, input=text,
        )
        vec = list(response.data[0].embedding)
        cache_file.write_text(json.dumps(vec), encoding="utf-8")
        return vec

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def context_for(self, trace: list[str],
                    fired_signals: list[str]) -> BusinessContext:
        # Structured fields from the static provider (skeleton).
        base = self._static.context_for(trace, fired_signals)
        # Build the query text from the trace + fired signals.
        query = (
            "Trade lifecycle phase: " + base.current_lifecycle_phase + ". " +
            "Signals fired: " + ", ".join(fired_signals) + ". " +
            "Trace: " + " -> ".join(trace) + ". " +
            "What operational or compliance context applies?"
        )
        q_vec = self._embed(query)
        scored = [
            (self._cosine(q_vec, c["embedding"]), c)
            for c in self.chunks
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        retrieved = []
        for score, c in scored[: self.top_k]:
            retrieved.append({
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "score": round(score, 4),
                "text": c["text"][:400],  # truncate for prompt budget
            })
        base.retrieved_excerpts = retrieved
        return base


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def default_provider() -> BusinessContextProvider:
    """The default provider used by the rest of the codebase.

    Returns the static mapping. Tests and `experiments/` scripts may
    override with a different provider (RAG, composite, etc.) at the
    call site.
    """
    return StaticMappingProvider()
