"""The parameter space — the "weights" Argus optimizes.

Every knob is declared once, with the pipeline *stage* it affects and a *cost* tier.
Two facts about a knob drive the whole optimizer:

  * its `stage` decides which cached forward-pass stages a change invalidates
    (a change to an MMR knob never re-runs the expensive chunk/embed work), and
  * its `cost` lets the optimizer prefer cheap moves before expensive ones.

The knob taxonomy mirrors a production RAG + enrichment pipeline 1:1:

    chunk_chars / chunk_overlap      -> vectorization splitter sizes
    retrieval_mode / hybrid_alpha    -> per-prompt retrieval + RRF fusion weight
    candidate_count                  -> first-stage fetch width (RAG_CANDIDATE_COUNT)
    rerank_result_count / _min_score -> cross-encoder top-k + noise floor
    mmr_enabled / _lambda / _sim     -> the diversity pass over reranked candidates
    doc_char_limit                   -> per-doc synthesis context slice
    synthesis_prompt                 -> the synthesis system prompt (a free-text weight)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum


class Stage(str, Enum):
    """Pipeline stages, in execution order. Used for cache invalidation."""

    CHUNK = "chunk"
    EMBED = "embed"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    MMR = "mmr"
    SYNTH = "synth"


# Execution order. A stage's output depends on its own knobs and every upstream stage's.
STAGE_ORDER: tuple[Stage, ...] = (
    Stage.CHUNK,
    Stage.EMBED,
    Stage.RETRIEVE,
    Stage.RERANK,
    Stage.MMR,
    Stage.SYNTH,
)

# Relative wall-clock cost of recomputing from a stage onward (used by the optimizer to
# prefer cheap knobs first). Maps directly to "cheap" reprobe vs "expensive" re-ingest.
STAGE_COST: dict[Stage, str] = {
    Stage.CHUNK: "expensive",   # rebuild chunks + re-embed + re-index
    Stage.EMBED: "expensive",   # re-embed + re-index
    Stage.RETRIEVE: "medium",   # re-query
    Stage.RERANK: "cheap",
    Stage.MMR: "cheap",
    Stage.SYNTH: "cheapest",
}


@dataclass(frozen=True)
class Knob:
    """A single tunable parameter declaration."""

    name: str
    stage: Stage
    kind: str  # "int" | "float" | "bool" | "choice" | "text"
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] | None = None
    description: str = ""

    @property
    def cost(self) -> str:
        return STAGE_COST[self.stage]


# The knob registry. Keys match `Params` field names exactly.
KNOBS: dict[str, Knob] = {
    "chunk_chars": Knob(
        "chunk_chars", Stage.CHUNK, "int", low=200, high=20000,
        description="Max characters per chunk. Too large dilutes a buried fact's relevance "
        "score among surrounding text; too small fragments context.",
    ),
    "chunk_overlap": Knob(
        "chunk_overlap", Stage.CHUNK, "int", low=0, high=400,
        description="Characters of overlap between adjacent chunks; guards facts that straddle "
        "a chunk boundary.",
    ),
    "retrieval_mode": Knob(
        "retrieval_mode", Stage.RETRIEVE, "choice", choices=("bm25", "vector", "hybrid"),
        description="Lexical (bm25), semantic (vector), or fused (hybrid) first-stage retrieval.",
    ),
    "hybrid_alpha": Knob(
        "hybrid_alpha", Stage.RETRIEVE, "float", low=0.0, high=1.0,
        description="Hybrid fusion weight: 0 = all lexical, 1 = all semantic.",
    ),
    "candidate_count": Knob(
        "candidate_count", Stage.RETRIEVE, "int", low=1, high=100,
        description="First-stage fetch width before reranking. Too small starves the reranker.",
    ),
    "rerank_result_count": Knob(
        "rerank_result_count", Stage.RERANK, "int", low=1, high=40,
        description="How many reranked chunks survive into synthesis (the doc budget).",
    ),
    "rerank_min_score": Knob(
        "rerank_min_score", Stage.RERANK, "float", low=0.0, high=1.0,
        description="Rerank relevance floor. Raise to drop distractors; too high starves recall.",
    ),
    "mmr_enabled": Knob(
        "mmr_enabled", Stage.MMR, "bool",
        description="Diversity pass that collapses near-duplicate chunks so one repeated fact "
        "can't monopolize the budget.",
    ),
    "mmr_lambda": Knob(
        "mmr_lambda", Stage.MMR, "float", low=0.0, high=1.0,
        description="MMR relevance/diversity trade-off: 1 = pure relevance, 0 = max diversity.",
    ),
    "mmr_sim_threshold": Knob(
        "mmr_sim_threshold", Stage.MMR, "float", low=0.5, high=1.0,
        description="Two chunks above this similarity are treated as duplicates.",
    ),
    "doc_char_limit": Knob(
        "doc_char_limit", Stage.SYNTH, "int", low=200, high=20000,
        description="Per-doc context slice handed to synthesis. Too small drops the tail.",
    ),
    "synthesis_prompt": Knob(
        "synthesis_prompt", Stage.SYNTH, "text",
        description="The synthesis system prompt — a free-text weight the optimizer rewrites. "
        "A grounded 'cite-or-refuse' prompt drops unsupported claims; a terse prompt does not.",
    ),
}


@dataclass(frozen=True)
class Params:
    """A concrete parameter assignment (one point in the search space)."""

    chunk_chars: int
    chunk_overlap: int
    retrieval_mode: str
    hybrid_alpha: float
    candidate_count: int
    rerank_result_count: int
    rerank_min_score: float
    mmr_enabled: bool
    mmr_lambda: float
    mmr_sim_threshold: float
    doc_char_limit: int
    synthesis_prompt: str

    def with_updates(self, **changes) -> "Params":
        """Return a clamped copy with the given knob changes applied."""
        merged = {**asdict(self), **changes}
        return clamp(Params(**merged))

    def to_dict(self) -> dict:
        return asdict(self)

    def stage_signature(self, stage: Stage) -> str:
        """A hash over every knob at or upstream of `stage` — the cache key for that stage."""
        cutoff = STAGE_ORDER.index(stage)
        relevant = {
            name: getattr(self, name)
            for name, knob in KNOBS.items()
            if STAGE_ORDER.index(knob.stage) <= cutoff
        }
        blob = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _clamp_number(value, knob: Knob):
    value = max(knob.low, min(knob.high, value))
    return int(round(value)) if knob.kind == "int" else float(value)


def clamp(params: Params) -> Params:
    """Project a parameter set back into legal ranges/choices. Idempotent."""
    changes: dict = {}
    for f in fields(params):
        knob = KNOBS[f.name]
        value = getattr(params, f.name)
        if knob.kind in ("int", "float"):
            changes[f.name] = _clamp_number(value, knob)
        elif knob.kind == "choice" and value not in (knob.choices or ()):
            changes[f.name] = knob.choices[0]  # type: ignore[index]
        elif knob.kind == "bool":
            changes[f.name] = bool(value)
    return replace(params, **changes)


# --- Reference points -------------------------------------------------------------

# A terse, ungrounded prompt: synthesis will parrot whatever was retrieved, distractors
# and all. The optimizer's job includes rewriting this into a grounded one.
BROKEN_PROMPT = "Answer the question."

# The deliberately-wrong starting point. Every knob is set to a value that the engineered
# corpus punishes: one giant chunk, lexical-only retrieval, a starved candidate pool, no
# rerank floor, no diversity pass, a clipped context, and a prompt that invites hallucination.
BROKEN_PARAMS = clamp(
    Params(
        chunk_chars=20000,
        chunk_overlap=0,
        retrieval_mode="bm25",
        hybrid_alpha=0.0,
        candidate_count=2,
        rerank_result_count=2,
        rerank_min_score=0.0,
        mmr_enabled=False,
        mmr_lambda=1.0,
        mmr_sim_threshold=0.99,
        doc_char_limit=600,
        synthesis_prompt=BROKEN_PROMPT,
    )
)


# A known-good prompt the optimizer can converge toward (it is NOT given this; it must
# discover that grounding the prompt raises precision). Kept here only for documentation
# and for the offline mock's prompt rewrite.
GROUNDED_PROMPT = (
    "You are a careful analyst. Answer ONLY using facts stated in the provided context. "
    "Cite each claim. If the context does not support a claim, omit it. Do not guess or "
    "include outside information."
)
