"""The engineered showcase corpus, questions, and gold answer key.

This is the demo's "all wrong -> all right" substrate. Every document and question is
constructed so that a *specific* broken parameter is the reason a fact is missed or a wrong
claim slips in. That makes the optimizer's textual gradients legible: each fix maps to one
named failure mode.

Failure modes engineered in:
  * BURIED FACT (chunk_chars / candidate_count): gold facts sit deep inside a long, padded
    document. As one giant chunk its query relevance is diluted, so short distractors
    outrank it; a narrow candidate pool never fetches it.
  * DUPLICATE DOMINATION (mmr_*): a common fact is repeated across near-identical docs that
    crowd a rare gold fact out of a small budget until the diversity pass collapses them.
  * DISTRACTOR (rerank_min_score / grounded prompt): a superseded doc carries a plausible
    but wrong fact; lexical retrieval loves its keyword density.
  * LEXICAL/SEMANTIC GAP (retrieval_mode / hybrid_alpha): a gold fact is phrased with
    different vocabulary than the query, so only semantic (vector) retrieval bridges it.

Facts are detected mechanically: a gold/trap fact counts as "present" when its lowercase
`trigger` substring appears in the finally-selected, length-clipped context.
"""

from __future__ import annotations

from dataclasses import dataclass

# A tiny hand-built synonym map used ONLY by the semantic ("vector") embedder, never by
# the lexical (bm25) path. It simulates what real sentence embeddings buy you: bridging
# vocabulary mismatch between a query and a relevantly-phrased passage.
SYNONYMS: dict[str, list[str]] = {
    "reset": ["recovery", "recover", "restore"],
    "password": ["credential", "credentials", "passphrase"],
    "forgotten": ["lost", "forgot"],
    "price": ["cost", "pricing", "fee"],
    "flight": ["airborne", "endurance"],
    "authentication": ["auth", "login", "signin"],
    "administrator": ["admin", "privileged"],
}


@dataclass(frozen=True)
class Fact:
    text: str       # human-readable claim, shown in the UI
    trigger: str    # lowercase substring whose presence in context means "surfaced"
    doc: str        # the document that legitimately supports it


@dataclass(frozen=True)
class Question:
    id: str
    query: str
    gold: tuple[Fact, ...]   # facts a correct answer must surface (recall)
    traps: tuple[Fact, ...]  # plausible-but-wrong claims; surfacing one hurts precision
    failure: str             # the knob this question is designed to stress
    split: str = "train"     # "train" or "holdout"


# --- Documents --------------------------------------------------------------------

# Filler used to pad the long spec doc so a buried fact's relevance is genuinely diluted.
_PAD = (
    "The Helios platform is engineered for industrial reliability across a wide range of "
    "operating conditions. Components are validated against thermal, vibration, and humidity "
    "tolerances. Routine maintenance intervals are documented in the service handbook. "
    "Firmware updates are delivered over the air and verified with signed manifests. "
    "Operators should review the pre-mission checklist before every sortie and log results. "
)


def _long_spec() -> str:
    head = (
        "Helios H2 Technical Specification (Revision 2026.2). "
        "This document describes the airframe, avionics, payload bays, and operational "
        "envelope of the Helios H2 industrial drone. "
    )
    # Bury two gold facts deep in the middle of a long, otherwise-unrelated body.
    buried = (
        "Under nominal load and standard atmospheric conditions, the Helios H2 drone has a "
        "maximum flight time of 47 minutes on a single charge. "
        "The Helios H2 ships with a standard warranty period of 24 months from date of "
        "delivery, covering airframe and avionics defects. "
    )
    return head + (_PAD * 6) + buried + (_PAD * 6)


DOCS: dict[str, str] = {
    # Long doc with two buried gold facts (flight time, warranty).
    "d_specs_long": _long_spec(),
    # Short, keyword-dense distractor about the PREVIOUS model — a precision trap.
    "d_specs_old": (
        "Helios H1 quick facts: flight time flight time endurance 32 minutes. The legacy "
        "Helios H1 drone maximum flight time was 32 minutes. Warranty 12 months."
    ),
    # The RARE gold fact. It shares fewer query terms than the near-duplicate access-policy
    # docs (no 'corporate systems'), so it reranks just below them and is crowded out of the
    # budget until MMR collapses the duplicates. (The duplicates themselves — d_policy_0..N —
    # are generated below so they outnumber the budget; that is what makes the diversity pass
    # necessary rather than merely helpful.)
    "d_policy_admin": (
        "Privileged Access Addendum: Administrator accounts must also present a hardware "
        "security token when they authenticate, in addition to their account credentials."
    ),
    # Pricing: current (gold) vs superseded 2023 (trap). Similar keywords.
    "d_pricing_current": (
        "Pricing (effective 2026): The current Enterprise plan costs $1,499 per month, billed "
        "annually. This is the current Enterprise plan price."
    ),
    "d_pricing_2023": (
        "Archived pricing sheet 2023: Enterprise plan Enterprise plan price $999 per month. "
        "Pricing pricing plan enterprise enterprise."
    ),
    # Password reset answer phrased semantically far from the query (LEXICAL/SEMANTIC GAP).
    "d_support_recovery": (
        "Account Help: If you have lost access to your login, credential recovery is performed "
        "through the self-service portal. Visit the self-service portal to restore access."
    ),
    # On-call: nuanced gold (secondary at night) vs oversimplified trap (primary).
    "d_oncall": (
        "On-call Runbook: During a Sev1 incident occurring at night, the secondary on-call "
        "engineer is paged first, because the primary covers daytime business hours only."
    ),
    "d_oncall_summary": (
        "Team wiki blurb: For incidents, the primary on-call engineer is paged first. "
        "On-call on-call incident paged primary."
    ),
    # Unrelated noise docs to add realistic retrieval pressure.
    "d_noise_office": (
        "Office facilities: The break room is restocked on Mondays. Visitor badges are issued "
        "at the front desk. Parking permits renew quarterly."
    ),
    "d_noise_brand": (
        "Brand guidelines: The Helios logo must keep clear-space equal to the cap height. "
        "Primary color is slate blue. Do not stretch the wordmark."
    ),
    "d_noise_legal": (
        "Legal notice: All trademarks are property of their respective owners. This document "
        "is provided without warranty of any kind for informational purposes."
    ),
}

# Generate N near-identical access-policy docs (the DUPLICATE DOMINATION trap). Variation is
# kept to one connective + one verb so any two stay above the MMR similarity threshold and
# collapse to one, while the gold admin doc is dissimilar enough to survive.
_DUP_COUNT = 9
for _i in range(_DUP_COUNT):
    _conn = ("using", "with", "via")[_i % 3]
    _verb = ("logged", "recorded", "audited")[_i % 3]
    DOCS[f"d_policy_{_i}"] = (
        f"Corporate Access Policy (copy {_i}): Administrators authenticate to corporate "
        f"systems {_conn} their account credentials. Administrator authentication to corporate "
        f"systems is {_verb} and reviewed for compliance."
    )


# --- Questions + gold answer key --------------------------------------------------

QUESTIONS: tuple[Question, ...] = (
    Question(
        id="q_flight_time",
        query="What is the maximum flight time of the Helios H2 drone?",
        gold=(Fact("Maximum flight time is 47 minutes.", "47 minutes", "d_specs_long"),),
        traps=(Fact("Flight time is 32 minutes (that is the H1).", "32 minutes", "d_specs_old"),),
        failure="buried fact (chunk_chars / candidate_count) + distractor",
        split="train",
    ),
    Question(
        id="q_admin_auth",
        query="How must administrators authenticate to corporate systems?",
        gold=(Fact("Administrators must present a hardware security token.",
                   "hardware security token", "d_policy_admin"),),
        traps=(),
        failure="duplicate domination (mmr) + candidate_count",
        split="train",
    ),
    Question(
        id="q_enterprise_price",
        query="What is the current price of the Enterprise plan?",
        gold=(Fact("Current Enterprise plan is $1,499 per month.", "$1,499", "d_pricing_current"),),
        traps=(Fact("Enterprise plan is $999 per month (archived 2023).", "$999", "d_pricing_2023"),),
        failure="distractor (rerank_min_score / grounded prompt)",
        split="train",
    ),
    Question(
        id="q_password_reset",
        query="How does a user reset a forgotten password?",
        gold=(Fact("Use the self-service portal for credential recovery.",
                   "self-service portal", "d_support_recovery"),),
        traps=(),
        failure="lexical/semantic gap (retrieval_mode / hybrid_alpha)",
        split="train",
    ),
    # --- Held-out: must improve from the SAME params, guarding against judge-gaming. ---
    Question(
        id="q_warranty",
        query="What is the warranty period for the Helios H2?",
        gold=(Fact("Warranty period is 24 months.", "24 months", "d_specs_long"),),
        traps=(Fact("Warranty is 12 months (that is the H1).", "12 months", "d_specs_old"),),
        failure="buried fact (chunk_chars / candidate_count) + distractor",
        split="holdout",
    ),
    Question(
        id="q_oncall_night",
        query="Who is paged first during a Sev1 incident at night?",
        gold=(Fact("The secondary on-call engineer is paged first at night.",
                   "secondary on-call", "d_oncall"),),
        traps=(Fact("The primary on-call engineer is paged first.",
                    "primary on-call", "d_oncall_summary"),),
        failure="distractor + grounding",
        split="holdout",
    ),
)


def train_questions() -> tuple[Question, ...]:
    return tuple(q for q in QUESTIONS if q.split == "train")


def holdout_questions() -> tuple[Question, ...]:
    return tuple(q for q in QUESTIONS if q.split == "holdout")
