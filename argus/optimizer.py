"""The optimizer — Argus's "backward pass". The brain is always an LLM.

It is given **symptom-level logs** only — the question, the system's answer, the grader's
score + category, and the retrieval log (mode, how many retrieved, rerank scores, the snippets
that were retrieved). It is deliberately NOT told which knob to change, where any "correct"
fact lives, or what the right answer is. It must infer cause -> knob from the logs, like an
engineer reading a trace. There is no rule-based fallback by design.

`build_logs`/`bundle_text` are pure instrumentation (no decisions). `LLMOptimizer` talks to any
backend exposing `complete_json`. It returns a `Proposal`; the loop applies it under a trust
region (accept iff loss does not worsen).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .judge import GradeReport
from .llm import model_for
from .params import KNOBS, Params, STAGE_ORDER
from .pipeline import QuestionTrace

_SNIPPET_PREVIEW = 220
_MAX_SNIPPETS = 6


@dataclass
class Proposal:
    updates: dict
    rationale: str
    reflection: str = ""
    confidence: float = 0.5
    source: str = "llm"
    logs: list = field(default_factory=list)   # the symptom logs the optimizer read this round

    def cheapest_stage(self) -> str:
        if not self.updates:
            return "—"
        return min((KNOBS[k].stage for k in self.updates if k in KNOBS),
                   key=lambda s: STAGE_ORDER.index(s)).value


def _coerce(name: str, value):
    knob = KNOBS[name]
    if knob.kind == "int":
        return int(round(float(value)))
    if knob.kind == "float":
        return float(value)
    if knob.kind == "bool":
        return value in (True, "true", "True", 1, "1")
    return str(value)


# --- Symptom-level logs (the optimizer's only observation) ------------------------

def build_logs(traces: dict[str, QuestionTrace], report: GradeReport) -> list[dict]:
    by_id = report.by_id()
    out = []
    for qid, tr in traces.items():
        g = by_id[qid]
        snippets = [(doc, txt[:_SNIPPET_PREVIEW].replace("\n", " "))
                    for doc, txt in tr.selected_snippets[:_MAX_SNIPPETS]]
        out.append({
            "qid": qid,
            "question": tr.question.query,
            "answer": tr.answer_text,
            "recall": round(g.recall, 2),
            "precision": round(g.precision, 2),
            "category": g.category,
            "wrong_claims": g.wrong_claims,
            "retrieval_mode": tr.retrieval_mode,
            "num_retrieved": len(tr.retrieved),
            "rerank_scores": [s for _, s, _ in tr.reranked][:8],
            "snippets": snippets,
        })
    return out


def bundle_text(params: Params, logs: list[dict], report: GradeReport, history: list[dict]) -> str:
    lines = ["CURRENT PARAMETERS:"]
    for name, knob in KNOBS.items():
        val = getattr(params, name)
        rng = f"[{knob.low}..{knob.high}]" if knob.low is not None else (
            "/".join(knob.choices) if knob.choices else knob.kind)
        lines.append(f"  {name} = {val!r}  {rng} (cost={knob.cost}) — {knob.description}")
    lines.append(f"\nLOSS = {report.loss:.3f}  (meanF1={report.mean_f1:.3f} "
                 f"recall={report.mean_recall:.3f} precision={report.mean_precision:.3f})")
    lines.append("\nPER-QUESTION LOGS (symptom-level — infer the cause yourself):")
    for d in logs:
        lines.append(f"  [{d['qid']}] recall={d['recall']} precision={d['precision']} "
                     f"category={d['category']}")
        lines.append(f"     question: {d['question']}")
        lines.append(f"     answer: {d['answer'][:300]!r}")
        if d["wrong_claims"]:
            lines.append(f"     grader flagged wrong/unsupported claims: {d['wrong_claims']}")
        lines.append(f"     retrieval: mode={d['retrieval_mode']} retrieved={d['num_retrieved']} "
                     f"rerank_scores={d['rerank_scores']}")
        if d["snippets"]:
            for doc, prev in d["snippets"]:
                lines.append(f"        snippet [{doc}]: {prev!r}")
        else:
            lines.append("        snippets: (nothing was retrieved)")
    if history:
        lines.append("\nHISTORY (most recent last):")
        for h in history[-6:]:
            lines.append(f"  epoch {h['round']}: loss={h['loss']:.3f} "
                         f"accepted={h['accepted']} moved={h.get('moved', [])}")
    return "\n".join(lines)


# --- LLM optimizer ----------------------------------------------------------------

_SYSTEM = """You are Argus, an optimizer that tunes a Retrieval-Augmented-Generation pipeline by
adjusting its parameters to reduce a loss (1 - composite F1, measured by an independent grader).

You are given SYMPTOM-LEVEL LOGS only: the question, the system's answer, the grader's score
(recall/precision) and category, and the retrieval log (mode, how many docs retrieved, rerank
scores, and the snippets that were retrieved). You are NOT told which knob to change, where any
correct information lives, or what the right answer is — you must infer the cause from the logs.

How to reason:
- Low recall + the answer missing information + few/low-score retrieved snippets, or snippets that
  don't contain what the question asks -> retrieval is too narrow or the wrong mode (consider
  candidate_count, retrieval_mode/hybrid_alpha, rerank_result_count, mmr, chunk_chars).
- The grader flags wrong/unsupported claims (low precision) -> the answer is asserting things the
  context doesn't support; consider grounding the synthesis_prompt (free text — rewrite it to
  instruct using only the context and abstaining otherwise) or raising rerank_min_score.
- Respect each knob's declared range/choices. Make decisive but bounded moves. Prefer cheap
  (downstream) knobs first unless an expensive one is the root cause. Do not re-propose a move the
  history shows was tried and reverted.

Reply with ONE JSON object:
{"reflection": "...", "analysis": "...",
 "gradients": [{"param": "<knob>", "proposed": <value>, "reason": "..."}],
 "confidence": 0.0-1.0}"""


class LLMOptimizer:
    source = "llm"

    def __init__(self, backend) -> None:
        self.backend = backend

    def propose(self, params: Params, traces: dict[str, QuestionTrace],
                report: GradeReport, history: list[dict]) -> Proposal:
        logs = build_logs(traces, report)   # the symptom-level view the optimizer is allowed to see
        user = bundle_text(params, logs, report, history)
        user += "\n\nPropose the next parameter update as JSON."
        try:
            data = self.backend.complete_json(_SYSTEM, user, model=model_for("optimizer"))
        except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash the loop
            return Proposal({}, f"LLM optimizer error: {exc}", source="llm-error", confidence=0.0, logs=logs)

        updates: dict = {}
        for grad in data.get("gradients", []):
            name = grad.get("param")
            if name in KNOBS and "proposed" in grad:
                try:
                    updates[name] = _coerce(name, grad["proposed"])
                except (TypeError, ValueError):
                    continue
        rationale = "; ".join(
            f"{grad.get('param')}->{grad.get('proposed')}: {grad.get('reason', '')}"
            for grad in data.get("gradients", [])
        ) or data.get("analysis", "")
        return Proposal(updates, rationale, reflection=data.get("reflection", ""),
                        confidence=float(data.get("confidence", 0.5)), source=self.source, logs=logs)
