"""The loss function: a REAL LLM judge grades each answer against the answer key.

The judge is a strong model. It reads the question, the pipeline's answer, the required facts
(the ground-truth key), and the known-incorrect claims, and decides — by meaning, robust to
phrasing — how much of the required information the answer conveys (recall) and whether it
asserts anything wrong (precision). It does NOT see the retrieval internals.

What flows onward to the optimizer is only the score + a category (see optimizer.build_logs) —
never the required facts themselves, so the optimizer is graded, not handed the answer.

Composite quality is F1-primary with a hallucination penalty. Loss = 1 - mean(composite).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .llm import model_for
from .pipeline import QuestionTrace

HALLUCINATION_PENALTY = 0.15

_JUDGE_SYSTEM = (
    "You are a strict, fair grader of question-answering. Compare an ANSWER to a list of "
    "REQUIRED FACTS and a list of KNOWN-INCORRECT CLAIMS. Judge by meaning, not exact wording "
    "(e.g. 'forty-seven minutes' satisfies '47 minutes'). An answer that abstains or says it "
    "cannot find the information conveys no required facts and makes no wrong claims. "
    "Reply with ONE JSON object and nothing else."
)


@dataclass
class QuestionGrade:
    question_id: str
    recall: float
    precision: float
    f1: float
    composite: float
    category: str                                  # correct | incomplete | unsupported-claim | ...
    wrong_claims: list[str] = field(default_factory=list)


def _judge_prompt(question, answer: str) -> str:
    required = "\n".join(f"  - {g.text}" for g in question.gold) or "  (none)"
    incorrect = "\n".join(f"  - {t.text}" for t in question.traps) or "  (none)"
    return (
        f"Question: {question.query}\n\n"
        f"Answer to grade:\n{answer or '(empty answer)'}\n\n"
        f"REQUIRED FACTS (a correct answer must convey each; match by meaning):\n{required}\n\n"
        f"KNOWN-INCORRECT CLAIMS (a good answer must NOT assert any of these):\n{incorrect}\n\n"
        'Return JSON: {"required_present": <int count of required facts the answer correctly '
        'conveys>, "wrong_claims": [<incorrect or unsupported claims the answer actually makes>], '
        '"category": "correct" | "incomplete" | "unsupported-claim" | "incomplete+unsupported"}'
    )


def grade_question(trace: QuestionTrace, backend, cache) -> QuestionGrade:
    q = trace.question
    n_req = len(q.gold) or 1
    sig = hashlib.sha1(trace.answer_text.encode("utf-8")).hexdigest()[:16]

    def _call() -> dict:
        try:
            return backend.complete_json(_JUDGE_SYSTEM, _judge_prompt(q, trace.answer_text),
                                         model=model_for("judge"), max_tokens=500)
        except Exception as exc:  # noqa: BLE001 — a flaky judge call shouldn't kill the run
            return {"required_present": 0, "wrong_claims": [], "category": f"judge-error: {exc}"}

    data = cache.get_or_compute("judge", sig, q.id, _call)

    present = max(0, min(n_req, int(data.get("required_present", 0))))
    wrong = [str(w) for w in data.get("wrong_claims", []) if str(w).strip()]
    recall = present / n_req
    precision = present / (present + len(wrong)) if (present + len(wrong)) else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
    composite = max(0.0, f1 - HALLUCINATION_PENALTY * len(wrong))
    return QuestionGrade(q.id, recall, precision, f1, composite,
                         str(data.get("category", "")), wrong)


@dataclass
class GradeReport:
    grades: list[QuestionGrade]
    mean_recall: float
    mean_precision: float
    mean_f1: float
    mean_composite: float
    loss: float

    def by_id(self) -> dict[str, QuestionGrade]:
        return {g.question_id: g for g in self.grades}


def grade(traces: dict[str, QuestionTrace], backend, cache) -> GradeReport:
    grades = [grade_question(t, backend, cache) for t in traces.values()]
    n = len(grades) or 1
    return GradeReport(
        grades=grades,
        mean_recall=sum(g.recall for g in grades) / n,
        mean_precision=sum(g.precision for g in grades) / n,
        mean_f1=sum(g.f1 for g in grades) / n,
        mean_composite=sum(g.composite for g in grades) / n,
        loss=1.0 - sum(g.composite for g in grades) / n,
    )
