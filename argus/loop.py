"""The convergence driver — one propose -> apply -> re-run -> re-grade epoch at a time.

Two separate signals, the standard ML split:
  * TRAIN loss steers optimization — a trust region applies a proposed move only if it does not
    worsen train loss (the optimizer is descending train).
  * VALIDATION (held-out) loss selects and stops — we keep the **best-on-validation checkpoint**
    and early-stop when val loss hasn't improved for `patience` epochs. The shipped result is the
    best-val checkpoint, NOT the final epoch, so a late epoch that overfits train (val regresses)
    is detected and discarded rather than returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .cache import StageCache
from .dataset import holdout_questions, train_questions
from .judge import GradeReport, grade
from .optimizer import Proposal
from .params import Params
from .pipeline import run_pipeline

EPS = 1e-9


@dataclass
class RoundRecord:
    round: int
    proposal: Proposal
    accepted: bool
    params: Params
    train: GradeReport
    holdout: GradeReport
    moved: list[str] = field(default_factory=list)


@dataclass
class Result:
    records: list[RoundRecord]
    initial_params: Params
    final_params: Params            # the best-on-validation checkpoint (what we ship)
    initial_train: GradeReport
    final_train: GradeReport
    initial_holdout: GradeReport
    final_holdout: GradeReport
    best_epoch: int                 # epoch the shipped checkpoint came from (0 = the start)
    cache_hits: int
    cache_misses: int


def run_optimization(
    optimizer,
    start: Params,
    *,
    backend,
    max_rounds: int = 16,
    patience: int = 2,
    use_cache: bool = True,
    on_start: Callable[[GradeReport, GradeReport], None] | None = None,
    on_round: Callable[[RoundRecord], None] | None = None,
) -> Result:
    cache = StageCache(enabled=use_cache)
    train_q = train_questions()
    holdout_q = holdout_questions()

    def evaluate(params: Params) -> tuple[GradeReport, GradeReport, dict]:
        traces = run_pipeline(params, train_q, cache, backend)
        hold = grade(run_pipeline(params, holdout_q, cache, backend), backend, cache)
        return grade(traces, backend, cache), hold, traces

    current = start
    cur_train, cur_hold, cur_traces = evaluate(current)
    initial_train, initial_hold = cur_train, cur_hold
    if on_start:
        on_start(initial_train, initial_hold)

    # Best-on-validation checkpoint (start with the broken baseline). Optimization descends
    # TRAIN loss; selection + early stopping ride VALIDATION loss.
    best = {"epoch": 0, "loss": cur_hold.loss, "params": current,
            "train": cur_train, "hold": cur_hold}
    val_stale = 0  # epochs since validation loss last strictly improved

    records: list[RoundRecord] = []
    history: list[dict] = []
    last_update: dict | None = None

    for r in range(1, max_rounds + 1):
        proposal: Proposal = optimizer.propose(current, cur_traces, cur_train, history)

        # No-op or a repeat of the last move => the optimizer is done.
        if not proposal.updates or proposal.updates == last_update:
            rec = RoundRecord(r, proposal, False, current, cur_train, cur_hold, [])
            records.append(rec)
            if on_round:
                on_round(rec)
            break

        candidate = current.with_updates(**proposal.updates)
        if candidate == current:
            rec = RoundRecord(r, proposal, False, current, cur_train, cur_hold, [])
            records.append(rec)
            if on_round:
                on_round(rec)
            break

        cand_train, cand_hold, cand_traces = evaluate(candidate)
        accepted = cand_train.loss <= cur_train.loss + EPS  # trust region on TRAIN loss
        moved = sorted(proposal.updates.keys())

        if accepted:
            current, cur_train, cur_hold, cur_traces = candidate, cand_train, cand_hold, cand_traces

        # Checkpoint + early stopping on VALIDATION loss. Update the checkpoint when val
        # strictly improves, or ties val but lowers train (a tidier point at equal generalization).
        improved_val = accepted and cur_hold.loss < best["loss"] - EPS
        tie_better_train = (accepted and abs(cur_hold.loss - best["loss"]) <= EPS
                            and cur_train.loss < best["train"].loss - EPS)
        if improved_val or tie_better_train:
            best = {"epoch": r, "loss": cur_hold.loss, "params": current,
                    "train": cur_train, "hold": cur_hold}
        val_stale = 0 if improved_val else val_stale + 1

        last_update = dict(proposal.updates)
        history.append({"round": r, "loss": cur_train.loss, "accepted": accepted, "moved": moved})
        rec = RoundRecord(r, proposal, accepted, current, cur_train, cur_hold, moved)
        records.append(rec)
        if on_round:
            on_round(rec)

        if val_stale >= patience:  # early stopping on validation
            break

    return Result(
        records=records,
        initial_params=start,
        final_params=best["params"],
        initial_train=initial_train,
        final_train=best["train"],
        initial_holdout=initial_hold,
        final_holdout=best["hold"],
        best_epoch=best["epoch"],
        cache_hits=cache.hits,
        cache_misses=cache.misses,
    )
