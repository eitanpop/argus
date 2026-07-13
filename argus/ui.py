"""Rich console UI — presents the run as a *machine-learning training job*.

Three things this view is built to make legible to an ML/AI audience (so it reads as a real
optimization system, not "we prompted an LLM"):

  (2) A training run: epochs, train loss + held-out **validation** loss side by side, ΔF1,
      a live loss curve, and an early-stopping line.
  (3) The LLM's job is *narrow*: an opening architecture panel shows the separation of
      concerns (search space | pipeline | judge | optimizer), and every epoch prints the
      optimizer's ENTIRE output — a small structured action — distinct from the system's
      measured response.
  (4) Prompt-as-parameter: when the optimizer rewrites the synthesis prompt, the UI shows the
      old->new text flagged as a free-text parameter that numeric optimization cannot touch.
"""

from __future__ import annotations

import dataclasses

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .judge import GradeReport
from .loop import Result, RoundRecord
from .params import BROKEN_PARAMS, KNOBS, Params

_BLOCKS = "▁▂▃▄▅▆▇█"

# Knobs whose value is free text (only an LLM optimizer can move these).
FREETEXT_KNOBS = {name for name, k in KNOBS.items() if k.kind == "text"}


def sparkline(values: list[float], lo: float = 0.0, hi: float = 1.0) -> str:
    if not values:
        return ""
    span = (hi - lo) or 1.0
    out = []
    for v in values:
        level = int((v - lo) / span * (len(_BLOCKS) - 1))
        out.append(_BLOCKS[max(0, min(len(_BLOCKS) - 1, level))])
    return "".join(out)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, str) and len(value) > 44:
        return value[:41] + "..."
    return str(value)


class Renderer:
    """Stateful per-epoch renderer (tracks previous params + the loss/F1 curves)."""

    def __init__(self, console: Console, optimizer_name: str) -> None:
        self.console = console
        self.prev = BROKEN_PARAMS
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.prev_f1 = 0.0
        self.optimizer_name = optimizer_name

    # (3) Make the scaffolding visible: the LLM is one bounded component.
    def architecture(self) -> None:
        n_total = len(KNOBS)
        n_text = len(FREETEXT_KNOBS)
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan")
        grid.add_column()
        grid.add_row("search space", f"{n_total} parameters "
                     f"({n_total - n_text} numeric/categorical + {n_text} free-text prompt)")
        grid.add_row("pipeline", "simulated retrieval + a real LLM answer-writer — the system under test")
        grid.add_row("judge", "a real LLM grading the answer vs a gold key — [bold]independent[/bold] of the optimizer")
        grid.add_row("optimizer", "a real LLM — emits one structured action per epoch from symptom-level logs")
        grid.add_row("objective", "minimize  1 − composite_F1  on train · validate on a held-out split")
        self.console.print(Panel(grid, title="Argus · a training run on a RAG pipeline",
                                 subtitle="the LLM is ONLY the optimizer", border_style="cyan"))

    def banner(self, initial_train: GradeReport, initial_val: GradeReport) -> None:
        self.train_losses.append(initial_train.loss)
        self.val_losses.append(initial_val.loss)
        self.prev_f1 = initial_train.mean_f1
        self.console.print(Text.assemble(
            ("epoch 0  ", "bold"),
            (f"optimizer={self.optimizer_name}   ", "dim"),
            (f"train_loss={initial_train.loss:.3f}  val_loss={initial_val.loss:.3f}  "
             f"F1={initial_train.mean_f1:.3f}", "yellow"),
        ))

    def epoch(self, rec: RoundRecord) -> None:
        prev_train_loss = self.train_losses[-1]
        self.train_losses.append(rec.train.loss)
        self.val_losses.append(rec.holdout.loss)
        d_f1 = rec.train.mean_f1 - self.prev_f1
        self.prev_f1 = rec.train.mean_f1

        body = Table.grid(padding=(0, 1))
        body.add_column()

        # (2) The Keras-style training line.
        if not rec.moved:
            tag = "[cyan]— converged[/cyan]"
        else:
            tag = "[green]accepted[/green]" if rec.accepted else "[red]rejected[/red]"
        body.add_row(Text.from_markup(
            f"[bold]train_loss[/bold] {rec.train.loss:.3f}   "
            f"[bold]val_loss[/bold] {rec.holdout.loss:.3f}   "
            f"F1 {rec.train.mean_f1:.3f} (val {rec.holdout.mean_f1:.3f})   "
            f"ΔF1 {d_f1:+.3f}   {tag}"
        ))

        # The symptom-level logs the optimizer was given this round (what's wrong — not why).
        if rec.proposal.logs:
            body.add_row(Text("logs the optimizer read (symptoms — what's wrong, not why):", style="yellow"))
            for d in rec.proposal.logs:
                body.add_row(Text(
                    f"  [{d['qid']}] recall={d['recall']} precision={d['precision']} "
                    f"cat={d['category']}  -  retrieved {d['num_retrieved']} docs (mode {d['retrieval_mode']})"))
                if d["wrong_claims"]:
                    body.add_row(Text(f"      grader flagged: {d['wrong_claims']}", style="dim"))

        # (3) The optimizer's ENTIRE output this epoch — a small structured action.
        if rec.proposal.reflection:
            body.add_row(Text(f"  reflect: {rec.proposal.reflection}", style="dim italic"))
        action = Table.grid(padding=(0, 2))
        action.add_column(style="bold")
        action.add_column()
        action.add_column()
        numeric_moves = [m for m in rec.moved if m not in FREETEXT_KNOBS]
        for name in numeric_moves:
            action.add_row(f"  {name}", f"[red]{_fmt(getattr(self.prev, name))}[/red]",
                           f"-> [green]{_fmt(getattr(rec.params, name))}[/green]")
        if not rec.moved:
            action.add_row("  (no move proposed)", "", "")
        body.add_row(Text("optimizer action (the LLM's whole output):", style="cyan"))
        body.add_row(action)
        body.add_row(Text(f"  why: {rec.proposal.rationale}", style="white"))
        body.add_row(Text(f"  confidence: {rec.proposal.confidence:.2f}", style="dim"))

        # (4) Prompt-as-parameter: a free-text move numeric optimization cannot make.
        for name in rec.moved:
            if name in FREETEXT_KNOBS:
                body.add_row(Panel(
                    Text.assemble(
                        ("before: ", "dim"), (f"{getattr(self.prev, name)!r}\n", "red"),
                        ("after:  ", "dim"), (f"{getattr(rec.params, name)!r}", "green"),
                    ),
                    title=f"free-text parameter rewritten: {name}",
                    subtitle="a free-text parameter — only an LLM can rewrite it",
                    border_style="magenta", title_align="left",
                ))

        body.add_row(Text.from_markup(
            f"[cyan]{sparkline(self.train_losses)}[/cyan] train   "
            f"[blue]{sparkline(self.val_losses)}[/blue] val"))

        self.console.print(Panel(
            body, title=f"epoch {rec.round}  ·  {rec.proposal.source}",
            title_align="left",
            border_style="green" if rec.accepted and rec.train.loss < prev_train_loss else "yellow",
        ))
        self.prev = rec.params


def final_report(console: Console, result: Result) -> None:
    records = result.records
    # We ship the best-on-validation checkpoint, not the final epoch. If a later epoch's val
    # loss came in worse than the shipped checkpoint, that's overfitting we detected and dropped.
    EPS = 1e-9
    overfit = bool(records) and records[-1].holdout.loss > result.final_holdout.loss + EPS
    if overfit:
        reason = (f"shipping best-val checkpoint (epoch {result.best_epoch}) — "
                  f"later epochs overfit: val loss regressed and was discarded")
    else:
        reason = f"shipping best-val checkpoint (epoch {result.best_epoch})"

    # Parameter diff, numeric knobs then the free-text one(s) called out separately.
    diff = Table(title="Parameters: start -> shipped best-val checkpoint (* = changed)",
                 title_style="bold", expand=False)
    diff.add_column("parameter", style="bold", no_wrap=True)
    diff.add_column("start", no_wrap=True, overflow="ellipsis", max_width=22)
    diff.add_column("shipped", no_wrap=True, overflow="ellipsis", max_width=22)
    diff.add_column("type", no_wrap=True)
    init = dataclasses.asdict(result.initial_params)
    fin = dataclasses.asdict(result.final_params)

    def _add(name: str, type_cell) -> None:
        changed = init[name] != fin[name]
        label = ("* " if changed else "  ") + name
        # Only colour when the value actually changed; unchanged rows stay neutral (dim).
        if changed:
            diff.add_row(label, Text(_fmt(init[name]), style="red"),
                         Text(_fmt(fin[name]), style="green"), type_cell)
        else:
            diff.add_row(label, Text(_fmt(init[name]), style="dim"),
                         Text("·", style="dim"), type_cell)

    for k in init:
        if k not in FREETEXT_KNOBS:
            _add(k, "numeric")
    for k in FREETEXT_KNOBS:
        _add(k, Text("free-text · LLM-only", style="magenta"))
    console.print(diff)

    # Per-question F1 before vs after (train + holdout).
    qt = Table(title="Per-question F1: before -> after", title_style="bold", expand=False)
    qt.add_column("question", style="bold")
    qt.add_column("split")
    qt.add_column("F1 before", justify="right")
    qt.add_column("F1 after", justify="right")
    before = {g.question_id: g for g in
              list(result.initial_train.grades) + list(result.initial_holdout.grades)}
    after = {g.question_id: g for g in
             list(result.final_train.grades) + list(result.final_holdout.grades)}
    holdout_ids = {g.question_id for g in result.final_holdout.grades}
    for qid in before:
        b, a = before[qid].f1, after[qid].f1
        color = "green" if a > b else ("white" if a == b else "red")
        qt.add_row(qid, "holdout" if qid in holdout_ids else "train",
                   f"{b:.2f}", f"[{color}]{a:.2f}[/{color}]")
    console.print(qt)

    summary = Text.assemble(
        (f"{reason}\n", "bold green"),
        (f"train  loss {result.initial_train.loss:.3f} -> {result.final_train.loss:.3f}   "
         f"F1 {result.initial_train.mean_f1:.3f} -> {result.final_train.mean_f1:.3f}"
         "   (shipped checkpoint — not necessarily lowest train, by design)\n", "white"),
        (f"val    loss {result.initial_holdout.loss:.3f} -> {result.final_holdout.loss:.3f}   "
         f"F1 {result.initial_holdout.mean_f1:.3f} -> {result.final_holdout.mean_f1:.3f}"
         "   (held-out — selection + early-stop ride this)\n", "white"),
        (f"forward-pass cache: {result.cache_hits}/"
         f"{result.cache_hits + result.cache_misses} stage computes reused", "dim"),
    )
    console.print(Panel(summary, border_style="green"))
