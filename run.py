"""Argus showcase entry point.

    python run.py --brain anthropic   # Anthropic API brain (needs ANTHROPIC_API_KEY)
    python run.py --brain local       # local model server, e.g. Ollama (no API cost)
    python run.py --rounds 20         # cap the number of optimization rounds

Argus optimizes ONLY via an LLM brain — there is no deterministic fallback. Watch a
deliberately-broken RAG pipeline converge from low F1 to high F1 as the LLM proposes textual
gradients over the pipeline trace, one knob group at a time.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from argus.llm import NoBrainConfigured, make_backend
from argus.loop import run_optimization
from argus.optimizer import LLMOptimizer
from argus.params import BROKEN_PARAMS
from argus.ui import Renderer, final_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Argus — LLM-as-optimizer over a RAG pipeline.")
    parser.add_argument("--brain", choices=["anthropic", "local", "agent"], default="anthropic",
                        help="which LLM reasons over the trace each epoch")
    parser.add_argument("--rounds", type=int, default=16, help="max optimization rounds")
    parser.add_argument("--patience", type=int, default=2, help="stop after N rounds without improvement")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable staged caching (ablation: same result, more computes)")
    args = parser.parse_args(argv)

    console = Console()

    try:
        backend = make_backend(args.brain)
    except NoBrainConfigured as exc:
        console.print(f"[red]No LLM brain available:[/red] {exc}")
        console.print("[dim]Argus has no rule-based fallback by design — configure a brain "
                      "(--brain anthropic | local).[/dim]")
        return 1

    optimizer = LLMOptimizer(backend)
    renderer = Renderer(console, f"{args.brain} brain")
    renderer.architecture()

    # The loop computes the broken starting grade once and hands it to the banner via on_start;
    # synthesis + judge run on `backend`, the optimizer reasons on it too.
    result = run_optimization(
        optimizer, BROKEN_PARAMS,
        backend=backend,
        max_rounds=args.rounds, patience=args.patience,
        use_cache=not args.no_cache,
        on_start=renderer.banner,
        on_round=renderer.epoch,
    )
    final_report(console, result)
    out = Path("result.json")
    _write_result(result, args.brain, out)
    console.print(f"[dim]checkpoint written -> {out.resolve()}[/dim]")
    return 0


def _write_result(result, brain: str, path: Path) -> None:
    """Persist the run as a JSON checkpoint — Argus's equivalent of a PyTorch `.pt`:
    the learned (shipped) parameters plus all metrics and the per-epoch trajectory."""
    before = {g.question_id: g for g in
              list(result.initial_train.grades) + list(result.initial_holdout.grades)}
    after = {g.question_id: g for g in
             list(result.final_train.grades) + list(result.final_holdout.grades)}
    holdout_ids = {g.question_id for g in result.final_holdout.grades}

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "brain": brain,
        "best_epoch": result.best_epoch,
        "shipped_params": dataclasses.asdict(result.final_params),
        "metrics": {
            "train": {
                "loss_before": result.initial_train.loss, "loss_after": result.final_train.loss,
                "f1_before": result.initial_train.mean_f1, "f1_after": result.final_train.mean_f1,
                "recall_after": result.final_train.mean_recall,
                "precision_after": result.final_train.mean_precision,
            },
            "val": {
                "loss_before": result.initial_holdout.loss, "loss_after": result.final_holdout.loss,
                "f1_before": result.initial_holdout.mean_f1, "f1_after": result.final_holdout.mean_f1,
                "recall_after": result.final_holdout.mean_recall,
                "precision_after": result.final_holdout.mean_precision,
            },
        },
        "per_question": [
            {"id": qid, "split": "holdout" if qid in holdout_ids else "train",
             "f1_before": before[qid].f1, "f1_after": after[qid].f1}
            for qid in before
        ],
        "epochs": [
            {"epoch": rec.round, "accepted": rec.accepted, "moved": rec.moved,
             "train_loss": rec.train.loss, "train_f1": rec.train.mean_f1,
             "val_loss": rec.holdout.loss, "val_f1": rec.holdout.mean_f1,
             "reflection": rec.proposal.reflection, "rationale": rec.proposal.rationale,
             "confidence": rec.proposal.confidence}
            for rec in result.records
        ],
        "cache": {"hits": result.cache_hits, "misses": result.cache_misses},
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
