"""Staged forward-pass cache.

Each pipeline stage caches its output under a key derived from a hash of *only* the knobs
at or upstream of that stage (`Params.stage_signature`). Changing a cheap downstream knob
(e.g. MMR lambda) therefore reuses the expensive chunk/embed/index results untouched —
the same economics as re-running enrichment over a frozen index instead of re-ingesting.

Hit/miss counters back the demo's ablation ("caching is correctness-neutral, cost-positive").
"""

from __future__ import annotations

from typing import Any, Callable


class StageCache:
    def __init__(self, enabled: bool = True) -> None:
        self._store: dict[tuple, Any] = {}
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, stage: str, signature: str, query: str, compute: Callable[[], Any]) -> Any:
        if not self.enabled:
            # Ablation mode: always recompute, never store. Results are identical; only the
            # number of stage computes changes — the demo's "caching is correctness-neutral,
            # cost-positive" check.
            self.misses += 1
            return compute()
        key = (stage, signature, query)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        value = compute()
        self._store[key] = value
        return value

    @property
    def total(self) -> int:
        return self.hits + self.misses

    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0
