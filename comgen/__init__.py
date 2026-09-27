"""ComGen — the revised agentic architecture, built on NatComGen's corpus.

What changes from `natspec_corpus` is the orchestration, not the evidence. The
corpus, the Σ(f) fact tables, the retrieval index and the call cache are read
exactly as they are; this package replaces the five-call straight line in
`natspec_corpus.runner` with

    Generator → Critic Group (deterministic ‖ semantic) → Generator (revise)
              → Judge → Aggregator → Linter

and loops the middle. The two critics sit in one group, both read the
generator's draft directly, and both sets of feedback go back to the generator
— there is no separate refiner agent, and no ranking judge between them.

RESULTS LIVE HERE, not in the corpus tree. `comgen/results/runs/<config>/
seed<n>/<split>.jsonl` holds the raw records, `comgen/results/tables/` the
report, `comgen/results/documented/` the emitted files. A ComGen run therefore
never writes into `NatSpecGold/runs`, so the C0–C7 ablation results it is
being compared against cannot be disturbed by it.

The one thing deliberately shared is the call cache. It is keyed on the whole
rendered request, so where ComGen issues a request C1 already issued — the same
intent reading, the same semantic critique of the same draft — the answer is
free and, being the same request, identical. Sharing it costs nothing and saves
hours; it is a cache, not a result.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

#: Everything this package writes goes under here.
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
TABLES = RESULTS / "tables"

__all__ = ["ROOT", "RESULTS", "RUNS", "TABLES"]
