"""The ablation matrix.

An ablation is a configuration, not a code path. Stage 2's runner already
takes the stages to run and the index to retrieve from, and Σ(f) is filtered
on the way into the prompt rather than rebuilt — so one corpus, one Σ(f) build
and one set of prompts serve every condition here. Nothing in this module
reimplements a metric or a prompt.

Run order matters. Calls are cached on the whole rendered request, so a
configuration that shares a prefix with one already run pays only for the
calls that differ: C6 (no critic, no refiner) issues the same L7 and L1b
requests as C1, so once C1 has run its marginal cost is one call per function
instead of three. `run_matrix` therefore runs the full system first and the
subsets after it, which turns a nominal 84 GPU-hours into roughly 35–40.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .errors import CorpusError
from .llm import Client
from .runner import Context, generate, load_contexts

#: Row families a fact table can be stripped of, for the Σ(f) ablations.
ROW_FAMILIES = ("facts", "nodes", "edges", "paths", "calls", "deps",
                "reverts", "events")


class ExperimentError(CorpusError):
    """A configuration could not be run."""


def drop_families(*families: str) -> Callable[[Optional[dict]], Optional[dict]]:
    """A Σ(f) filter that removes whole row families from the fact table.

    Applied to the table on its way into the prompt, never to the table on
    disk. That is what lets one Σ(f) build serve every ablation, and it means
    an ablation cannot accidentally change the evidence a later run sees.
    """
    unknown = set(families) - set(ROW_FAMILIES)
    if unknown:
        raise ExperimentError(f"unknown row families: {sorted(unknown)}")

    def f(table: Optional[dict]) -> Optional[dict]:
        if table is None:
            return None
        out = dict(table)
        for fam in families:
            out[fam] = []
        if "facts" in families:
            return out
        # `interaction_order` is an F-row, so dropping R* alone would leave the
        # reentrancy verdict behind — the ablation is meant to remove the
        # guard evidence entirely.
        if "reverts" in families:
            out["facts"] = [r for r in out.get("facts", [])
                            if r.get("kind") != "interaction_order"]
        return out

    return f


def drop_all(table: Optional[dict]) -> Optional[dict]:
    """No structural evidence at all: the prompt says so explicitly."""
    return None


def keep_all(table: Optional[dict]) -> Optional[dict]:
    return table


@dataclass(frozen=True)
class Config:
    name: str
    label: str
    stages: Sequence[str]
    retrieval: bool = True
    sigma: Callable[[Optional[dict]], Optional[dict]] = keep_all

    @property
    def calls(self) -> int:
        return len(self.stages)


FULL = Config("C1", "full system", ("L7", "L1b", "L2", "L3", "L8"))

CONFIGS: List[Config] = [
    FULL,
    Config("C0", "zero-shot baseline", ("L1b",), retrieval=False,
           sigma=drop_all),
    Config("C2", "no Σ(f) — source only", FULL.stages, sigma=drop_all),
    Config("C3", "Σ(f) without D* rows", FULL.stages,
           sigma=drop_families("deps")),
    Config("C4", "Σ(f) without R* and interaction order", FULL.stages,
           sigma=drop_families("reverts")),
    Config("C5", "no retrieval", FULL.stages, retrieval=False),
    Config("C6", "no critic, no refiner", ("L7", "L1b", "L8")),
    Config("C7", "no intent reasoner", ("L1b", "L2", "L3", "L8")),
]

BY_NAME = {c.name: c for c in CONFIGS}

#: Full system first: every later configuration then reuses its cached calls.
RUN_ORDER = ["C1", "C6", "C7", "C5", "C3", "C4", "C2", "C0"]


def ordered(names: Optional[Sequence[str]] = None) -> List[Config]:
    chosen = list(names or RUN_ORDER)
    unknown = [n for n in chosen if n not in BY_NAME]
    if unknown:
        raise ExperimentError(f"unknown configurations: {unknown}")
    return [BY_NAME[n] for n in sorted(chosen, key=RUN_ORDER.index)]


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def run_config(corpus_root: Path, config: Config, client: Client, *,
               split: str = "val", seed: int = 0, index=None,
               out_root: Optional[Path] = None,
               contexts: Optional[List[Context]] = None,
               limit: Optional[int] = None,
               progress: bool = False) -> Path:
    """One configuration, one seed, one split. Resumable like any run."""
    contexts = contexts if contexts is not None else load_contexts(
        corpus_root, split)
    if limit:
        contexts = contexts[:limit]
    out_root = out_root or (corpus_root / "runs")
    path = out_root / config.name / f"seed{seed}" / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["pair_id"])

    use_index = index if config.retrieval else None
    with path.open("a", encoding="utf-8") as fh:
        for i, ctx in enumerate(contexts, 1):
            if ctx.pair["id"] in done:
                continue
            # Rebuilt rather than mutated: the same Context list is reused
            # for every configuration, so filtering in place would leak one
            # ablation's evidence into the next.
            scoped = Context(pair=ctx.pair, table=config.sigma(ctx.table),
                             unit=ctx.unit, version=ctx.version)
            try:
                rec = generate(scoped, client, use_index,
                               stages=list(config.stages))
            except CorpusError as e:
                rec = {"pair_id": ctx.pair["id"], "file": ctx.pair["file"],
                       "function": ctx.pair["name"], "split": split,
                       "error": str(e)[:400]}
            rec["config"] = config.name
            rec["seed"] = seed
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if progress:
                print(f"  [{config.name} seed{seed} {i}/{len(contexts)}] "
                      f"{ctx.pair['name']}")
    return path


def run_matrix(corpus_root: Path, client_factory: Callable[[int], Client], *,
               split: str = "val", seeds: Sequence[int] = (0, 1, 2),
               names: Optional[Sequence[str]] = None, index=None,
               out_root: Optional[Path] = None,
               limit: Optional[int] = None,
               progress: bool = False) -> Dict[str, List[Path]]:
    """Every configuration × every seed, in cache-warm order.

    `client_factory(seed)` supplies a client per seed, so the seed reaches the
    model's sampler *and* the cache key — two seeds must never share a cached
    call, or the seed axis measures nothing.
    """
    contexts = load_contexts(corpus_root, split)
    if limit:
        contexts = contexts[:limit]
    out: Dict[str, List[Path]] = {}
    for config in ordered(names):
        for seed in seeds:
            client = client_factory(seed)
            out.setdefault(config.name, []).append(
                run_config(corpus_root, config, client, split=split, seed=seed,
                           index=index, out_root=out_root, contexts=contexts,
                           progress=progress))
    return out


def load_results(out_root: Path, split: str = "val") -> List[dict]:
    """Every record from every configuration and seed under `out_root`."""
    rows: List[dict] = []
    for path in sorted(out_root.rglob(f"{split}.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows
