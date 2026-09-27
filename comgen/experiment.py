"""ComGen's configurations, run on NatComGen's machinery.

Nothing here reimplements a runner, a metric or a resume. `run_config` in
`natspec_corpus.experiment` already does all of it — Σ(f) filtered on the way
into the prompt, retrieval switched per configuration, output appended as JSONL
and restarts skipping the ids already written — and since it now takes the
per-function pipeline as an argument, ComGen supplies its own and inherits the
rest. This module is therefore a configuration list and a call.

WHAT THE CONFIGURATIONS TEST. C0–C7 asked what Σ(f) and each agent are worth
inside the five-call line. These ask what the *loop* is worth, and each one
removes exactly one thing:

    G1  full ComGen                 the architecture as designed
    G3  one round only              the loop's budget cut to v2's single pass.
                                    G1 vs G3 is the round-count result — the
                                    one number that says whether looping earns
                                    its variable cost at all.
    G5  semantic critic only        the Critic Group minus its free half
    G4  deterministic critic only   the Critic Group minus its model half.
                                    G4 vs G5 separates "criticism helps" from
                                    "a model's criticism helps", which C1
                                    could not distinguish because its
                                    deterministic check only ever vetoed.
    G6  no contract intent          what one call per file buys
    G7  no retrieval                as C5
    G2  no Σ(f)                     as C2
    G0  zero-shot                   as C0, and directly comparable to it

RUN ORDER IS COST. Calls are cached on the whole rendered request, so a
configuration that issues a request G1 already issued pays nothing for it. G1
runs first; then the configurations that share its round-0 generator call
(G3, G5, G4) are nearly free up to the point where they diverge; then the ones
that change what the generator reads, which share nothing.

RESULTS STAY IN `comgen/results/`. Not in the corpus tree. A ComGen run cannot
overwrite, extend or otherwise touch the C0–C7 records it is being compared
against — which matters more than tidiness, because those records are the
thesis's existing result and they took 35 GPU-hours.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from natspec_corpus.errors import CorpusError
from natspec_corpus.experiment import (
    Config,
    drop_all,
    drop_families,
    keep_all,
    load_results,
    run_config,
)
from natspec_corpus.llm import Client
from natspec_corpus.runner import load_contexts

from . import RUNS
from . import intent as I
from .orchestrator import DEFAULT_ROUNDS, generate

FULL_STAGES = ("IC", "I7", "G", "CD", "CS", "R", "J")

FULL = Config("G1", "full ComGen", FULL_STAGES)

CONFIGS: List[Config] = [
    FULL,
    Config("G3", "one round only (v2's single pass)", FULL_STAGES),
    Config("G5", "semantic critic only", ("IC", "I7", "G", "CS", "R", "J")),
    Config("G4", "deterministic critic only", ("IC", "I7", "G", "CD", "R", "J")),
    Config("G6", "no contract intent", ("I7", "G", "CD", "CS", "R", "J")),
    Config("G7", "no retrieval", FULL_STAGES, retrieval=False),
    Config("G2", "no Σ(f) — source only", FULL_STAGES, sigma=drop_all),
    Config("G0", "zero-shot baseline", ("G",), retrieval=False,
           sigma=drop_all),
]

BY_NAME = {c.name: c for c in CONFIGS}

#: The round budget per configuration. It lives here rather than on `Config`
#: so `natspec_corpus.experiment` needs no new field and the C0–C7
#: configurations stay byte-identical to the ones that produced the published
#: numbers.
ROUNDS: Dict[str, int] = {
    "G1": DEFAULT_ROUNDS,
    "G3": 1,
    "G5": DEFAULT_ROUNDS,
    "G4": DEFAULT_ROUNDS,
    "G6": DEFAULT_ROUNDS,
    "G7": DEFAULT_ROUNDS,
    "G2": DEFAULT_ROUNDS,
    "G0": 1,
}

#: Cache-warm order. See the module docstring.
RUN_ORDER = ["G1", "G3", "G5", "G4", "G6", "G7", "G2", "G0"]


def ordered(names: Optional[Sequence[str]] = None) -> List[Config]:
    chosen = list(names or RUN_ORDER)
    unknown = [n for n in chosen if n not in BY_NAME]
    if unknown:
        raise CorpusError(f"unknown ComGen configurations: {unknown}")
    return [BY_NAME[n] for n in sorted(chosen, key=RUN_ORDER.index)]


def pipeline_for(config: Config, client: Client) -> Callable[..., dict]:
    """ComGen's per-function pipeline, bound to this configuration.

    The `ContractIntent` memo is created here, once per configuration and seed,
    which is what makes the contract reading cost one call per FILE. Bound per
    client on purpose: the client carries the seed, and a memo shared across
    seeds would hand seed 1 the reading produced under seed 0 and quietly flatten
    the seed axis this run exists to measure.
    """
    return partial(generate,
                   rounds=ROUNDS.get(config.name, DEFAULT_ROUNDS),
                   contract=I.ContractIntent(
                       client, enabled="IC" in config.stages))


def run_one(corpus_root: Path, name: str, client: Client, *,
            split: str = "val", seed: int = 0, index=None,
            out_root: Optional[Path] = None, contexts=None,
            limit: Optional[int] = None, progress: bool = False) -> Path:
    """One ComGen configuration, one seed, one split."""
    config = BY_NAME.get(name)
    if config is None:
        raise CorpusError(f"unknown ComGen configuration: {name}")
    return run_config(corpus_root, config, client, split=split, seed=seed,
                      index=index, out_root=Path(out_root or RUNS),
                      contexts=contexts, limit=limit, progress=progress,
                      generate_fn=pipeline_for(config, client))


def run_matrix(corpus_root: Path, client_factory: Callable[[int], Client], *,
               split: str = "val", seeds: Sequence[int] = (0, 1, 2),
               names: Optional[Sequence[str]] = None, index=None,
               out_root: Optional[Path] = None, limit: Optional[int] = None,
               progress: bool = False) -> Dict[str, List[Path]]:
    """Every ComGen configuration × every seed, in cache-warm order."""
    contexts = load_contexts(corpus_root, split)
    if limit:
        contexts = contexts[:limit]
    out: Dict[str, List[Path]] = {}
    for config in ordered(names):
        for seed in seeds:
            client = client_factory(seed)
            out.setdefault(config.name, []).append(
                run_one(corpus_root, config.name, client, split=split,
                        seed=seed, index=index, out_root=out_root,
                        contexts=contexts, limit=limit, progress=progress))
    return out


def load(split: str = "val", out_root: Optional[Path] = None) -> List[dict]:
    """Every ComGen record, from `comgen/results/runs` unless told otherwise."""
    return load_results(Path(out_root or RUNS), split)


__all__ = ["CONFIGS", "BY_NAME", "ROUNDS", "RUN_ORDER", "FULL", "FULL_STAGES",
           "ordered", "pipeline_for", "run_one", "run_matrix", "load",
           "drop_all", "drop_families", "keep_all"]
