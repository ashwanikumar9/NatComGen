"""`python -m comgen` — run, report, emit.

A command line rather than a heredoc inside `run_pipeline.sh`. The pipeline's
existing stages embed their Python in the shell script, which was fine when
each was fifteen lines, but it means none of them can be run alone, tested, or
given a traceback worth reading. ComGen's stage is one line of shell calling
this instead.

    python -m comgen run    [--corpus DIR] [--split val] [--seeds "0 1 2"]
                            [--limit N] [--configs "G1 G3"] [--models JSON]
    python -m comgen report [--corpus DIR] [--split val] [--baseline G1]
    python -m comgen emit   [--corpus DIR] [--split val] [--config G1]

Defaults are read from the environment where the pipeline already exports them
(CORPUS, SPLIT, SEEDS, LIMIT, MODELS, OLLAMA), so the shell stage passes
nothing and a person running it by hand passes only what they are changing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from natspec_corpus.cache import CallCache
from natspec_corpus.llm import Client, OllamaBackend
from natspec_corpus.retrieve import build_index
from natspec_corpus.runner import load_contexts
from natspec_corpus.versioning import RunVersion

from . import RESULTS, RUNS
from . import experiment as X


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def _corpus(args) -> Path:
    root = args.corpus or _env("CORPUS")
    if not root:
        raise SystemExit("no corpus: pass --corpus DIR or set CORPUS")
    p = Path(root)
    if not (p / "pairs.jsonl").exists():
        raise SystemExit(f"{p} has no pairs.jsonl — is that the corpus root?")
    return p


def _models(args) -> dict:
    raw = args.models or _env("MODELS", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--models is not JSON: {e}")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def cmd_run(args) -> int:
    root = _corpus(args)
    models = _models(args)
    seeds = [int(s) for s in (args.seeds or _env("SEEDS", "0 1 2")).split()]
    limit = int(args.limit) if args.limit else (
        int(_env("LIMIT")) if _env("LIMIT") else None)
    names = args.configs.split() if args.configs else None
    host = args.ollama or _env("OLLAMA", "http://localhost:11434")

    # The call cache is shared with NatComGen on purpose: it is keyed on the
    # whole rendered request, so where ComGen issues a request C1 already
    # issued the answer is free and identical. A cache is not a result.
    cache = CallCache(root / ".call-cache")
    index = build_index(root)
    contexts = load_contexts(root, args.split)
    if limit:
        contexts = contexts[:limit]

    print(f"ComGen: {len(contexts)} functions, split {args.split}, "
          f"seeds {seeds}", flush=True)
    print(f"results -> {RUNS}", flush=True)

    for config in X.ordered(names):
        for seed in seeds:
            client = Client(OllamaBackend(host), models=models, cache=cache,
                            seed=seed)
            print(f"== {config.name} ({config.label}) seed {seed} "
                  f"rounds={X.ROUNDS.get(config.name)}", flush=True)
            X.run_one(root, config.name, client, split=args.split, seed=seed,
                      index=index, contexts=contexts, progress=True)
    print(f"\ncache: {cache.hits} hits, {cache.misses} misses")
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def cmd_report(args) -> int:
    from . import report as R
    root = _corpus(args)
    rows = X.load(args.split)
    if not rows:
        raise SystemExit(f"no records under {RUNS} for split {args.split}")
    written = R.write(root, rows, split=args.split, models=_models(args),
                      baseline=args.baseline)
    print(json.dumps({k: v.name for k, v in sorted(written.items())}, indent=1))
    print(f"\ntables in {RESULTS / 'tables'}; this run's numbers are the ones "
          f"named in RUNS.md — nothing was overwritten")
    return 0


# --------------------------------------------------------------------------
# emit  (Aggregator + Linter Agent)
# --------------------------------------------------------------------------

def cmd_emit(args) -> int:
    from .aggregator import assemble
    from .linter_agent import lint, summarise
    root = _corpus(args)
    rows = [r for r in X.load(args.split) if r.get("config") == args.config]
    if not rows:
        raise SystemExit(f"no {args.config} records under {RUNS}")
    pairs = {json.loads(l)["id"]: json.loads(l)
             for l in (root / "pairs.jsonl").read_text(
                 encoding="utf-8").splitlines() if l.strip()}

    version = RunVersion.open(RESULTS, ["lint_report.json"], ["documented"])
    out = version.dir("documented")
    results = []
    for a in assemble(root, rows, pairs):
        a.write(out)
        r = lint(a)
        results.append(r)
        flag = "ok" if r.ok else "; ".join(r.issues)
        print(f"  {a.rel}  {a.placed} placed  {flag}", flush=True)

    s = summarise(results)
    version.write_json("lint_report.json",
                       {"summary": s, "files": [r.to_dict() for r in results]})
    version.record(stage="comgen-emit", split=args.split, config=args.config,
                   files=s["files"], accepted=s["accepted"])
    print(f"\n{s['accepted']}/{s['files']} files accepted -> {out.name}/")
    if s["issues"]:
        print(json.dumps(s["issues"], indent=1))
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    # The shared options live on each subcommand rather than above them, so
    # `python -m comgen run --corpus DIR` works the way anyone would type it.
    # argparse only accepts a top-level optional BEFORE the subcommand, and a
    # parser that rejects the obvious spelling gets typed the obvious way once,
    # fails, and is never trusted again.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", help="corpus root (default: $CORPUS)")
    common.add_argument("--split", default=_env("SPLIT", "val"))
    common.add_argument("--models", help="JSON mapping model slots to tags")

    ap = argparse.ArgumentParser(prog="python -m comgen",
                                 description="ComGen — the revised "
                                             "architecture, on NatComGen's "
                                             "corpus")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", parents=[common],
                       help="run the configurations")
    r.add_argument("--seeds")
    r.add_argument("--limit")
    r.add_argument("--configs", help=f"subset of {' '.join(X.RUN_ORDER)}")
    r.add_argument("--ollama")
    r.set_defaults(fn=cmd_run)

    q = sub.add_parser("report", parents=[common], help="write the tables")
    q.add_argument("--baseline", default="G1")
    q.set_defaults(fn=cmd_report)

    e = sub.add_parser("emit", parents=[common],
                       help="document files and lint them")
    e.add_argument("--config", default="G1")
    e.set_defaults(fn=cmd_emit)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
