"""Run NatComGen over the re-grounded corpus.

The same `run_config` the main ablation matrix uses, pointed at a different
corpus root. Nothing about the pipeline is special-cased for the benchmark —
that is the point of having gone to the trouble of building a real corpus
rather than a bespoke evaluation path.

Two defaults differ from the main matrix, and both are choices rather than
conveniences.

**C5 (no retrieval) is the default configuration.** SmartDoc's train and test
sets come from one random split of one crawl, so a retrieval pool built from
their training functions would be doing on our side exactly what inflates the
published numbers on theirs. Running without retrieval measures the model and
the fact tables. `--configs C1,C5` runs both and shows the difference, which
is the honest way to report it.

**One seed by default**, because a seed axis on a subset this small buys
variance estimates nobody should trust. Ask for more when the coverage is
large enough to support them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from natspec_corpus.cache import CallCache                # noqa: E402
from natspec_corpus.experiment import BY_NAME, run_config  # noqa: E402
from natspec_corpus.llm import Client, OllamaBackend      # noqa: E402
from natspec_corpus.runner import load_contexts           # noqa: E402

SLOTS = ("GENERATOR_MODEL", "INTENT_REASONER_MODEL", "SEMANTIC_CRITIC_MODEL",
         "REFINER_MODEL", "VERIFIER_MODEL")


def _reachable(host: str) -> bool:
    try:
        urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False


def start_stub(port: int) -> subprocess.Popen:
    root = Path(__file__).resolve().parents[2]
    p = subprocess.Popen([sys.executable, str(root / "tools/stub_ollama.py"),
                          "--port", str(port)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        if _reachable(f"http://127.0.0.1:{port}"):
            return p
        time.sleep(0.25)
    p.terminate()
    raise SystemExit("the offline stub did not come up")


def main(argv=None) -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=here / "data/smartdoc_corpus")
    ap.add_argument("--split", default="val")
    ap.add_argument("--configs", default="C5")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ollama", default=os.environ.get(
        "OLLAMA", "http://localhost:11434"))
    ap.add_argument("--models", default="{}")
    ap.add_argument("--stub", action="store_true",
                    help="run against the offline stub; measures nothing")
    a = ap.parse_args(argv)

    names = [c.strip() for c in a.configs.replace(",", " ").split() if c.strip()]
    for n in names:
        if n not in BY_NAME:
            raise SystemExit(f"unknown configuration {n}; "
                             f"have {sorted(BY_NAME)}")
    seeds = [int(s) for s in a.seeds.replace(",", " ").split()]
    models = json.loads(a.models) if a.models else {}

    stub = None
    host = a.ollama
    if a.stub:
        stub = start_stub(11577)
        host = "http://127.0.0.1:11577"
        models = {s: "stub" for s in SLOTS}
    elif not _reachable(host):
        raise SystemExit(
            f"no Ollama at {host}. Start it, pass --ollama HOST, or pass "
            "--stub to exercise the wiring with a model that measures "
            "nothing.")
    else:
        missing = [s for s in SLOTS if s not in models]
        if missing:
            # An unmapped slot is sent to Ollama as its own literal name and
            # fails on the first call of that stage — hours in, if it is the
            # refiner. Cheaper to say so now.
            print(f"  ! unmapped model slots: {', '.join(missing)}",
                  file=sys.stderr)

    try:
        needs_index = any(BY_NAME[n].retrieval for n in names)
        idx = None
        if needs_index:
            from natspec_corpus.retrieve import build_index
            idx = build_index(a.corpus)
        ctxs = load_contexts(a.corpus, a.split)
        if not ctxs:
            raise SystemExit(
                f"no pairs in split {a.split!r} of {a.corpus} — run the build "
                "stage first")
        cache = CallCache(a.corpus / ".call-cache")
        for name in names:
            for seed in seeds:
                client = Client(OllamaBackend(host), models=models,
                                cache=cache, seed=seed)
                print(f"== {name} seed {seed} "
                      f"({a.limit or len(ctxs)} functions)", flush=True)
                run_config(a.corpus, BY_NAME[name], client, split=a.split,
                           seed=seed, index=idx, contexts=ctxs,
                           limit=a.limit, out_root=a.corpus / "runs",
                           progress=True)
    finally:
        if stub is not None:
            stub.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
