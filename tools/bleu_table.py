"""BLEU and ROUGE-L for every configuration of both architectures, matched.

Why this exists rather than reading two `main.md` files side by side. The main
table prints four of the eight BLEU/ROUGE cells and leaves out `dev` entirely,
and — more importantly — each architecture's table is computed over whichever
functions that architecture happened to complete. NatComGen's run 3 has rows
with n between 33 and 40 because some functions errored; ComGen's has 40. A
comparison across two such tables silently compares different function sets.

So this scores every configuration of both architectures on **exactly the
functions common to all of them**, which is the only way a row-to-row
difference means anything.

    python3 tools/bleu_table.py                       # val, seed 0
    python3 tools/bleu_table.py --seed 1 --split val
    python3 tools/bleu_table.py --all-seeds           # pool every seed

ABOUT THE AVERAGE. `BLEU avg` is the unweighted mean of the per-field
sentence-BLEU means. It is NOT corpus BLEU, and on this project's own data the
two differ by more than an order of magnitude — sentence BLEU with add-one
smoothing over a one-line @notice is a far more forgiving number than corpus
BLEU over pooled n-gram counts. Papers quote the corpus figure. For that, run
`tools/overall_bleu.py`, which reproduces SmartDoc's 47.39 and agrees with nltk
to 1e-9. Do not put a number from this table next to a published one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import sys

# Runnable as `python3 tools/<this>.py` from the repo root or anywhere
# else: the package sits one directory up, not on the default path.
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from natspec_corpus.evaluate import aggregate, score_record
from natspec_corpus.report import markdown_table

FIELDS = ("notice", "dev", "param", "return")


def load(root: Path, split: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(Path(root).rglob(f"{split}.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def matched_ids(rows: Sequence[dict], seeds: Optional[Sequence[int]] = None
                ) -> set:
    """Pair ids present, without error, in every (config, seed) that ran.

    An intersection and not a union: a function one configuration failed on is
    a function no configuration may be scored on, or the rows stop being
    comparable.
    """
    groups: Dict[tuple, set] = {}
    for r in rows:
        if r.get("error") or r.get("config") is None:
            continue
        if seeds is not None and r.get("seed") not in seeds:
            continue
        groups.setdefault((r["config"], r.get("seed")), set()).add(r["pair_id"])
    if not groups:
        return set()
    shared = set.intersection(*groups.values())
    return shared


def block_for(rows: Sequence[dict], pairs: Dict[str, dict], config: str,
              ids: set, seeds: Optional[Sequence[int]]) -> Optional[dict]:
    scored = [score_record(r, pairs[r["pair_id"]]) for r in rows
              if r.get("config") == config and not r.get("error")
              and r["pair_id"] in ids
              and (seeds is None or r.get("seed") in seeds)]
    if not scored:
        return None
    return aggregate(scored)["all"]


def _n(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def rows_for(metric: str, systems: Sequence[tuple]) -> tuple:
    """One table for `bleu` or for `rouge_l`."""
    key = "BLEU" if metric == "bleu" else "ROUGE-L"
    headers = (["system", "config", "what it removes", "n"]
               + [f"{f} {key}" for f in FIELDS] + [f"{key} avg"])
    out: List[List[str]] = []
    for system, name, label, block in systems:
        if block is None:
            continue
        per = [(block.get(f) or {}).get(metric) for f in FIELDS]
        out.append([system, name, label, block.get("n", 0)]
                   + [_n(v) for v in per] + [_n(_mean(per))])
    return headers, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="data/NatSpecGold")
    ap.add_argument("--comgen-runs", default="comgen/results/runs")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-seeds", action="store_true",
                    help="pool every seed instead of taking one")
    args = ap.parse_args(argv)

    corpus = Path(args.corpus)
    pairs = {json.loads(l)["id"]: json.loads(l)
             for l in (corpus / "pairs.jsonl").read_text(
                 encoding="utf-8").splitlines() if l.strip()}
    seeds = None if args.all_seeds else [args.seed]

    from comgen.experiment import BY_NAME as G_BY, RUN_ORDER as G_ORDER
    from natspec_corpus.experiment import BY_NAME as C_BY, RUN_ORDER as C_ORDER

    trees = [("NatComGen", corpus / "runs", C_ORDER, C_BY),
             ("ComGen", Path(args.comgen_runs), G_ORDER, G_BY)]

    loaded = []
    for system, root, order, by_name in trees:
        if not Path(root).exists():
            print(f"note: {root} does not exist — skipping {system}")
            continue
        rows = load(root, args.split)
        rows = [r for r in rows if r.get("pair_id") in pairs]
        if not rows:
            print(f"note: no {args.split} records under {root}")
            continue
        loaded.append((system, rows, order, by_name))

    if not loaded:
        raise SystemExit("nothing to compare")

    # Matched across BOTH architectures, not just within each.
    ids = set.intersection(*[matched_ids(rows, seeds)
                             for _, rows, _, _ in loaded])
    if not ids:
        raise SystemExit("the two run trees share no function that every "
                         "configuration completed")

    systems = []
    for system, rows, order, by_name in loaded:
        for name in order:
            block = block_for(rows, pairs, name, ids, seeds)
            systems.append((system, name, by_name[name].label, block))

    scope = "all seeds pooled" if args.all_seeds else f"seed {args.seed}"
    print(f"# BLEU and ROUGE-L — {args.split} split, {scope}, "
          f"{len(ids)} functions common to every configuration\n")
    for metric in ("bleu", "rouge_l"):
        h, r = rows_for(metric, systems)
        print(markdown_table(h, r))
        print()
    print("`avg` is the unweighted mean of the per-field sentence-BLEU / "
          "ROUGE-L means. It is NOT corpus BLEU — see tools/overall_bleu.py "
          "for the figure comparable to a published one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
