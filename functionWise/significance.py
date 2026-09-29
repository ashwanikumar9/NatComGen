"""Do the ablation findings survive the function-only scope?

This is the question that decides whether narrowing the evaluation to
functions is free. On val, 82 of 148 pairs are functions -- the scope roughly
halves n, and Holm correction over six or eight comparisons is unforgiving
about lost power. The Sigma(f) result at p<0.0001 will survive anything; the
marginal ones may not.

So rather than re-running the tests once and hoping, this runs them under both
scopes and prints what changed:

    python3 -m functionWise.significance --split val --baseline G1
    python3 -m functionWise.significance --split val --baseline G8 --system comgen

A finding that flips is not a failure. It says the effect lived partly in the
event, error and modifier pairs -- which is worth knowing, and worth a
sentence in the paper either way.

The metric is the one `comgen/report.py` uses for its ablations: per-function
claim-support rate, falling back to notice BLEU where support is undefined.
Changing it here would make these p-values incomparable with the existing
tables, so it is not configurable.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from natspec_corpus.evaluate import score_record                  # noqa: E402
from natspec_corpus.report import ablation_table_rows, seed_table  # noqa: E402
from natspec_corpus.report import markdown_table                  # noqa: E402
from natspec_corpus.stats import ablation_table                   # noqa: E402
from natspec_corpus.versioning import next_path                   # noqa: E402
from tools.bleu_table import load                                 # noqa: E402

from functionWise import scope                                    # noqa: E402


def metric(rec: dict, pair: dict) -> float:
    """One function's score, exactly as `comgen/report.py` computes it."""
    s = score_record(rec, pair)
    if s["support_rate"] is not None:
        return s["support_rate"]
    return (s["by_kind"].get("notice") or {}).get("bleu", 0.0)


def per_seed_scores(rows: Sequence[dict], pairs: Dict[str, dict],
                    order: Sequence[str]) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for name in order:
        seeds = sorted({r["seed"] for r in rows if r.get("config") == name
                        and r.get("seed") is not None})
        for seed in seeds:
            scored = {
                r["pair_id"]: metric(r, pairs[r["pair_id"]])
                for r in rows
                if r.get("config") == name and r.get("seed") == seed
                and not r.get("error") and r.get("pair_id") in pairs}
            if scored:
                out[name][seed] = scored
    return out


def run_scope(rows, every, split, kinds, order, by_name, baseline,
              resamples: int) -> Optional[dict]:
    pairs = scope.scoped(every, split=split, kinds=kinds)
    rows = [r for r in rows if r.get("pair_id") in pairs]
    if not rows:
        return None
    per_seed = per_seed_scores(rows, pairs, order)
    if baseline not in per_seed or len(per_seed) < 2:
        return None
    # Pairing requires the same functions on both sides. Restrict every
    # configuration to the functions all of them completed, or `compare`
    # silently pairs over whichever subset two configurations happen to share.
    common = set.intersection(*[set(next(iter(v.values())))
                                for v in per_seed.values()])
    per_seed = {name: {s: {k: val for k, val in d.items() if k in common}
                       for s, d in seeds.items()}
                for name, seeds in per_seed.items()}
    table = ablation_table(per_seed[baseline],
                           {n: v for n, v in per_seed.items() if n != baseline},
                           resamples=resamples)
    table["_n"] = len(common)
    table["_pairs"] = len(pairs)
    return table


def verdicts(table: dict) -> Dict[str, bool]:
    return {n: bool(c["reject"]) and not c["within_noise"]
            for n, c in table.get("comparisons", {}).items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=HERE / "data/NatSpecGold")
    ap.add_argument("--runs", type=Path, default=None,
                    help="default: comgen/results/runs, or <corpus>/runs "
                         "with --system natcomgen")
    ap.add_argument("--system", default="comgen",
                    choices=("comgen", "natcomgen"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--baseline", default="G1",
                    help="the full system every ablation is compared against")
    ap.add_argument("--kinds", default="function")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=HERE / "results/functionwise")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    if args.system == "comgen":
        from comgen.experiment import BY_NAME as by_name, RUN_ORDER as order
        runs = args.runs or HERE / "comgen/results/runs"
    else:
        from natspec_corpus.experiment import BY_NAME as by_name
        from natspec_corpus.experiment import RUN_ORDER as order
        runs = args.runs or args.corpus / "runs"
    if not Path(runs).exists():
        raise SystemExit(f"no run tree at {runs}")

    every = scope.load_pairs(args.corpus)
    rows = load(Path(runs), args.split)
    if not rows:
        raise SystemExit(f"no {args.split}.jsonl rows under {runs}")

    narrow = scope.parse_kinds(args.kinds)
    labels = {n: by_name[n].label for n in order if n in by_name}

    doc = [f"# Does the scope change the findings? - {args.system}, "
           f"{args.split}, baseline {args.baseline}\n"]
    tables: Dict[str, dict] = {}
    for tag, kinds in (("all declarations", scope.ALL),
                       ("+".join(narrow), narrow)):
        table = run_scope(rows, every, args.split, kinds, order, by_name,
                          args.baseline, args.resamples)
        if table is None:
            doc.append(f"## {tag}\n\nnot enough data.\n")
            continue
        tables[tag] = table
        h, r = ablation_table_rows(table, labels)
        doc.append(f"## {tag}\n\n{table['_n']} declarations scored "
                   f"({table['_pairs']} in this scope)\n\n"
                   + markdown_table(h, r) + "\n")
        h, r = seed_table(table)
        doc.append("Seed spread\n\n" + markdown_table(h, r) + "\n")

    if len(tables) == 2:
        (wide_tag, wide), (narrow_tag, narrow_t) = list(tables.items())
        a, b = verdicts(wide), verdicts(narrow_t)
        rows_ = []
        for name in sorted(set(a) | set(b)):
            before, after = a.get(name), b.get(name)
            if before is None or after is None:
                change = "not run in both"
            elif before == after:
                change = "unchanged"
            elif before and not after:
                change = "**LOST**"
            else:
                change = "**GAINED**"
            rows_.append([labels.get(name, name),
                          "-" if before is None else ("yes" if before else "no"),
                          "-" if after is None else ("yes" if after else "no"),
                          change])
        doc.append("## What the scope change costs\n\n"
                   + markdown_table(
                       ["Ablation", f"significant, {wide_tag}",
                        f"significant, {narrow_tag}", "verdict"], rows_)
                   + "\n\n`significant` means Holm-adjusted p < 0.05 **and** "
                     "a difference larger than the seed-to-seed spread. A "
                     "**LOST** row is not a reason to abandon the scope: it "
                     "says the effect lived partly in the declarations the "
                     "scope removes, which is a finding in its own right and "
                     "belongs in the write-up. A finding that survives the "
                     "halved sample is stronger than it was before.\n")

    text = "\n".join(doc)
    print(text)
    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        path = next_path(args.out /
                         f"significance_{args.system}_{args.split}_"
                         f"{args.baseline}.md")
        path.write_text(text, encoding="utf-8")
        print(f"\nwritten to {path}", file=sys.stderr)
    if args.json:
        path = next_path(args.json)
        path.write_text(json.dumps(tables, indent=1, default=str),
                        encoding="utf-8")
        print(f"written to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
