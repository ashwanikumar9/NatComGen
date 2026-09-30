"""One run against another, on exactly the functions both completed.

Comparing a new run to `main_5.md` does not work: the matched set changes when
a configuration's coverage changes, and on this project it moved from 99
functions to 105 between two reports. C1 -- which was never re-run -- shifted
0.6 Ba, 1.3 ROUGE-L and 1.1 METEOR purely from that. Any claim about a prompt
change smaller than those numbers would have been set composition wearing a
result's clothes.

So this pairs the two runs by pair id, drops anything either side is missing,
and reports the difference on the intersection.

    python3 -m functionWise.compare_runs \\
        --before "comgen/results/runs_archive/*/G1-seed*-test.jsonl" \\
        --after  "comgen/results/runs/G1/seed*/test.jsonl" \\
        --label G1 --split test

Globs rather than run roots, because an archived run rarely keeps the
`<config>/<seed>/` layout the loaders expect.

Three comparisons, each paired:

  surface     corpus BLEU/ROUGE/METEOR over the same segments
  support     claim support per function, with a bootstrap interval and a
              Wilcoxon p -- the one that says whether the change cost accuracy
  gates       McNemar on whether the caller restriction is stated, which is
              the right test for a before/after on the same binary outcome
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from natspec_corpus import access as A                      # noqa: E402
from natspec_corpus.evaluate import score_record            # noqa: E402
from natspec_corpus.report import markdown_table            # noqa: E402
from natspec_corpus.stats import compare                    # noqa: E402
from natspec_corpus.versioning import next_path             # noqa: E402

from functionWise import metrics, scope, surface            # noqa: E402


def read(pattern: str) -> Dict[int, Dict[str, dict]]:
    """{seed: {pair_id: record}} from a glob of jsonl files."""
    out: Dict[int, Dict[str, dict]] = {}
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files matched {pattern!r}")
    for path in files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            out.setdefault(rec.get("seed", 0), {})[rec["pair_id"]] = rec
    return out


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar on the discordant pairs.

    `b` improved-to-worse, `c` worse-to-improved. The concordant pairs carry
    no information about a change and are excluded by construction — which is
    why this and not a two-proportion test: the two samples are the same
    functions.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def surface_row(recs: Dict[int, Dict[str, dict]], pairs: Dict[str, dict],
                ids: Sequence[str], target: str, mode: str) -> dict:
    refs, hyps = [], []
    for seed in sorted(recs):
        for pid in ids:
            rec = recs[seed].get(pid)
            if rec is None:
                continue
            ref, hyp = surface.pair_strings(pairs[pid], rec.get("final") or "",
                                            target=target, mode=mode)
            if ref.strip():
                refs.append(ref)
                hyps.append(hyp)
    return metrics.score(refs, hyps)


def support(recs: Dict[str, dict], pairs: Dict[str, dict],
            ids: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for pid in ids:
        rec = recs.get(pid)
        if rec is None:
            continue
        s = score_record(rec, pairs[pid])
        out[pid] = (s["support_rate"] if s["support_rate"] is not None
                    else (s["by_kind"].get("notice") or {}).get("bleu", 0.0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="glob of jsonl files")
    ap.add_argument("--after", required=True, help="glob of jsonl files")
    ap.add_argument("--label", default="run")
    ap.add_argument("--corpus", type=Path, default=HERE / "data/NatSpecGold")
    ap.add_argument("--split", default="test")
    ap.add_argument("--kinds", default="function")
    ap.add_argument("--normalize", default="paper", choices=surface.MODES)
    ap.add_argument("--seed", type=int, default=0,
                    help="the seed the paired tests use")
    ap.add_argument("--out", type=Path, default=HERE / "results/functionwise")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    pairs = scope.scoped(scope.load_pairs(args.corpus), split=args.split,
                         kinds=scope.parse_kinds(args.kinds))
    tables = scope.load_tables(args.corpus)
    before, after = read(args.before), read(args.after)

    common_seeds = sorted(set(before) & set(after))
    if not common_seeds:
        raise SystemExit(f"no seed in common: before {sorted(before)}, "
                         f"after {sorted(after)}")
    ids = set(pairs)
    for src in (before, after):
        for seed in common_seeds:
            ids &= set(src[seed])
    ids = sorted(ids)
    if not ids:
        raise SystemExit("no function completed in both runs")

    doc = [f"# {args.label}: before against after — {args.split}\n",
           f"{len(ids)} functions completed by both runs, seeds "
           f"{', '.join(map(str, common_seeds))}, `--normalize "
           f"{args.normalize}`. Anything either run missed is excluded from "
           f"both, so no row rests on a different function set.\n"]

    keep = ["n", "Ba", "B1", "B2", "B4", "ROUGE-1", "ROUGE-L", "METEOR"]
    for target, title in (("whole", "Whole comment"), ("notice", "@notice"),
                          ("dev", "@dev"), ("param", "@param"),
                          ("return", "@return")):
        b = surface_row({s: before[s] for s in common_seeds}, pairs, ids,
                        target, args.normalize)
        a = surface_row({s: after[s] for s in common_seeds}, pairs, ids,
                        target, args.normalize)
        if not b.get("n") and not a.get("n"):
            continue
        rows = [["before"] + metrics.cells(b, keep),
                ["after"] + metrics.cells(a, keep),
                ["Δ"] + ["—" if k in ("n",) else
                         f"{a.get(k, 0) - b.get(k, 0):+.2f}" for k in keep]]
        doc.append(f"## {title}\n\n"
                   + markdown_table(["run"] + keep, rows) + "\n")

    # ---- claim support, paired ------------------------------------------
    sb = support(before[args.seed], pairs, ids)
    sa = support(after[args.seed], pairs, ids)
    shared = sorted(set(sb) & set(sa))
    if shared:
        c = compare({k: sb[k] for k in shared}, {k: sa[k] for k in shared})
        d = c.to_dict()
        doc.append("## Claim support (paired, seed "
                   f"{args.seed})\n\n"
                   + markdown_table(
                       ["n", "before", "after", "Δ", "95% CI", "p", "Cliff's δ"],
                       [[d["n"], f"{d['mean_a']:.3f}", f"{d['mean_b']:.3f}",
                         f"{d['diff']:+.3f}",
                         f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]",
                         f"{d['p_value']:.4f}", f"{d['cliffs_delta']:+.3f}"]])
                   + "\n\nA negative Δ here means the change bought surface "
                     "similarity with accuracy, and is a reason to revert it "
                     "whatever the BLEU did.\n")

    # ---- gates, McNemar --------------------------------------------------
    rows, b_yes = [], 0
    bb = cc = both = neither = 0
    for pid in ids:
        gates = A.caller_gates(pairs[pid], tables.get(pid))
        if not gates:
            continue
        rb = before[args.seed].get(pid)
        ra = after[args.seed].get(pid)
        if rb is None or ra is None:
            continue
        sb_ = all(A.states(rb.get("final") or "", g) for g in gates)
        sa_ = all(A.states(ra.get("final") or "", g) for g in gates)
        both += sb_ and sa_
        neither += (not sb_) and (not sa_)
        bb += sb_ and not sa_
        cc += (not sb_) and sa_
    gated = both + neither + bb + cc
    if gated:
        p = mcnemar(bb, cc)
        doc.append("## Caller gates (McNemar, seed "
                   f"{args.seed})\n\n"
                   + markdown_table(
                       ["gated functions", "stated before", "stated after",
                        "lost", "gained", "p (exact)"],
                       [[gated, both + bb, both + cc, bb, cc, f"{p:.5f}"]])
                   + "\n\nMcNemar and not a two-proportion test: both columns "
                     "are the same functions, so only the functions that "
                     "changed carry information. `lost` and `gained` are the "
                     "discordant pairs the test is computed on.\n")

    text = "\n".join(doc)
    print(text)
    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        path = next_path(args.out / f"compare_{args.label}_{args.split}.md")
        path.write_text(text, encoding="utf-8")
        print(f"\nwritten to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
