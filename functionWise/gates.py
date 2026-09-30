"""Does the comment say who may call the function? A reference-free metric.

Every other number in this package compares a generated comment against a
reference. This one does not, and that is the point: NatSpecGold's own
comments state the caller restriction on only 55% of gated functions in train
and 46% in test, so a reference-based metric caps the system at the
reference's own incompleteness. Whether a comment names a gate the declaration
carries is decidable from the code.

    python3 -m functionWise.gates --split test --configs G8,G1,C1
    python3 -m functionWise.gates --split train --gold-only

Three numbers per configuration:

  gate recall     of functions with a caller gate, the share whose comment
                  states it. The headline.
  invention rate  of functions with NO caller gate, the share whose comment
                  claims one anyway. The cost of pushing recall up.
  gold recall     the same recall computed on the reference, printed beside
                  it, because beating it is the interesting result and
                  matching it is not.

Note the split coverage before reading anything: val has ZERO caller-gated
functions, so this metric cannot be developed or validated there. Use train to
develop and test to report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from natspec_corpus import access as A                       # noqa: E402
from natspec_corpus.report import markdown_table             # noqa: E402
from natspec_corpus.versioning import next_path              # noqa: E402
from tools.bleu_table import load                            # noqa: E402

from functionWise import scope                               # noqa: E402


def gold_text(pair: dict) -> str:
    return ". ".join(x for x in (pair.get("notice") or "",
                                 pair.get("dev") or "") if x)


def measure(pairs: Dict[str, dict], text_of, tables=None) -> dict:
    gated = ungated = stated = invented = 0
    missed: List[str] = []
    for pid, pair in pairs.items():
        table = (tables or {}).get(pid)
        comment = text_of(pid)
        if comment is None:
            continue
        gates = A.caller_gates(pair, table)
        if gates:
            gated += 1
            if all(A.states(comment, g) for g in gates):
                stated += 1
            else:
                missed.append(pid)
        else:
            ungated += 1
            if A.invented(pair, table, comment):
                invented += 1
    return {"gated": gated, "stated": stated, "ungated": ungated,
            "invented": invented, "missed": missed,
            "recall": stated / gated if gated else None,
            "invention_rate": invented / ungated if ungated else None}


def _pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=HERE / "data/NatSpecGold")
    ap.add_argument("--comgen-runs", type=Path,
                    default=HERE / "comgen/results/runs")
    ap.add_argument("--natcomgen-runs", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="G8,G1,C1")
    ap.add_argument("--kinds", default="function")
    ap.add_argument("--gold-only", action="store_true",
                    help="skip the run trees; report the corpus alone")
    ap.add_argument("--list-missed", type=int, default=0,
                    help="print N pair ids the first configuration missed")
    ap.add_argument("--out", type=Path, default=HERE / "results/functionwise")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    if args.natcomgen_runs is None:
        args.natcomgen_runs = args.corpus / "runs"

    every = scope.load_pairs(args.corpus)
    kinds = scope.parse_kinds(args.kinds)
    tables = scope.load_tables(args.corpus)
    if not tables:
        print("note: no sigma tables found — caller guards in the body will "
              "not be seen, only modifiers on the declaration",
              file=sys.stderr)

    doc = [f"# Caller gates — {args.split}, {'+'.join(kinds)}\n"]

    # coverage first: this metric is undefined where nothing is gated
    head, rows = ["split", "functions", "caller-gated", "share",
                  "gold states it"], []
    for s in ("train", "val", "test"):
        sub = scope.scoped(every, split=s, kinds=kinds)
        if not sub:
            continue
        m = measure(sub, lambda pid: gold_text(sub[pid]), tables)
        rows.append([s, str(len(sub)), str(m["gated"]),
                     _pct(m["gated"] / len(sub)), _pct(m["recall"])])
    doc.append("## Coverage\n\n" + markdown_table(head, rows))
    doc.append("\nThe metric is undefined on a split with no gated functions. "
               "Develop on train, report on test.\n")

    pairs = scope.scoped(every, split=args.split, kinds=kinds)
    gold = measure(pairs, lambda pid: gold_text(pairs[pid]), tables)

    results = {"gold": gold}
    head = ["system", "n scored", "caller-gated", "states the gate",
            "gate recall", "ungated", "invents one", "invention rate"]
    rows = [["gold (reference)", str(len(pairs)), str(gold["gated"]),
             str(gold["stated"]), _pct(gold["recall"]), str(gold["ungated"]),
             str(gold["invented"]), _pct(gold["invention_rate"])]]

    if not args.gold_only:
        run_rows: List[dict] = []
        for root in (args.comgen_runs, args.natcomgen_runs):
            if Path(root).exists():
                run_rows += load(Path(root), args.split)
        run_rows = [r for r in run_rows if r.get("pair_id") in pairs
                    and not r.get("error") and r.get("seed") == args.seed]
        by: Dict[str, Dict[str, str]] = {}
        for r in run_rows:
            by.setdefault(r.get("config"), {})[r["pair_id"]] = r.get("final") or ""
        for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
            got = by.get(name)
            if not got:
                print(f"note: no {args.split} rows for {name} at seed "
                      f"{args.seed}", file=sys.stderr)
                continue
            sub = {k: v for k, v in pairs.items() if k in got}
            m = measure(sub, lambda pid: got.get(pid), tables)
            results[name] = m
            rows.append([name, str(len(sub)), str(m["gated"]),
                         str(m["stated"]), _pct(m["recall"]), str(m["ungated"]),
                         str(m["invented"]), _pct(m["invention_rate"])])
            if args.list_missed and m["missed"]:
                doc.append(f"\n`{name}` missed: "
                           + ", ".join(f"`{p}`" for p in
                                       m["missed"][:args.list_missed]) + "\n")

    doc.append("\n## Gate recall\n\n" + markdown_table(head, rows))
    doc.append("\n`gate recall` is the share of caller-gated functions whose "
               "comment states the restriction; `invention rate` is the share "
               "of ungated functions whose comment claims one. Recall bought "
               "by invention is not an improvement, so the two are always "
               "read together. Neither uses the reference, so a configuration "
               "may exceed the gold row — which is the result worth having.\n")

    text = "\n".join(doc)
    print(text)
    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        path = next_path(args.out / f"gates_{args.split}.md")
        path.write_text(text, encoding="utf-8")
        print(f"\nwritten to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
