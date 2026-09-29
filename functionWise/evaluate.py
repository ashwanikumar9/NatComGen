"""Function-level evaluation of every configuration, in publishable form.

    python3 -m functionWise.evaluate --split test --all-seeds
    python3 -m functionWise.evaluate --split val --compare-normalizations
    python3 -m functionWise.evaluate --kinds callable --normalize none

What this does that `tools/evaluate_all.py` does not:

  * scores **functions only**, so the population matches SmartDoc, CCGIR,
    SCCLLM and SmartBT;
  * reports Ba/B1..B4 (the published convention) separately from cumulative
    BLEU-1..4, so the two are never averaged into each other;
  * scores a **notice-only** view as well as the whole comment, because the
    baselines emit a user notice and nothing else;
  * states the tokenisation, and can print the same rows under all three so
    the measurement component of a gap is visible.

Every row is scored on exactly the functions every configuration completed, so
a row-to-row difference is not partly a difference of function sets. Output is
versioned: an existing file is never overwritten.
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

from natspec_corpus.report import markdown_table              # noqa: E402
from natspec_corpus.versioning import next_path               # noqa: E402
from tools.bleu_table import load, matched_ids                # noqa: E402

from functionWise import metrics, scope, surface              # noqa: E402

PUBLISHED_CONTEXT = [
    ["SmartDoc (ASE'21)", "notice", "1k random", "47.39", "56.51", "46.78",
     "44.27", "43.08", "51.86", "-"],
]

MAIN_KEYS = ["n", "empty", "Ba", "B1", "B2", "B3", "B4",
             "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
             "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]
FIELD_KEYS = ["n", "Ba", "B1", "ROUGE-L", "METEOR"]


# --------------------------------------------------------------------------

def collect(rows: Sequence[dict], pairs: Dict[str, dict], config: str,
            ids: set, seeds: Optional[Sequence[int]], *, target: str,
            mode: str) -> tuple:
    """(references, hypotheses) for one configuration on one view.

    A pair whose gold is empty for this view is dropped -- a function with no
    @return cannot be scored on @return, and keeping it would score every
    configuration against the empty string and call the result agreement.
    The drop depends only on the reference, so it removes the same functions
    from every row.
    """
    refs, hyps = [], []
    for rec in rows:
        if rec.get("config") != config or rec.get("error"):
            continue
        if rec.get("pair_id") not in ids:
            continue
        if seeds is not None and rec.get("seed") not in seeds:
            continue
        pair = pairs[rec["pair_id"]]
        ref, hyp = surface.pair_strings(pair, rec.get("final") or "",
                                        target=target, mode=mode)
        if not ref.strip():
            continue
        refs.append(ref)
        hyps.append(hyp)
    return refs, hyps


def table_for(loaded, pairs, ids, seeds, *, target: str, mode: str,
              keys: Sequence[str]) -> tuple:
    headers = ["system", "config", "what it removes"] + list(keys)
    rows, results = [], {}
    for system, run_rows, order, by_name in loaded:
        for name in order:
            refs, hyps = collect(run_rows, pairs, name, ids, seeds,
                                 target=target, mode=mode)
            if not hyps:
                continue
            s = metrics.score(refs, hyps)
            results[f"{system}/{name}"] = s
            rows.append([system, name, by_name[name].label]
                        + metrics.cells(s, keys))
    return (headers, rows), results


def load_trees(args, pairs) -> list:
    from comgen.experiment import BY_NAME as G_BY, RUN_ORDER as G_ORDER
    from natspec_corpus.experiment import BY_NAME as C_BY, RUN_ORDER as C_ORDER
    trees = [("NatComGen", Path(args.natcomgen_runs), C_ORDER, C_BY),
             ("ComGen", Path(args.comgen_runs), G_ORDER, G_BY)]
    loaded = []
    for system, root, order, by_name in trees:
        if not root.exists():
            print(f"note: {root} does not exist - skipping {system}",
                  file=sys.stderr)
            continue
        rows = [r for r in load(root, args.split) if r.get("pair_id") in pairs]
        if rows:
            loaded.append((system, rows, order, by_name))
        else:
            print(f"note: no in-scope {args.split} rows under {root} "
                  f"- skipping {system}", file=sys.stderr)
    return loaded


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=HERE / "data/NatSpecGold")
    ap.add_argument("--comgen-runs", type=Path,
                    default=HERE / "comgen/results/runs")
    ap.add_argument("--natcomgen-runs", type=Path, default=None,
                    help="default: <corpus>/runs")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--kinds", default="function",
                    help="function (default) | callable | all | a,b,c")
    ap.add_argument("--normalize", default="paper", choices=surface.MODES,
                    help="paper (default) lowercases and drops punctuation; "
                         "none reproduces tools/evaluate_all.py")
    ap.add_argument("--compare-normalizations", action="store_true",
                    help="print the whole-comment table under all three")
    ap.add_argument("--out", type=Path, default=HERE / "results/functionwise")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    if args.natcomgen_runs is None:
        args.natcomgen_runs = args.corpus / "runs"

    kinds = scope.parse_kinds(args.kinds)
    every = scope.load_pairs(args.corpus)
    pairs = scope.scoped(every, split=args.split, kinds=kinds)
    if not pairs:
        raise SystemExit(f"no {args.split} pairs of kind(s) "
                         f"{', '.join(kinds)} in {args.corpus}")
    seeds = None if args.all_seeds else [args.seed]

    loaded = load_trees(args, pairs)
    if not loaded:
        raise SystemExit(
            "no run files found. Expected <corpus>/runs and "
            "comgen/results/runs to hold <split>.jsonl files; pass "
            "--comgen-runs / --natcomgen-runs if they live elsewhere.")

    ids = set.intersection(*[matched_ids(rows, seeds)
                             for _, rows, _, _ in loaded])
    ids &= set(pairs)
    if not ids:
        raise SystemExit("no in-scope function was completed by every "
                         "configuration")

    scope_str = "+".join(kinds)
    seed_str = "all seeds pooled" if args.all_seeds else f"seed {args.seed}"
    doc: List[str] = []
    doc.append(f"# Function-level evaluation - {args.split}, {scope_str}, "
               f"{seed_str}\n")
    doc.append(f"{len(ids)} declarations, common to every configuration, "
               f"scored with `--normalize {args.normalize}`.\n")

    h, r = scope.census(every, kinds)
    doc.append("## Scope\n\n" + markdown_table(h, r))
    doc.append("\n`*` marks a kind inside the scope. Everything else is "
               "excluded from every number below. This filter is the point of "
               "the table: the systems this work is compared against document "
               "functions, and Sigma(f) has no coverage for events or "
               "errors, so scoring them together attributes a fact-table gap "
               "to the generator.\n")

    h, r = scope.reference_lengths(pairs)
    doc.append("## Gold length, in scope\n\n" + markdown_table(h, r))
    doc.append("\nBLEU and METEOR both reward matching this distribution. A "
               "generator writing three sentences where the corpus writes one "
               "loses precision on every one of them for a reason unrelated "
               "to being wrong.\n")

    results: Dict[str, dict] = {}
    for target, title, note in (
            ("whole", "Whole comment",
             "Every field joined in canonical order. This is the row to put "
             "beside a paper that documents a whole comment."),
            ("notice", "@notice only",
             "The view that matches SmartDoc, CCGIR and SCCLLM, which emit a "
             "user notice and nothing else. This is the only genuinely "
             "like-for-like comparison with them.")):
        (h, r), res = table_for(loaded, pairs, ids, seeds, target=target,
                                mode=args.normalize, keys=MAIN_KEYS)
        if not r:
            continue
        results[target] = res
        doc.append(f"## {title}\n\n" + markdown_table(h, r) + f"\n\n{note}\n")

    per_field_rows, per_field_headers = [], None
    for field in ("notice", "dev", "param", "return"):
        (h, r), res = table_for(loaded, pairs, ids, seeds, target=field,
                                mode=args.normalize, keys=FIELD_KEYS)
        per_field_headers = ["field"] + h
        for row in r:
            per_field_rows.append([field] + row)
        results.setdefault("fields", {}).update(
            {f"{field}/{k}": v for k, v in res.items()})
    if per_field_rows:
        doc.append("## Per field\n\n"
                   + markdown_table(per_field_headers, per_field_rows)
                   + "\n\nCorpus BLEU per field, not the sentence BLEU in "
                     "`main.md`. `n` differs per field because a function "
                     "with no @return is not scored on @return.\n")

    if args.compare_normalizations:
        rows = []
        headers = None
        for mode in surface.MODES:
            (h, r), _ = table_for(loaded, pairs, ids, seeds, target="whole",
                                  mode=mode, keys=MAIN_KEYS)
            headers = ["normalize"] + h
            rows.extend([[mode] + row for row in r])
        doc.append("## Whole comment under each tokenisation\n\n"
                   + markdown_table(headers, rows)
                   + "\n\nThe spread across these three rows for one "
                     "configuration is the measurement component of any gap "
                     "to a published number. `none` is whitespace splitting, "
                     "as in `tools/evaluate_all.py`; `paper` lowercases and "
                     "drops punctuation; `identifiers` additionally splits "
                     "camelCase and snake_case.\n")

    doc.append("## Reading the BLEU columns\n\n"
               "`Ba` is cumulative BLEU-4. `B1..B4` are the **individual** "
               "n-gram precisions. Together they are the LeClair/McMillan "
               "reporting that SmartDoc's published table uses, and they are "
               "the numbers to place beside it.\n\n"
               "`BLEU-1..BLEU-4` are **cumulative**: the running geometric "
               "mean over orders 1..n. `BLEU-4` and `Ba` are the same "
               "statistic; `B2` and `BLEU-2` are not, and on the same "
               "predictions they differ widely. Never average across the two "
               "groups, and never quote one against a paper reporting the "
               "other.\n\n"
               "BLEU is corpus-level: n-gram counts pooled over the split, "
               "one brevity penalty. ROUGE and METEOR are per-comment means. "
               "METEOR is exact+stem with no WordNet stage.\n")

    doc.append("## Published figures, for orientation only\n\n"
               + markdown_table(
                   ["system", "target", "split", "Ba", "B1", "B2", "B3", "B4",
                    "ROUGE-L", "METEOR"], PUBLISHED_CONTEXT)
               + "\n\nThese are **not** comparable to the tables above: "
                 "different corpus, random rather than project-disjoint "
                 "split, and their own tokenisation. They are here so the "
                 "convention is visible, not so the numbers can be "
                 "subtracted. A like-for-like figure requires running their "
                 "released artefact on this corpus.\n")

    if len(ids) < 50:
        doc.append(f"\n> **Warning.** Only {len(ids)} declarations are common "
                   f"to every configuration. At this n, B3, B4 and ROUGE-2 "
                   f"are not stable statistics.\n")

    text = "\n".join(doc)
    print(text)

    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        target = next_path(args.out /
                           f"{args.split}_{scope_str}_{args.normalize}.md")
        target.write_text(text, encoding="utf-8")
        print(f"\nwritten to {target}", file=sys.stderr)
    if args.json:
        path = next_path(args.json)
        path.write_text(json.dumps(
            {"split": args.split, "kinds": list(kinds),
             "normalize": args.normalize, "n": len(ids),
             "seeds": "all" if args.all_seeds else [args.seed],
             "results": results}, indent=1), encoding="utf-8")
        print(f"written to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
