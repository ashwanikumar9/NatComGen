"""What the corpus's own comments look like, and the rules that follow.

The surface metrics reward matching the reference's phrasing, and nothing in
the pipeline optimises for that: Sigma(f) fixes what is said, not how. This
reads the training split and reports the house style as numbers -- opening
verb, length, punctuation, article use, backticks -- then emits those findings
as a rules block for the generator prompt.

    python3 -m functionWise.conventions --split train
    python3 -m functionWise.conventions --split train --emit-rules

Only the TRAIN split is read by default, and that is not a detail: mining the
style of val or test is fitting the evaluation set, and every number produced
afterwards would be worth nothing.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from natspec_corpus.report import markdown_table          # noqa: E402
from natspec_corpus.versioning import next_path           # noqa: E402

from tools.bleu_table import load                        # noqa: E402

from functionWise import scope, surface                   # noqa: E402

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_SENT = re.compile(r"[.!?]+(?:\s|$)")


def _texts(pairs: Dict[str, dict]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"notice": [], "dev": [], "param": [],
                                 "return": []}
    for p in pairs.values():
        for k in ("notice", "dev"):
            if (p.get(k) or "").strip():
                out[k].append(p[k].strip())
        for t in (p.get("params") or {}).values():
            if (t or "").strip():
                out["param"].append(t.strip())
        for r in (p.get("returns") or []):
            t = (r.get("text") or "").strip()
            if t:
                out["return"].append(t)
    return out


def _shape(texts: Sequence[str]) -> dict:
    if not texts:
        return {}
    lens = sorted(len(t.split()) for t in texts)
    firsts = Counter()
    for t in texts:
        m = _WORD.search(t)
        if m:
            firsts[m.group(0)] += 1
    return {
        "n": len(texts),
        "median_words": median(lens),
        "p10": lens[max(0, int(len(lens) * .10) - 1)],
        "p90": lens[min(len(lens) - 1, int(len(lens) * .90))],
        "ends_period": sum(1 for t in texts if t.rstrip().endswith(".")) / len(texts),
        "one_sentence": sum(1 for t in texts
                            if len([s for s in _SENT.split(t) if s.strip()]) <= 1
                            ) / len(texts),
        "starts_the": sum(1 for t in texts
                          if t.lower().startswith("the ")) / len(texts),
        "starts_this_function": sum(
            1 for t in texts
            if re.match(r"this (function|method)", t, re.I)) / len(texts),
        "has_backticks": sum(1 for t in texts if "`" in t) / len(texts),
        "third_person_s": sum(1 for t in texts
                              if re.match(r"[A-Z][a-z]+s\b", t)) / len(texts),
        "firsts": firsts.most_common(10),
    }


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def rules_block(shapes: Dict[str, dict]) -> str:
    """The findings as a prompt block. Every number here is measured."""
    n, pa, re_, dv = (shapes.get(k) or {} for k in
                      ("notice", "param", "return", "dev"))
    lines = ["HOUSE STYLE (measured on the training split — match it):"]
    if n:
        verbs = ", ".join(w for w, _ in n["firsts"][:6])
        lines.append(
            f"  - @notice: ONE sentence ({_pct(n['one_sentence'])} of the "
            f"corpus is), about {n['median_words']:.0f} words, at most "
            f"{n['p90']}. Third person singular, opening on the verb: "
            f"{verbs}. Never open with \"This function\" "
            f"({_pct(n['starts_this_function'])} of the corpus does)."
            + (" End with a period." if n["ends_period"] > .7 else ""))
    if pa:
        lines.append(
            f"  - @param: a noun phrase of about {pa['median_words']:.0f} "
            f"words, at most {pa['p90']}"
            + (f", opening with \"The\" ({_pct(pa['starts_the'])} of the "
               f"corpus does)" if pa["starts_the"] > .4 else "")
            + ("." if pa["ends_period"] > .5 else ", with no closing period.")
            + " Never restate the type.")
    if re_:
        lines.append(
            f"  - @return: about {re_['median_words']:.0f} words, at most "
            f"{re_['p90']}"
            + (f", opening with \"The\"" if re_["starts_the"] > .4 else "")
            + ".")
    if dv:
        lines.append(
            f"  - @dev: about {dv['median_words']:.0f} words. This is where "
            f"caller obligations and revert conditions go — not a second "
            f"summary of what the function does.")
    ident = max((s.get("has_backticks", 0) for s in shapes.values()
                 if s), default=0)
    lines.append(
        "  - Wrap identifiers in backticks."
        if ident > .25 else
        "  - Do not wrap identifiers in backticks; the corpus writes them "
        "plain.")
    lines.append("  - Write the whole comment to a budget: if every tag is "
                 "present it should still read as documentation, not a report.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=HERE / "data/NatSpecGold")
    ap.add_argument("--split", default="train")
    ap.add_argument("--kinds", default="function")
    ap.add_argument("--emit-rules", action="store_true",
                    help="print only the prompt block")
    ap.add_argument("--compare-runs", default="",
                    help="configs to measure against the gold shape, "
                         "e.g. G8,G1,C1")
    ap.add_argument("--comgen-runs", type=Path,
                    default=HERE / "comgen/results/runs")
    ap.add_argument("--natcomgen-runs", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "results/functionwise")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    if args.split != "train":
        print(f"warning: mining style from '{args.split}'. Only 'train' is "
              f"safe; anything else fits the evaluation set.", file=sys.stderr)

    pairs = scope.scoped(scope.load_pairs(args.corpus), split=args.split,
                         kinds=scope.parse_kinds(args.kinds))
    shapes = {k: _shape(v) for k, v in _texts(pairs).items()}

    if args.emit_rules:
        print(rules_block(shapes))
        return 0

    if args.compare_runs:
        doc = [f"# Generated shape against the corpus — {args.split}\n"]
        head = ["field", "system", "n", "median words", "vs gold", "p90",
                "one sentence", "starts 'The'"]
        rows = []
        runs = args.natcomgen_runs or (args.corpus / "runs")
        rrows = []
        for root in (args.comgen_runs, runs):
            if Path(root).exists():
                rrows += load(Path(root), args.split)
        rrows = [r for r in rrows if r.get("pair_id") in pairs
                 and not r.get("error") and r.get("seed") == args.seed]
        by = {}
        for r in rrows:
            by.setdefault(r.get("config"), []).append(r.get("final") or "")
        for field in ("notice", "dev", "param", "return"):
            g = shapes.get(field)
            if not g:
                continue
            rows.append([field, "gold", str(g["n"]),
                         f"{g['median_words']:.0f}", "—", str(g["p90"]),
                         _pct(g["one_sentence"]), _pct(g["starts_the"])])
            for name in [c.strip() for c in args.compare_runs.split(",")
                         if c.strip()]:
                finals = by.get(name)
                if not finals:
                    continue
                texts = []
                for f in finals:
                    vals = surface.hypothesis_view(f, field if field in
                                                   ("notice", "dev") else field)
                    texts += [v for v in vals.values() if v.strip()]
                s = _shape(texts)
                if not s:
                    continue
                ratio = s["median_words"] / max(g["median_words"], 1)
                rows.append([field, name, str(s["n"]),
                             f"{s['median_words']:.0f}", f"{ratio:.2f}x",
                             str(s["p90"]), _pct(s["one_sentence"]),
                             _pct(s["starts_the"])])
        doc.append(markdown_table(head, rows))
        doc.append("\n`vs gold` is the ratio of median length. BLEU has no "
                   "verbosity penalty — it charges extra words through "
                   "precision instead — so a ratio above about 1.3 is losing "
                   "points on every n-gram order for a reason unrelated to "
                   "being wrong.\n")
        text = "\n".join(doc)
        print(text)
        if not args.no_write:
            args.out.mkdir(parents=True, exist_ok=True)
            path = next_path(args.out / f"shape_{args.split}.md")
            path.write_text(text, encoding="utf-8")
            print(f"\nwritten to {path}", file=sys.stderr)
        return 0

    head = ["field", "n", "median", "p10", "p90", "one sentence", "ends '.'",
            "starts 'The'", "\"This function\"", "backticks"]
    rows = []
    for k in ("notice", "dev", "param", "return"):
        s = shapes.get(k)
        if not s:
            continue
        rows.append([k, str(s["n"]), f"{s['median_words']:.0f}", str(s["p10"]),
                     str(s["p90"]), _pct(s["one_sentence"]),
                     _pct(s["ends_period"]), _pct(s["starts_the"]),
                     _pct(s["starts_this_function"]), _pct(s["has_backticks"])])
    doc = [f"# House style — {args.split}, functions\n",
           "## Shape\n\n" + markdown_table(head, rows)]
    for k in ("notice", "param", "return"):
        s = shapes.get(k)
        if s and s["firsts"]:
            doc.append(f"\n**{k} opens with** — "
                       + ", ".join(f"`{w}` ×{c}" for w, c in s["firsts"]))
    doc.append("\n## Rules block\n\n```\n" + rules_block(shapes) + "\n```\n")
    doc.append("Paste into the generator prompt. Every number in it is "
               "measured on this split, so it can be regenerated when the "
               "corpus changes rather than maintained by hand.\n")

    text = "\n".join(doc)
    print(text)
    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        path = next_path(args.out / f"conventions_{args.split}.md")
        path.write_text(text, encoding="utf-8")
        print(f"\nwritten to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
