"""SmartDoc's released artefact, re-scored.

Their package ships the test set, the references and every system's raw
predictions -- for the random split (RQ1) and for a five-fold cross-project
split their paper barely discusses. That is enough to ask three questions
their tables do not answer, with no model and no GPU:

  reproduction   how often a system emits the reference VERBATIM
  leakage        how often a test reference appears in the training file
  gates          whether the generated notice says who may call the function

The first two matter because BLEU cannot tell reproduction from summarisation,
and this corpus contains a great deal of copied boilerplate -- OpenZeppelin
vesting, ERC20 -- so splitting by contract does not separate the comments.

    python3 -m functionWise.smartdoc --pkg ~/Ashwani/MTP/smartdoc-pkg

The BLEU here is `benchmarks/smartdoc/bleu.corpus_bleu`, which reproduces
their published 47.39 / 56.51 / 43.08 from this artefact exactly -- so the
subset figures below are computed in their own convention, not ours.

ONE CAVEAT, STATED UP FRONT. Excluding the cases a system reproduced verbatim
changes the estimand: it answers "how well does it do when it is actually
generating", not "how well does it do". It is a diagnostic and belongs beside
the headline number, never instead of it.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from benchmarks.smartdoc.bleu import corpus_bleu                 # noqa: E402
from natspec_corpus import access as A                           # noqa: E402
from natspec_corpus.report import markdown_table                 # noqa: E402
from natspec_corpus.versioning import next_path                  # noqa: E402

SYSTEMS = ("attendgru", "ast-attendgru", "re2com", "smartdoc")
FOLDS = tuple(f"cross_project/fold_{i}" for i in range(1, 6))
W = {"Ba": (0.25,) * 4, "B1": (1, 0, 0, 0), "B2": (0, 1, 0, 0),
     "B3": (0, 0, 1, 0), "B4": (0, 0, 0, 1)}


def lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def bleu(pairs: Sequence[Tuple[str, str]], key: str = "Ba") -> Optional[float]:
    if not pairs:
        return None
    refs = [[r.split()] for r, _ in pairs]
    hyps = [h.split() for _, h in pairs]
    return round(corpus_bleu(refs, hyps, W[key]) * 100, 2)


def aligned(pkg: Path, split: str, system: str) -> List[Tuple[str, str]]:
    ref = lines(pkg / "final_results" / split / "ref.txt")
    hyp = lines(pkg / "final_results" / split / f"{system}.out")
    n = min(len(ref), len(hyp))
    return list(zip(ref[:n], hyp[:n]))


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", type=Path, required=True,
                    help="the cloned xing-hu/SmartDoc repository, dataset.zip "
                         "already unzipped in place")
    ap.add_argument("--out", type=Path, default=HERE / "results/functionwise")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    pkg = args.pkg.expanduser()
    if not (pkg / "final_results").is_dir():
        raise SystemExit(f"{pkg} has no final_results/ — clone "
                         f"https://github.com/xing-hu/SmartDoc and unzip "
                         f"dataset.zip inside it")
    ds = pkg / "dataset"
    if not (ds / "test" / "test.token.code").exists():
        raise SystemExit(f"run `unzip -o dataset.zip -d .` inside {pkg} first")

    doc = ["# SmartDoc's released artefact, re-scored\n",
           "Computed from `xing-hu/SmartDoc` with "
           "`benchmarks/smartdoc/bleu.corpus_bleu`, which reproduces their "
           "published RQ1 figures exactly from these same files.\n"]

    # ---- 1. every system, both splits ------------------------------------
    head = ["split", "system", "n", "Ba", "B1", "B4", "reproduced verbatim"]
    rows = []
    for label, splits in (("RQ1 (random)", ["RQ1"]),
                          ("cross-project (5 folds)", list(FOLDS))):
        for system in SYSTEMS:
            per, total, exact = {k: [] for k in W}, 0, 0
            for split in splits:
                pairs = aligned(pkg, split, system)
                for k in W:
                    per[k].append(bleu(pairs, k) or 0.0)
                total += len(pairs)
                exact += sum(1 for r, h in pairs if r.strip() == h.strip())
            mean = {k: sum(v) / len(v) for k, v in per.items()}
            rows.append([label, system, str(total), _fmt(mean["Ba"]),
                         _fmt(mean["B1"]), _fmt(mean["B4"]),
                         f"{exact / total * 100:.1f}%"])
    doc.append("## Every system, both splits\n\n" + markdown_table(head, rows))
    doc.append("\n`reproduced verbatim` is the share of test functions whose "
               "output is character-identical to the reference. Across folds "
               "the BLEU ranking of the four systems is their reproduction "
               "ranking, which is what BLEU cannot distinguish from quality.\n")

    # ---- 2. SmartDoc, with the reproduced cases removed ------------------
    rq1 = aligned(pkg, "RQ1", "smartdoc")
    train = {l.strip() for l in lines(ds / "train" / "train.token.nl")}
    cross: List[Tuple[str, str]] = []
    for split in FOLDS:
        cross += aligned(pkg, split, "smartdoc")

    def subset(pairs, drop_exact=False, drop_leaked=False):
        out = pairs
        if drop_exact:
            out = [(r, h) for r, h in out if r.strip() != h.strip()]
        if drop_leaked:
            out = [(r, h) for r, h in out if r.strip() not in train]
        return out

    head = ["split", "subset", "n", "Ba"]
    rows = []
    for label, pairs, leak in (("RQ1", rq1, True),
                               ("cross-project", cross, False)):
        rows.append([label, "as published", str(len(pairs)),
                     _fmt(bleu(pairs))])
        s = subset(pairs, drop_exact=True)
        rows.append([label, "excluding verbatim reproductions", str(len(s)),
                     _fmt(bleu(s))])
        if leak:
            s = subset(pairs, drop_leaked=True)
            rows.append([label, "references absent from train", str(len(s)),
                         _fmt(bleu(s))])
            s = subset(pairs, drop_exact=True, drop_leaked=True)
            rows.append([label, "neither leaked nor reproduced", str(len(s)),
                         _fmt(bleu(s))])
    doc.append("\n## SmartDoc, by subset\n\n" + markdown_table(head, rows))
    dup = sum(1 for r, _ in rq1 if r.strip() in train)
    doc.append(f"\n{dup} of {len(rq1)} RQ1 references "
               f"({dup / len(rq1) * 100:.1f}%) appear verbatim in "
               f"`train.token.nl`. Splitting by contract does not fix this: "
               f"the corpus is full of copied boilerplate, so the "
               f"cross-project folds reproduce *more*, not less.\n")

    # ---- 3. caller gates on their own test set ---------------------------
    code = lines(ds / "test" / "test.token.code")
    ref = [r for r, _ in rq1]
    hyp = [h for _, h in rq1]
    n = min(len(code), len(ref), len(hyp))
    gated = []
    for c, r, h in zip(code[:n], ref[:n], hyp[:n]):
        gs = A.caller_gates({"code": c}, None)
        if gs:
            gated.append((gs, r, h))
    if gated:
        head = ["comment", "gated functions", "states the gate", "rate"]
        rows = []
        for label, idx in (("gold reference", 1), ("SmartDoc output", 2)):
            ok = sum(1 for gs, *t in gated
                     if all(A.states(t[idx - 1], g) for g in gs))
            rows.append([label, str(len(gated)), str(ok),
                         f"{ok / len(gated) * 100:.0f}%"])
        doc.append("\n## Caller gates, on SmartDoc's own test set\n\n"
                   + markdown_table(head, rows))
        doc.append("\nRead each system against **its own corpus's gold**, not "
                   "across corpora: their references state the restriction far "
                   "less often than NatSpecGold's do, so the raw rates are not "
                   "comparable. What is comparable is the gap. A system trained "
                   "to imitate its references cannot exceed them; one grounded "
                   "in a fact table can.\n")

    text = "\n".join(doc)
    print(text)
    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        path = next_path(args.out / "smartdoc_rescored.md")
        path.write_text(text, encoding="utf-8")
        print(f"\nwritten to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
