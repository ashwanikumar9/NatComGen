"""Whole-comment evaluation: corpus BLEU, ROUGE-1/2/L and METEOR.

The existing `main.md` scores each NatSpec field separately with sentence BLEU.
That is the right table for asking *which part* of a comment a configuration
gets right, and the wrong one for comparing against published work: nobody
reports per-field sentence BLEU, and on this project's own data it runs more
than an order of magnitude above the corpus figure. This tool scores the whole
comment as one string, the way every paper in this area does.

    python3 tools/evaluate_all.py                       # both architectures
    python3 tools/evaluate_all.py --all-seeds --json out.json

WHAT EACH NUMBER IS

  BLEU      Corpus BLEU-4: n-gram counts pooled over the whole split, one
            brevity penalty for the corpus. Computed by
            `benchmarks/smartdoc/bleu.py`, which reproduces SmartDoc's
            published 47.39 on their own data and agrees with nltk to 1e-9.
            B1..B4 are the individual n-gram precisions.
  ROUGE-1/2 Unigram and bigram F1, per comment, averaged.
  ROUGE-L   LCS-based F with beta=1.2, per comment, averaged — the same
            `evaluate.rouge_l` the per-field tables use.
  METEOR    METEOR 1.0: aligned unigram precision and recall combined as
            Fmean = 10PR/(R+9P), times (1 - 0.5*(chunks/matches)^3).

ABOUT METEOR AND WORDNET. Full METEOR has three matching stages: exact, stem,
and WordNet synonymy. This implements the first two. The synonymy stage needs a
WordNet download, which on a shared machine with no sudo is a dependency this
project cannot promise, and a metric that silently changes when a data file is
or is not present is worse than one that is honestly narrower. So the number
here is METEOR-exact+stem, it is reported under that name, and it is consistent
across runs and machines. `--check-nltk` compares against nltk's implementation
when it is installed; the gap is the synonymy stage and is expected.

Every row is scored on exactly the functions every configuration completed, so a
row-to-row difference is not partly a difference of function sets.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from natspec_corpus.evaluate import reference_fields, rouge_l               # noqa: E402
from natspec_corpus.report import markdown_table                           # noqa: E402
from tools.bleu_table import load, matched_ids                             # noqa: E402
from tools.overall_bleu import render, score as bleu_score                 # noqa: E402


# --------------------------------------------------------------------------
# ROUGE-1 / ROUGE-2
# --------------------------------------------------------------------------

def rouge_n(candidate: str, reference: str, n: int = 1) -> float:
    """F1 over n-gram counts. Clipped: a repeated n-gram counts once per
    occurrence in the reference, not once per occurrence in the candidate."""
    c, r = candidate.split(), reference.split()
    if len(c) < n or len(r) < n:
        return 0.0
    cg = Counter(tuple(c[i:i + n]) for i in range(len(c) - n + 1))
    rg = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
    overlap = sum((cg & rg).values())
    if not overlap:
        return 0.0
    p = overlap / sum(cg.values())
    rec = overlap / sum(rg.values())
    return 2 * p * rec / (p + rec)


# --------------------------------------------------------------------------
# METEOR (exact + stem)
# --------------------------------------------------------------------------

#: Longest first, so `-ing` is tried before `-g` would be. Deliberately short:
#: an aggressive stemmer invents matches, which inflates the metric in the one
#: direction nobody would notice.
_SUFFIXES = ("ization", "iveness", "fulness", "ousness", "ation", "ments",
             "ingly", "edly", "ment", "ness", "ions", "ing", "ies", "ed",
             "es", "ly", "s")


def stem(word: str) -> str:
    w = word.lower()
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def _align(hyp: Sequence[str], ref: Sequence[str]) -> List[Tuple[int, int]]:
    """Greedy two-stage alignment: exact matches first, then stems.

    Exact before stem matters. Aligning `deposits` to `deposit` on the stem
    pass, when an exact `deposits` was available further along the reference,
    both loses a true match and invents a chunk break.
    """
    pairs: List[Tuple[int, int]] = []
    used_h, used_r = set(), set()
    for key in (lambda w: w, stem):
        table: Dict[str, List[int]] = {}
        for j, word in enumerate(ref):
            if j not in used_r:
                table.setdefault(key(word), []).append(j)
        for i, word in enumerate(hyp):
            if i in used_h:
                continue
            slot = table.get(key(word))
            while slot and slot[0] in used_r:
                slot.pop(0)
            if slot:
                j = slot.pop(0)
                used_h.add(i)
                used_r.add(j)
                pairs.append((i, j))
    return sorted(pairs)


def _chunks(pairs: Sequence[Tuple[int, int]]) -> int:
    """Contiguous runs. A run continues only when both sides advance by one."""
    if not pairs:
        return 0
    n = 1
    for (i0, j0), (i1, j1) in zip(pairs, pairs[1:]):
        if not (i1 == i0 + 1 and j1 == j0 + 1):
            n += 1
    return n


def meteor(candidate: str, reference: str, *, alpha: float = 0.9,
           beta: float = 3.0, gamma: float = 0.5) -> float:
    """METEOR 1.0 with exact and stem matching. See the module docstring."""
    hyp, ref = candidate.split(), reference.split()
    if not hyp or not ref:
        return 0.0
    pairs = _align(hyp, ref)
    m = len(pairs)
    if not m:
        return 0.0
    p, r = m / len(hyp), m / len(ref)
    fmean = p * r / (alpha * p + (1 - alpha) * r)
    penalty = gamma * (_chunks(pairs) / m) ** beta
    return fmean * (1 - penalty)


def check_nltk(refs: Sequence[str], hyps: Sequence[str]) -> str:
    """nltk's METEOR for comparison. A gap is the WordNet synonymy stage."""
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
    except Exception as e:                                   # noqa: BLE001
        return f"nltk METEOR unavailable ({type(e).__name__}); skipped"
    ours = sum(meteor(h, r) for r, h in zip(refs, hyps)) / max(len(hyps), 1)
    theirs = sum(nltk_meteor([r.split()], h.split())
                 for r, h in zip(refs, hyps)) / max(len(hyps), 1)
    return (f"METEOR ours {ours * 100:.2f} vs nltk {theirs * 100:.2f} "
            f"(difference {abs(ours - theirs) * 100:.2f} — the synonymy stage)")


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

def texts(rows: Sequence[dict], pairs: Dict[str, dict], config: str,
          ids: set, seeds: Optional[Sequence[int]]
          ) -> Tuple[List[str], List[str]]:
    """(references, predictions) as whole-comment strings, same order."""
    refs, hyps = [], []
    for rec in rows:
        if rec.get("config") != config or rec.get("error"):
            continue
        if rec["pair_id"] not in ids:
            continue
        if seeds is not None and rec.get("seed") not in seeds:
            continue
        pair = pairs[rec["pair_id"]]
        fields = reference_fields(pair)
        ref = " ".join(v.strip() for v in fields.values() if v and v.strip())
        refs.append(ref.strip())
        hyps.append(render(rec.get("final") or ""))
    return refs, hyps


def evaluate(refs: Sequence[str], hyps: Sequence[str]) -> dict:
    out = dict(bleu_score(list(refs), list(hyps)))
    n = max(len(hyps), 1)
    out["ROUGE-1"] = round(sum(rouge_n(h, r, 1)
                               for r, h in zip(refs, hyps)) / n * 100, 2)
    out["ROUGE-2"] = round(sum(rouge_n(h, r, 2)
                               for r, h in zip(refs, hyps)) / n * 100, 2)
    out["ROUGE-L"] = round(sum(rouge_l(h, r)
                               for r, h in zip(refs, hyps)) / n * 100, 2)
    out["METEOR"] = round(sum(meteor(h, r)
                              for r, h in zip(refs, hyps)) / n * 100, 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=HERE / "data/NatSpecGold")
    ap.add_argument("--comgen-runs", type=Path,
                    default=HERE / "comgen/results/runs")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--check-nltk", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    pairs = {json.loads(l)["id"]: json.loads(l)
             for l in (args.corpus / "pairs.jsonl").read_text(
                 encoding="utf-8").splitlines() if l.strip()}
    seeds = None if args.all_seeds else [args.seed]

    from comgen.experiment import BY_NAME as G_BY, RUN_ORDER as G_ORDER
    from natspec_corpus.experiment import BY_NAME as C_BY, RUN_ORDER as C_ORDER

    loaded = []
    for system, root, order, by_name in (
            ("NatComGen", args.corpus / "runs", C_ORDER, C_BY),
            ("ComGen", args.comgen_runs, G_ORDER, G_BY)):
        if not Path(root).exists():
            print(f"note: {root} does not exist — skipping {system}")
            continue
        rows = [r for r in load(Path(root), args.split)
                if r.get("pair_id") in pairs]
        if rows:
            loaded.append((system, rows, order, by_name))
    if not loaded:
        raise SystemExit("nothing to evaluate")

    ids = set.intersection(*[matched_ids(rows, seeds)
                             for _, rows, _, _ in loaded])
    if not ids:
        raise SystemExit("no function was completed by every configuration")

    headers = ["system", "config", "what it removes", "n", "BLEU", "B1", "B2",
               "B3", "B4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]
    table, results = [], {}
    for system, rows, order, by_name in loaded:
        for name in order:
            refs, hyps = texts(rows, pairs, name, ids, seeds)
            if not hyps:
                continue
            s = evaluate(refs, hyps)
            results[f"{system}/{name}"] = s
            table.append([system, name, by_name[name].label, s["n"]]
                         + [f"{s[k]:.2f}" for k in
                            ("BLEU", "B1", "B2", "B3", "B4", "ROUGE-1",
                             "ROUGE-2", "ROUGE-L", "METEOR")])

    scope = "all seeds pooled" if args.all_seeds else f"seed {args.seed}"
    print(f"# Whole-comment evaluation — {args.split}, {scope}, "
          f"{len(ids)} functions common to every configuration\n")
    print(markdown_table(headers, table))
    print("\nBLEU is corpus BLEU-4 (pooled counts, one brevity penalty) — the "
          "figure comparable to a published one. ROUGE and METEOR are "
          "per-comment and averaged. METEOR is exact+stem matching with no "
          "WordNet stage; see the module docstring.")
    if args.check_nltk:
        first = loaded[0]
        refs, hyps = texts(first[1], pairs, first[2][0], ids, seeds)
        print("\n" + check_nltk(refs, hyps))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
