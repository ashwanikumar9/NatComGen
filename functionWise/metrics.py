"""The metrics, in the conventions the related work actually uses.

The trap this module exists to close: "BLEU-1" names two different statistics
in this literature. SmartDoc's table reports `Ba, B1, B2, B3, B4` where Ba is
cumulative BLEU-4 and B1..B4 are the *individual* n-gram precisions -- the
LeClair/McMillan reporting. Cumulative BLEU-n (the running geometric mean) is
the other reading, and on the same predictions the two disagree by a wide
margin at n>1. Putting one next to the other is an error a reviewer finds in
thirty seconds.

So both are computed, both are labelled, and the report prints them in
separate column groups that are never averaged together.

BLEU is corpus-level throughout: n-gram counts pooled over the split, one
brevity penalty. ROUGE and METEOR are per-comment and averaged, as everyone
reports them. The implementations are the repository's own -- `corpus_bleu`
reproduces SmartDoc's published 47.39 from their released output and agrees
with nltk to 1e-9; METEOR is exact+stem with no WordNet stage.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from benchmarks.smartdoc.bleu import corpus_bleu            # noqa: E402
from natspec_corpus.evaluate import rouge_l                 # noqa: E402
from tools.evaluate_all import meteor, rouge_n              # noqa: E402

#: SmartDoc / LeClair reporting: Ba is cumulative BLEU-4, B1..B4 are the
#: individual n-gram precisions. These are the numbers to place beside a
#: published table.
PUBLISHED = {"Ba": (0.25, 0.25, 0.25, 0.25), "B1": (1, 0, 0, 0),
             "B2": (0, 1, 0, 0), "B3": (0, 0, 1, 0), "B4": (0, 0, 0, 1)}

#: Cumulative BLEU-n: the running geometric mean over orders 1..n. Reported
#: separately and never mixed with the group above.
CUMULATIVE = {"BLEU-1": (1, 0, 0, 0),
              "BLEU-2": (0.5, 0.5, 0, 0),
              "BLEU-3": (1 / 3, 1 / 3, 1 / 3, 0),
              "BLEU-4": (0.25, 0.25, 0.25, 0.25)}

COLUMNS = (["n", "empty"] + list(PUBLISHED) + list(CUMULATIVE)
           + ["ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"])


def _bleu(refs: Sequence[str], hyps: Sequence[str], weights) -> float:
    r = [[x.split()] for x in refs]
    h = [x.split() for x in hyps]
    return round(corpus_bleu(r, h, tuple(weights)) * 100, 2)


def score(refs: Sequence[str], hyps: Sequence[str]) -> Dict[str, float]:
    """Every column, for one system on one scope.

    An empty hypothesis stays in the corpus and scores zero. Dropping the
    cases where a configuration produced nothing would score configurations
    over different segment sets, which is the same mistake as an unmatched
    function set one level down.
    """
    refs, hyps = list(refs), list(hyps)
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references vs {len(hyps)} hypotheses")
    out: Dict[str, float] = {"n": len(hyps),
                             "empty": sum(1 for h in hyps if not h.strip())}
    if not hyps:
        return {**out, **{k: 0.0 for k in COLUMNS if k not in ("n", "empty")}}
    for name, w in PUBLISHED.items():
        out[name] = _bleu(refs, hyps, w)
    for name, w in CUMULATIVE.items():
        out[name] = _bleu(refs, hyps, w)
    n = len(hyps)
    out["ROUGE-1"] = round(sum(rouge_n(h, r, 1)
                               for r, h in zip(refs, hyps)) / n * 100, 2)
    out["ROUGE-2"] = round(sum(rouge_n(h, r, 2)
                               for r, h in zip(refs, hyps)) / n * 100, 2)
    out["ROUGE-L"] = round(sum(rouge_l(h, r)
                               for r, h in zip(refs, hyps)) / n * 100, 2)
    out["METEOR"] = round(sum(meteor(h, r)
                              for r, h in zip(refs, hyps)) / n * 100, 2)
    return out


def cells(s: Dict[str, float], keys: Sequence[str]) -> list:
    out = []
    for k in keys:
        v = s.get(k)
        out.append("-" if v is None else
                   (str(v) if k in ("n", "empty") else f"{v:.2f}"))
    return out
