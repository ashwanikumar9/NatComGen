"""Corpus BLEU, matching what SmartDoc's `evaluate.py` computes.

Their script calls `nltk.translate.bleu_score.corpus_bleu` with default
arguments over whitespace-split lines: uniform 4-gram weights for `Ba`, and
one-hot weights for `B1`..`B4`. Reimplemented here rather than depended on,
for one reason that matters and one that does not.

The one that matters: nltk's default is **no smoothing**, and with no
smoothing a single hypothesis line that shares no 4-gram with its reference
sets `p4 = 0` and therefore drives the whole corpus score to zero — except
that corpus BLEU pools n-gram counts across the corpus first, so it survives
as long as *some* line has a 4-gram in common. That pooling is the whole
reason corpus BLEU and mean sentence BLEU are different numbers, and it is
exactly the distinction that made an earlier version of this project's
write-up report 0.036 where it meant 0.64. Writing it out leaves nowhere for
that confusion to hide.

The one that does not: nltk is a large dependency to add to a pipeline that
otherwise needs none.

`check_against_nltk` exists so the claim "matches their script" is tested
rather than asserted — where nltk happens to be installed, the two must
agree to 1e-9.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Sequence, Tuple


def _ngrams(toks: Sequence[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def _closest_len(ref_lens: Sequence[int], hyp_len: int) -> int:
    return min(ref_lens, key=lambda r: (abs(r - hyp_len), r))


def corpus_bleu(references: List[List[List[str]]],
                hypotheses: List[List[str]],
                weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)) -> float:
    """`references[i]` is a list of reference token lists for hypothesis i."""
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references vs "
                         f"{len(hypotheses)} hypotheses")
    order = len(weights)
    num = [0] * order
    den = [0] * order
    hyp_total = ref_total = 0

    for refs, hyp in zip(references, hypotheses):
        for i in range(order):
            n = i + 1
            h = _ngrams(hyp, n)
            # A hypothesis shorter than n has no n-grams at all. nltk still
            # charges it a denominator of 1 (`max(1, sum(counts.values()))`),
            # which quietly lowers p3 and p4 on a corpus with short lines —
            # here by 0.11 BLEU on SmartDoc's own output. Skipping those rows
            # instead would make this a different metric wearing their name.
            den[i] += max(1, sum(h.values()))
            if not h:
                continue
            merged: Counter = Counter()
            for r in refs:
                rc = _ngrams(r, n)
                for g, c in rc.items():
                    if c > merged[g]:
                        merged[g] = c
            num[i] += sum(min(c, merged[g]) for g, c in h.items())
        hyp_total += len(hyp)
        ref_total += _closest_len([len(r) for r in refs], len(hyp))

    if den[0] == 0 or num[0] == 0:
        return 0.0
    logs = 0.0
    for w, n_, d_ in zip(weights, num, den):
        if w == 0:
            continue
        if n_ == 0:
            return 0.0                       # nltk's unsmoothed behaviour
        logs += w * (math.log(n_) - math.log(d_))
    bp = 1.0 if hyp_total > ref_total else (
        0.0 if hyp_total == 0 else math.exp(1 - ref_total / hyp_total))
    return bp * math.exp(logs)


def leclair(ref_lines: Sequence[str], hyp_lines: Sequence[str]) -> dict:
    """`Ba, B1..B4` as percentages, rounded to 2dp — their exact reporting."""
    refs = [[r.split()] for r in ref_lines]
    preds = [h.split() for h in hyp_lines]
    w = {"Ba": (0.25, 0.25, 0.25, 0.25), "B1": (1, 0, 0, 0), "B2": (0, 1, 0, 0),
         "B3": (0, 0, 1, 0), "B4": (0, 0, 0, 1)}
    out = {k: round(corpus_bleu(refs, preds, v) * 100, 2) for k, v in w.items()}
    out["n"] = len(preds)
    return out


def check_against_nltk(ref_lines, hyp_lines) -> str:
    try:
        import nltk.translate.bleu_score as nb
    except ImportError:
        return "nltk not installed — skipped"
    refs = [[r.split()] for r in ref_lines]
    preds = [h.split() for h in hyp_lines]
    for name, w in (("Ba", (0.25,) * 4), ("B1", (1, 0, 0, 0)),
                    ("B2", (0, 1, 0, 0)), ("B3", (0, 0, 1, 0)),
                    ("B4", (0, 0, 0, 1))):
        a = corpus_bleu(refs, preds, w)
        b = nb.corpus_bleu(refs, preds, weights=w)
        if abs(a - b) > 1e-9:
            return f"MISMATCH {name}: ours {a!r}, nltk {b!r}"
    return "agrees with nltk to 1e-9"
