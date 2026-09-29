# functionWise — function-level evaluation

`tools/evaluate_all.py` answers *which configuration of ours is best*. This
package answers a different question: *what number may be printed next to
SmartDoc's 47.39*. Those need different scopes and different conventions, and
mixing them produces a table that does not survive review.

## What it changes

**Scope.** Functions only. NatSpecGold holds 860 pairs and 175 of them are
events, errors, modifiers, constructors and `receive`. Every system this work
is compared against — SmartDoc, CCGIR, SCCLLM, SmartBT — documents functions.
Scoring a mixed population against their numbers compares two different
things, and it drags in exactly the declarations Σ(f) has no coverage for, so
a fact-table gap shows up as a generator weakness.

| split | functions | total | kept |
|---|---|---|---|
| train | 451 | 519 | 87% |
| val | 82 | 148 | 55% |
| test | 152 | 193 | 79% |

Val roughly halves. That is the cost, and `significance.py` exists to measure
whether it is worth paying.

**Convention.** `Ba, B1..B4` is the LeClair/McMillan reporting that SmartDoc's
published table uses: `Ba` is cumulative BLEU-4 and `B1..B4` are the
**individual** n-gram precisions. `BLEU-1..BLEU-4` are **cumulative** — the
running geometric mean over orders 1..n. `Ba` and `BLEU-4` are the same
statistic. `B2` and `BLEU-2` are not, and on the same predictions they differ
widely. Both groups are printed, never averaged together.

**Surface.** The existing tables split on whitespace, so `amount.` and
`amount` are different tokens and every backtick costs a match.
`--normalize paper` (the default) lowercases and strips punctuation, as the
code-summarisation literature does. `--normalize none` reproduces
`tools/evaluate_all.py`. `--normalize identifiers` additionally splits
camelCase and snake_case. The mode is named in every report.

**Views.** The whole comment *and* `@notice` alone. The notice view is the
only genuinely like-for-like comparison with SmartDoc, CCGIR and SCCLLM, which
emit a user notice and nothing else; scoring a four-tag block against their
one-line references loses precision for a reason that is not quality.

## Running it

```bash
# the main table — held-out split, every seed
python3 -m functionWise.evaluate --split test --all-seeds

# how much of the gap to a published number is tokenisation
python3 -m functionWise.evaluate --split val --compare-normalizations

# reproduce the old numbers, to see exactly what the scope changed
python3 -m functionWise.evaluate --split val --kinds all --normalize none

# do the ablation findings survive the halved sample?
python3 -m functionWise.significance --split val --baseline G1
python3 -m functionWise.significance --split val --baseline G8
```

Output goes to `results/functionwise/`, versioned — an existing file is never
overwritten, it becomes `_2`, `_3`.

## Reading `significance.py`

It runs the ablation tests under both scopes and prints what changed. A
**LOST** row means a finding that was significant over all declarations is not
significant over functions alone. That is not a reason to abandon the scope:
it says the effect lived partly in the events, errors and modifiers the scope
removes, which is a finding of its own and belongs in the write-up. A result
that survives a halved sample is stronger than it was before.

`significant` there means Holm-adjusted p < 0.05 **and** a difference larger
than the seed-to-seed spread. Both conditions, always.

## What it deliberately does not do

* It does not re-implement a metric. `corpus_bleu` is the repository's own,
  which reproduces SmartDoc's published 47.39 from their released output and
  agrees with nltk to 1e-9; ROUGE-L and METEOR come from `natspec_corpus` and
  `tools`. A second implementation would be a second source of truth.
* It does not change the ablation metric. Claim-support rate with a notice-BLEU
  fallback is what `comgen/report.py` uses; changing it here would make these
  p-values incomparable with the existing tables.
* It does not claim comparability with published figures. Those appear in the
  report under "for orientation only", because a like-for-like number requires
  running their released artefact on this corpus — different corpus, random
  rather than project-disjoint split, and their own tokenisation stand between
  the two otherwise.

## Tests

```bash
python3 -m pytest functionWise/tests -q
```

Fourteen tests, including two end-to-end runs over a synthetic corpus and run
tree — so the package is verifiable on a machine that holds no results, which
is every machine except the GPU box.
