"""Significance testing for paired configurations.

This module exists because the paper this work builds on does not have one.
SAGE reports single runs at near-zero temperature with no significance testing
anywhere, on 141 labelled samples. Adding it here costs almost no compute and
is the cheapest place a careful thesis beats a published system.

Three decisions are baked in, and each rules out a tempting mistake:

**Everything is paired.** Two configurations see the same functions, so a
comparison is a per-function difference, never two independent means. An
unpaired test on 193 functions throws away the pairing that makes the
comparison sensitive in the first place.

**Seeds are a separate axis from functions.** Bootstrapping over functions
answers "would another sample of functions show this?"; the seed spread
answers "would another run show this?". A difference smaller than the
seed-to-seed spread is not a difference, whatever the p-value says, and
`describe` reports both so that cannot be glossed over.

**Effect size travels with every p-value.** With 193 functions a significant
BLEU difference of 0.004 is still nothing.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


class StatsError(ValueError):
    """The inputs cannot support the test asked of them."""


# --------------------------------------------------------------------------
# paired comparison
# --------------------------------------------------------------------------

@dataclass
class Comparison:
    n: int
    mean_a: float
    mean_b: float
    diff: float                 # b - a, in the metric's own units
    ci_low: float
    ci_high: float
    p_value: float
    cliffs_delta: float
    test: str

    @property
    def significant(self) -> bool:
        """Zero outside the interval AND the test agreeing. Both, because a
        bootstrap interval and a rank test can disagree on skewed data, and
        the honest reading is the conservative one."""
        return (self.ci_low > 0 or self.ci_high < 0) and self.p_value < 0.05

    @property
    def magnitude(self) -> str:
        d = abs(self.cliffs_delta)
        return ("negligible" if d < 0.147 else "small" if d < 0.33
                else "medium" if d < 0.474 else "large")

    def to_dict(self) -> dict:
        return {"n": self.n, "mean_a": self.mean_a, "mean_b": self.mean_b,
                "diff": self.diff, "ci_low": self.ci_low,
                "ci_high": self.ci_high, "p_value": self.p_value,
                "cliffs_delta": self.cliffs_delta, "test": self.test,
                "significant": self.significant, "magnitude": self.magnitude}


def _pairs(a: Dict[str, float], b: Dict[str, float]) -> Tuple[List[float],
                                                              List[float]]:
    keys = sorted(set(a) & set(b))
    if not keys:
        raise StatsError("the two configurations share no functions")
    return [a[k] for k in keys], [b[k] for k in keys]


def paired_bootstrap(a: Dict[str, float], b: Dict[str, float], *,
                     resamples: int = 10000, alpha: float = 0.05,
                     seed: int = 0) -> Tuple[float, float, float]:
    """(mean difference, low, high) for b − a, resampling FUNCTIONS.

    The resampling unit is the function, not the observation, because the two
    configurations are evaluated on the same functions and the difference is
    what carries the signal.
    """
    xs, ys = _pairs(a, b)
    diffs = [y - x for x, y in zip(xs, ys)]
    n = len(diffs)
    point = sum(diffs) / n
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(int((1 - alpha / 2) * resamples), resamples - 1)]
    return point, lo, hi


def wilcoxon(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, str]:
    """Two-sided Wilcoxon signed-rank on the paired differences.

    scipy is used when present because its exact small-sample branch is worth
    having; the normal approximation with tie and continuity correction is
    implemented as a fallback so results do not depend on which packages
    happen to be installed on a given machine.
    """
    xs, ys = _pairs(a, b)
    diffs = [y - x for x, y in zip(xs, ys)]
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return 1.0, "wilcoxon (all differences zero)"
    try:
        from scipy.stats import wilcoxon as _w
        return float(_w(xs, ys, zero_method="wilcox").pvalue), "wilcoxon"
    except Exception:                                  # noqa: BLE001
        return _wilcoxon_normal(nonzero), "wilcoxon (normal approximation)"


def _wilcoxon_normal(diffs: Sequence[float]) -> float:
    n = len(diffs)
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for d, r in zip(diffs, ranks) if d > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return 1.0
    z = (abs(w_plus - mu) - 0.5) / sigma               # continuity correction
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def cliffs_delta(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Non-parametric effect size in [-1, 1]. Positive means b tends higher."""
    xs, ys = _pairs(a, b)
    gt = sum(1 for x in xs for y in ys if y > x)
    lt = sum(1 for x in xs for y in ys if y < x)
    return (gt - lt) / (len(xs) * len(ys))


def compare(a: Dict[str, float], b: Dict[str, float], *,
            resamples: int = 10000, seed: int = 0) -> Comparison:
    """b against a, paired by function."""
    xs, ys = _pairs(a, b)
    diff, lo, hi = paired_bootstrap(a, b, resamples=resamples, seed=seed)
    p, test = wilcoxon(a, b)
    return Comparison(n=len(xs), mean_a=sum(xs) / len(xs),
                      mean_b=sum(ys) / len(ys), diff=diff, ci_low=lo,
                      ci_high=hi, p_value=p, cliffs_delta=cliffs_delta(a, b),
                      test=test)


# --------------------------------------------------------------------------
# many comparisons
# --------------------------------------------------------------------------

def holm(p_values: Dict[str, float], alpha: float = 0.05
         ) -> Dict[str, Tuple[float, bool]]:
    """Holm–Bonferroni. Returns name -> (adjusted p, reject).

    Six ablations against the full system is six tests. Reporting six
    uncorrected p-values and calling the ones below 0.05 significant is how a
    null result becomes a finding.
    """
    if not p_values:
        return {}
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Tuple[float, bool]] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[name] = (adj, adj < alpha)
    # Holm is a step-down procedure: once a hypothesis is not rejected,
    # nothing after it is either.
    stop = False
    for name, _ in items:
        adj, rej = out[name]
        if stop:
            out[name] = (adj, False)
        elif not rej:
            stop = True
    return out


# --------------------------------------------------------------------------
# the seed axis
# --------------------------------------------------------------------------

@dataclass
class SeedSpread:
    mean: float
    sd: float
    seeds: int
    values: List[float]

    def dominates(self, difference: float) -> bool:
        """Is a reported difference larger than the run-to-run noise?"""
        return abs(difference) > self.sd


def describe(per_seed: Dict[int, Dict[str, float]]) -> SeedSpread:
    """Mean and standard deviation of a configuration's score across seeds."""
    means = [sum(v.values()) / len(v) for v in per_seed.values() if v]
    if not means:
        raise StatsError("no seeds to describe")
    m = sum(means) / len(means)
    var = (sum((x - m) ** 2 for x in means) / (len(means) - 1)
           if len(means) > 1 else 0.0)
    return SeedSpread(mean=m, sd=math.sqrt(var), seeds=len(means),
                      values=means)


def ablation_table(full: Dict[int, Dict[str, float]],
                   others: Dict[str, Dict[int, Dict[str, float]]], *,
                   seed: int = 0, resamples: int = 10000) -> dict:
    """Every configuration against the full system, corrected and described.

    Comparisons use one seed's per-function scores (pairing requires the same
    functions, and mixing seeds would pair a function with itself under a
    different sample). The seed spread is reported alongside so a difference
    inside the noise can be seen for what it is.
    """
    base = full.get(seed) or next(iter(full.values()))
    rows, ps = {}, {}
    for name, per_seed in others.items():
        theirs = per_seed.get(seed) or next(iter(per_seed.values()))
        c = compare(base, theirs, resamples=resamples, seed=seed)
        rows[name] = c
        ps[name] = c.p_value
    adjusted = holm(ps)
    spread = {name: describe(per_seed) for name, per_seed in others.items()}
    spread["full"] = describe(full)
    return {
        "comparisons": {n: {**c.to_dict(),
                            "p_adjusted": adjusted[n][0],
                            "reject": adjusted[n][1],
                            "within_noise": not spread[n].dominates(c.diff)}
                        for n, c in rows.items()},
        "seed_spread": {n: {"mean": s.mean, "sd": s.sd, "seeds": s.seeds}
                        for n, s in spread.items()},
    }
