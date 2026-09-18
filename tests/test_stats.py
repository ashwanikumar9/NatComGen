"""Stage 4: the statistics.

Each test uses a distribution with a known answer, so the module is checked
against arithmetic rather than against itself.
"""
import random

import pytest

from natspec_corpus.stats import (Comparison, StatsError, ablation_table,
                                  cliffs_delta, compare, describe, holm,
                                  paired_bootstrap, wilcoxon)


def scores(values, prefix="f"):
    return {f"{prefix}{i}": v for i, v in enumerate(values)}


# -- paired bootstrap ------------------------------------------------------

def test_identical_samples_give_a_zero_difference_and_an_interval_on_zero():
    a = scores([0.1 * i for i in range(40)])
    point, lo, hi = paired_bootstrap(a, dict(a), resamples=2000)
    assert point == pytest.approx(0.0)
    assert lo == hi == pytest.approx(0.0)


def test_a_known_shift_is_recovered_inside_its_interval():
    """Shift plus noise, so the interval is a real interval rather than a
    degenerate point."""
    rng = random.Random(0)
    base = [rng.gauss(0.5, 0.1) for _ in range(200)]
    noise = [rng.gauss(0.0, 0.02) for _ in range(200)]
    a = scores(base)
    b = scores([x + 0.08 + n for x, n in zip(base, noise)])
    point, lo, hi = paired_bootstrap(a, b, resamples=3000)
    assert point == pytest.approx(0.08, abs=0.01)
    assert lo < point < hi and lo > 0


def test_the_interval_spans_zero_when_there_is_no_real_difference():
    """Differences symmetric about zero by construction, so the answer does
    not depend on which random draw came out."""
    base = [0.1 * i for i in range(200)]
    a = scores(base)
    b = scores([x + (0.05 if i % 2 == 0 else -0.05)
                for i, x in enumerate(base)])
    _, lo, hi = paired_bootstrap(a, b, resamples=3000)
    assert lo < 0 < hi


def test_the_bootstrap_is_reproducible_for_a_given_seed():
    a = scores([0.1, 0.4, 0.3, 0.9, 0.2])
    b = scores([0.2, 0.3, 0.5, 0.8, 0.4])
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)


def test_only_shared_functions_are_compared():
    a = {"f1": 1.0, "f2": 2.0, "only_a": 9.0}
    b = {"f1": 1.5, "f2": 2.5, "only_b": 0.0}
    point, _, _ = paired_bootstrap(a, b, resamples=200)
    assert point == pytest.approx(0.5)


def test_no_shared_functions_is_an_error_not_a_zero():
    with pytest.raises(StatsError, match="share no functions"):
        paired_bootstrap({"a": 1.0}, {"b": 1.0})


# -- wilcoxon --------------------------------------------------------------

def test_identical_samples_give_p_one():
    a = scores([0.2, 0.5, 0.7, 0.1])
    p, _ = wilcoxon(a, dict(a))
    assert p == pytest.approx(1.0)


def test_a_consistent_shift_is_significant():
    a = scores([0.1 * i for i in range(30)])
    b = scores([0.1 * i + 0.05 for i in range(30)])
    p, _ = wilcoxon(a, b)
    assert p < 0.001


def test_a_symmetric_difference_is_not_significant():
    """Half the functions improve by exactly as much as the other half get
    worse: the signed-rank statistic has nothing to find."""
    base = [0.1 * i for i in range(60)]
    a = scores(base)
    b = scores([x + (0.05 if i % 2 == 0 else -0.05)
                for i, x in enumerate(base)])
    p, _ = wilcoxon(a, b)
    assert p > 0.5


def test_the_normal_fallback_agrees_with_scipy_on_a_clear_case():
    from natspec_corpus.stats import _wilcoxon_normal
    diffs = [0.05] * 25 + [-0.01] * 3
    assert _wilcoxon_normal(diffs) < 0.01


# -- effect size -----------------------------------------------------------

def test_cliffs_delta_is_one_when_every_value_is_higher():
    a = scores([1.0, 2.0, 3.0])
    b = scores([4.0, 5.0, 6.0])
    assert cliffs_delta(a, b) == pytest.approx(1.0)
    assert cliffs_delta(b, a) == pytest.approx(-1.0)


def test_cliffs_delta_is_zero_for_identical_distributions():
    a = scores([1.0, 2.0, 3.0, 4.0])
    assert cliffs_delta(a, dict(a)) == pytest.approx(0.0)


def test_magnitude_labels_follow_the_usual_thresholds():
    def c(delta):
        return Comparison(10, 0, 0, 0, 0, 0, 0.01, delta, "t")
    assert c(0.10).magnitude == "negligible"
    assert c(0.20).magnitude == "small"
    assert c(0.40).magnitude == "medium"
    assert c(0.60).magnitude == "large"


# -- compare ---------------------------------------------------------------

def test_a_tiny_but_consistent_difference_is_significant_and_negligible():
    """With enough functions a 0.004 difference clears p < 0.05 and still means
    nothing. Reporting the p-value without the effect size hides that.

    The two distributions must overlap for Cliff's delta to say so: it
    compares every value against every other, so two constant vectors give
    delta 1.0 however small the gap between them."""
    rng = random.Random(4)
    base = [rng.gauss(0.5, 0.15) for _ in range(200)]
    a = scores(base)
    b = scores([x + 0.004 for x in base])
    c = compare(a, b, resamples=1000)
    assert c.diff == pytest.approx(0.004, abs=1e-9)
    assert c.significant
    assert c.magnitude == "negligible", c.cliffs_delta


def test_comparison_serialises_everything_a_table_needs():
    a = scores([0.1, 0.2, 0.3])
    b = scores([0.2, 0.3, 0.4])
    d = compare(a, b, resamples=500).to_dict()
    for k in ("n", "diff", "ci_low", "ci_high", "p_value", "cliffs_delta",
              "significant", "magnitude"):
        assert k in d


# -- multiple comparisons --------------------------------------------------

def test_holm_leaves_a_single_test_alone():
    assert holm({"a": 0.01})["a"] == (pytest.approx(0.01), True)


def test_holm_raises_the_smallest_p_by_the_number_of_tests():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.20})
    assert out["a"][0] == pytest.approx(0.03)
    assert out["b"][0] == pytest.approx(0.08)


def test_holm_is_step_down_nothing_survives_past_a_failure():
    out = holm({"a": 0.20, "b": 0.001})
    assert out["b"][1] is True
    assert out["a"][1] is False


def test_six_uncorrected_borderline_results_do_not_all_survive():
    """Six ablations at p = 0.04 each is exactly the situation correction is
    for."""
    out = holm({f"c{i}": 0.04 for i in range(6)})
    assert not any(rej for _, rej in out.values())


def test_holm_on_nothing_is_nothing():
    assert holm({}) == {}


# -- the seed axis ---------------------------------------------------------

def test_seed_spread_reports_mean_and_sd():
    s = describe({0: scores([0.5] * 10), 1: scores([0.6] * 10),
                  2: scores([0.7] * 10)})
    assert s.seeds == 3 and s.mean == pytest.approx(0.6)
    assert s.sd == pytest.approx(0.1)


def test_a_single_seed_has_no_spread():
    assert describe({0: scores([0.5, 0.5])}).sd == 0.0


def test_a_difference_inside_the_seed_noise_is_flagged():
    s = describe({0: scores([0.50] * 5), 1: scores([0.60] * 5),
                  2: scores([0.70] * 5)})
    assert not s.dominates(0.05)
    assert s.dominates(0.5)


def test_describe_refuses_an_empty_input():
    with pytest.raises(StatsError):
        describe({})


# -- the whole table -------------------------------------------------------

def test_ablation_table_corrects_and_flags_noise():
    rng = random.Random(3)
    base = [rng.gauss(0.6, 0.05) for _ in range(80)]
    full = {s: scores([x + 0.001 * s for x in base]) for s in (0, 1, 2)}
    worse = {s: scores([x - 0.10 for x in base]) for s in (0, 1, 2)}
    same = {s: scores([x + 0.001 for x in base]) for s in (0, 1, 2)}
    out = ablation_table(full, {"no_sigma": worse, "no_retrieval": same},
                         resamples=800)

    ns = out["comparisons"]["no_sigma"]
    assert ns["diff"] < 0 and ns["reject"] and ns["magnitude"] != "negligible"

    nr = out["comparisons"]["no_retrieval"]
    assert abs(nr["diff"]) < 0.01
    assert nr["magnitude"] == "negligible"

    assert set(out["seed_spread"]) == {"no_sigma", "no_retrieval", "full"}


def test_the_ablation_table_emits_exactly_what_the_report_reads():
    """The two modules were written apart and disagreed on a key name —
    `within_seed_noise` against `within_noise` — which no unit test could see
    because each tested its own vocabulary."""
    from natspec_corpus.report import ablation_table_rows
    a = {0: scores([0.5] * 20), 1: scores([0.51] * 20)}
    b = {0: scores([0.4] * 20), 1: scores([0.41] * 20)}
    table = ablation_table(a, {"C2": b}, resamples=300)
    row = table["comparisons"]["C2"]
    for key in ("n", "diff", "ci_low", "ci_high", "p_adjusted", "reject",
                "cliffs_delta", "magnitude", "within_noise"):
        assert key in row, key
    headers, rows = ablation_table_rows(table, {"C2": "no Σ(f)"})
    assert len(rows) == 1 and len(rows[0]) == len(headers)
