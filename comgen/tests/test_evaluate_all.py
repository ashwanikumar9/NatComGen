"""tools/evaluate_all.py — the metrics, on inputs whose answers are known.

METEOR and ROUGE are easy to implement subtly wrong and impossible to eyeball,
so each property is pinned against a case where the right answer is derivable
by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import evaluate_all as E                             # noqa: E402


# --------------------------------------------------------------------------
# ROUGE-n
# --------------------------------------------------------------------------

def test_rouge_1_is_one_for_identical_text():
    assert E.rouge_n("the amount to add", "the amount to add", 1) == 1.0


def test_rouge_1_is_zero_for_no_overlap():
    assert E.rouge_n("alpha beta", "gamma delta", 1) == 0.0


def test_rouge_1_is_the_f1_of_unigram_overlap():
    # 2 of 3 candidate unigrams overlap, 2 of 2 reference: P=2/3, R=1, F=0.8
    assert E.rouge_n("the amount extra", "the amount", 1) == pytest.approx(0.8)


def test_rouge_2_needs_adjacency_not_just_presence():
    assert E.rouge_n("amount the", "the amount", 1) == 1.0, "unigrams match"
    assert E.rouge_n("amount the", "the amount", 2) == 0.0, "the bigram does not"


def test_rouge_n_is_zero_when_the_text_is_shorter_than_n():
    assert E.rouge_n("word", "word", 2) == 0.0


def test_repeated_candidate_ngrams_are_clipped():
    """`the the the` against `the` must not score as three matches."""
    assert E.rouge_n("the the the", "the", 1) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# the stemmer
# --------------------------------------------------------------------------

def test_the_stemmer_strips_common_suffixes():
    assert E.stem("deposits") == E.stem("deposit") == "deposit"
    assert E.stem("returned") == "return"
    assert E.stem("quickly") == "quick"
    assert E.stem("Updating") == "updat", "case-folded, longest suffix first"
    assert E.stem("payments") == "pay", "-ments before -s"


def test_the_stemmer_refuses_to_shorten_a_word_below_three_letters():
    """An aggressive stemmer invents matches, which inflates METEOR in the one
    direction nobody would notice."""
    assert E.stem("is") == "is"
    assert E.stem("as") == "as"
    assert E.stem("les") == "les"


# --------------------------------------------------------------------------
# METEOR
# --------------------------------------------------------------------------

def test_meteor_of_identical_text_is_near_one_not_exactly_one():
    """METEOR charges a fragmentation penalty even on a perfect match: one
    chunk over m matches. This is the real metric's behaviour, not a bug."""
    s = E.meteor("the amount of deposit to add", "the amount of deposit to add")
    assert 0.98 < s < 1.0


def test_meteor_is_zero_when_nothing_aligns():
    assert E.meteor("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_meteor_is_zero_for_an_empty_prediction():
    assert E.meteor("", "the amount to add") == 0.0
    assert E.meteor("the amount to add", "") == 0.0


def test_scrambling_the_same_words_costs_meteor_but_not_rouge_1():
    """The fragmentation penalty is the whole reason METEOR is worth reporting
    beside ROUGE-1: same words, wrong order, same ROUGE-1, lower METEOR."""
    ref = "records the deposit and mints a position token"
    scrambled = "token position a mints and deposit the records"
    assert E.rouge_n(scrambled, ref, 1) == 1.0
    assert E.meteor(scrambled, ref) < E.meteor(ref, ref) * 0.75


def test_a_stem_match_counts_but_an_exact_match_is_preferred():
    """Aligning `deposits` to `deposit` on the stem pass, when an exact
    `deposits` was available further along the reference, both loses a true
    match and invents a chunk break."""
    assert E.meteor("deposits", "deposit") > 0.0
    both = E.meteor("the deposits", "the deposit deposits")
    assert both > 0.0


def test_chunks_counts_contiguous_runs_on_both_sides():
    assert E._chunks([]) == 0
    assert E._chunks([(0, 0), (1, 1), (2, 2)]) == 1
    assert E._chunks([(0, 0), (1, 5)]) == 2, "reference side jumped"
    assert E._chunks([(0, 0), (2, 1)]) == 2, "hypothesis side jumped"


def test_meteor_rewards_recall_more_than_precision():
    """alpha=0.9 in Fmean = PR/(aP + (1-a)R) weights recall nine to one, which
    is METEOR's defining choice: omitting the reference's content is punished
    harder than adding to it."""
    ref = "the amount of stablecoin to deposit now"
    short = "the amount"                        # high precision, low recall
    long = "the amount of stablecoin to deposit now and some extra words here"
    assert E.meteor(long, ref) > E.meteor(short, ref)


# --------------------------------------------------------------------------
# the aggregate
# --------------------------------------------------------------------------

def test_evaluate_reports_every_metric_and_the_count():
    refs = ["the amount to add to the total", "the identifier of the pool"]
    hyps = ["the amount to add to the total", "the pool identifier"]
    s = E.evaluate(refs, hyps)
    for key in ("BLEU", "B1", "B2", "B3", "B4", "ROUGE-1", "ROUGE-2",
                "ROUGE-L", "METEOR"):
        assert key in s, key
        assert 0.0 <= s[key] <= 100.0
    assert s["n"] == 2


def test_a_perfect_prediction_scores_a_hundred_bleu():
    refs = ["the amount of stablecoin to deposit in the pool"]
    s = E.evaluate(refs, list(refs))
    assert s["BLEU"] == pytest.approx(100.0, abs=0.01)
    assert s["ROUGE-1"] == pytest.approx(100.0, abs=0.01)


def test_an_empty_prediction_is_counted_not_skipped():
    s = E.evaluate(["the amount to add"], [""])
    assert s["empty_hypotheses"] == 1
    assert s["METEOR"] == 0.0


# --------------------------------------------------------------------------
# BLEU-1 and BLEU-2 beside BLEU-4
#
# On the first real comparison, BLEU-4 read 0.00 for four of sixteen
# configurations. Those had not produced nothing — they had produced no 4-gram
# appearing in any reference, and one zero factor zeroes a geometric mean. On
# short NatSpec comments over a few dozen functions that is routine.
# --------------------------------------------------------------------------

def test_bleu_4_is_zero_when_no_four_gram_matches_but_bleu_1_is_not():
    refs = ["the amount of stablecoin to deposit in the pool"]
    hyps = ["amount stablecoin pool deposit the of to in"]      # same words
    s = E.evaluate(refs, hyps)
    assert s["BLEU-4"] == 0.0
    assert s["BLEU-1"] > 50.0, "every unigram matches"
    assert s["BLEU-2"] < s["BLEU-1"]


def test_all_three_bleus_agree_on_a_perfect_prediction():
    refs = ["the amount of stablecoin to deposit in the pool"]
    s = E.evaluate(refs, list(refs))
    assert s["BLEU-1"] == pytest.approx(100.0, abs=0.01)
    assert s["BLEU-2"] == pytest.approx(100.0, abs=0.01)
    assert s["BLEU-4"] == pytest.approx(100.0, abs=0.01)


def test_bleu_4_still_equals_the_geometric_mean_of_its_parts():
    """A property worth pinning: the reported BLEU-4 is the geometric mean of
    B1..B4, so a reader can reconstruct it from the parts."""
    refs = ["the amount of stablecoin to deposit in the pool now"]
    hyps = ["the amount of stablecoin to deposit in the vault now"]
    s = E.evaluate(refs, hyps)
    geo = (s["B1"] * s["B2"] * s["B3"] * s["B4"]) ** 0.25
    assert s["BLEU-4"] == pytest.approx(geo, rel=0.01)
