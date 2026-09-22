"""The metric. If this is not their metric, nothing else in the benchmark
means anything.
"""
import json
from pathlib import Path

import pytest

from benchmarks.smartdoc import bleu

DATA = Path(__file__).resolve().parents[1] / "data/smartdoc"
PUBLISHED_BA = {"smartdoc": 47.39, "re2com": 29.37,
                "attendgru": 29.01, "ast-attendgru": 26.01}


def _lines(name):
    p = DATA / name
    if not p.exists():
        pytest.skip(f"{p} absent — run: python -m benchmarks.smartdoc.fetch")
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


@pytest.mark.parametrize("system,expected", sorted(PUBLISHED_BA.items()))
def test_reproduces_the_published_scores(system, expected):
    """The paper reports 47.39 for SmartDoc. So must we, from their output."""
    assert bleu.leclair(_lines("ref.txt"), _lines(f"{system}.out"))["Ba"] == \
        expected


def test_agrees_with_nltk_where_nltk_is_installed():
    msg = bleu.check_against_nltk(_lines("ref.txt"), _lines("smartdoc.out"))
    assert "MISMATCH" not in msg, msg


def test_short_hypothesis_still_charges_a_denominator():
    """nltk's `max(1, ...)`, and the reason our first attempt read 47.50.

    A two-word hypothesis has no trigram and no 4-gram. Skipping it entirely
    would raise p3 and p4 — a different metric wearing the same name.
    """
    refs = [[["a", "b", "c", "d"]], [["x", "y"]]]
    hyps = [["a", "b", "c", "d"], ["x", "y"]]
    with_short = bleu.corpus_bleu(refs, hyps)
    alone = bleu.corpus_bleu(refs[:1], hyps[:1])
    assert with_short < alone


def test_identical_corpus_scores_one():
    refs = [[["the", "owner", "may", "withdraw"]]]
    assert bleu.corpus_bleu(refs, [refs[0][0]]) == pytest.approx(1.0)


def test_no_overlap_scores_zero():
    refs = [[["a", "b", "c", "d"]]]
    assert bleu.corpus_bleu(refs, [["w", "x", "y", "z"]]) == 0.0


def test_length_mismatch_is_an_error_not_a_truncation():
    """Their own `evaluate.py` zips, which silently drops the tail.

    A reference file one line short would shift nothing and lose one row, and
    the resulting score would look entirely plausible.
    """
    with pytest.raises(ValueError):
        bleu.corpus_bleu([[["a"]], [["b"]]], [["a"]])


def test_brevity_penalty_bites():
    refs = [[["a", "b", "c", "d", "e", "f", "g", "h"]]]
    full = bleu.corpus_bleu(refs, [refs[0][0]], weights=(1, 0, 0, 0))
    short = bleu.corpus_bleu(refs, [["a", "b"]], weights=(1, 0, 0, 0))
    assert short < full == pytest.approx(1.0)


def test_leakage_split_is_measured_not_assumed():
    """434 of the 1,000 references appear verbatim in their training set.

    This is the decomposition of the headline number, so it is pinned: if a
    future change to the tokenisation or the file loading alters it, the
    claim in the README stops being true and this fails.
    """
    from benchmarks.smartdoc.score import seen_in_training
    if not (DATA / "dataset/train/train.token.nl").exists():
        pytest.skip("SmartDoc dataset absent")
    seen = seen_in_training(DATA)
    assert len(seen) == 434

    refs, out = _lines("ref.txt"), _lines("smartdoc.out")
    unseen = [i for i in range(len(refs)) if i not in seen]
    memorised = bleu.leclair([refs[i] for i in seen], [out[i] for i in seen])
    general = bleu.leclair([refs[i] for i in unseen], [out[i] for i in unseen])
    assert memorised["Ba"] == 81.68
    assert general["Ba"] == 17.53
    # The published 47.39 sits between them, much nearer the memorised half.
    assert general["Ba"] < 47.39 < memorised["Ba"]
