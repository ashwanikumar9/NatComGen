"""The prompt-validation harness — M1, M2 and M5.

This module had no tests at all, which was the wrong way round: it is the
thing that decides whether the prompts are good enough to run, so a bug in it
would let a bad prompt through silently. A scripted model lets every branch be
exercised, including the ones that are supposed to fail the gate.
"""
import json

import pytest

from conftest import corpus_or_mini
from natspec_corpus.harness import (critic_calibration, parse_rates,
                                    verifier_stability)
from natspec_corpus.llm import Client, MockBackend


def _require_sigma() -> None:
    if not (corpus_or_mini() / "sigma" / "sigma.jsonl").exists():
        pytest.skip("M2 needs fact tables and no compiler is available")


GOOD = {
    "L7": {"purpose": "p", "purpose_ids": ["F1"], "caller": None,
           "caller_ids": [], "preconditions": [], "effects": [],
           "returns_meaning": [], "unknowns": []},
    "L1b": {"natspec": "/// @notice Adds two values.",
            "claims": [{"text": "p", "ids": ["F1"]}], "used_examples": False},
    "L2": {"defects": [], "missing_high_value_facts": []},
    "L3": {"natspec": "/// @notice Adds two values.", "changed": False,
           "removed": [], "claims": [{"text": "p", "ids": ["F1"]}]},
    "L8": {"verdicts": [{"claim": "p", "rows": ["F1"],
                         "verdict": "SUPPORTED"}], "gate": "PASS"},
}


def client(handler=None, **kw):
    return Client(MockBackend(handler or (lambda pid, req: GOOD[pid])), **kw)


# -- M1: parse rates -------------------------------------------------------

def test_a_clean_model_meets_the_gate():
    out = parse_rates(corpus_or_mini(), client(), n=3)
    assert out["sampled"] == 3
    assert out["hard_failures"] == []
    assert out["gate_m1_met"] is True
    assert set(out["per_prompt"]) == {"L7", "L1b", "L2", "L3", "L8"}
    assert all(v["first_attempt_parse_rate"] == 1.0
               for v in out["per_prompt"].values())


def test_a_prompt_that_needs_a_retry_fails_the_gate():
    """A prompt needing a retry a fifth of the time produces a distribution
    shaped partly by the repair message — which is not what is being
    measured."""
    seen = set()

    def handler(pid, req):
        key = (pid, req["messages"][-1]["content"][:80])
        if pid == "L2" and key not in seen:
            seen.add(key)
            return "Sure, here is the audit you asked for."
        return GOOD[pid]

    out = parse_rates(corpus_or_mini(), client(handler), n=3)
    assert out["per_prompt"]["L2"]["first_attempt_parse_rate"] < 1.0
    assert out["gate_m1_met"] is False


def test_a_prompt_that_never_parses_is_a_hard_failure():
    def handler(pid, req):
        return "no json ever" if pid == "L8" else GOOD[pid]

    out = parse_rates(corpus_or_mini(), client(handler, max_attempts=2), n=2)
    assert out["hard_failures"] and out["gate_m1_met"] is False
    assert "L8" in out["hard_failures"][0]["error"]


def test_the_sample_is_reproducible_for_a_seed():
    a = parse_rates(corpus_or_mini(), client(), n=3, seed=5)
    b = parse_rates(corpus_or_mini(), client(), n=3, seed=5)
    assert a["sampled"] == b["sampled"]


# -- M2: critic calibration ------------------------------------------------

def test_a_critic_that_always_fires_has_a_bad_false_positive_rate():
    """A critic that flags everything is worse than no critic, and the
    verified pairs are what shows it."""
    _require_sigma()

    def handler(pid, req):
        if pid == "L2":
            return {"defects": [{"quote": "x", "verdict": "UNSUPPORTED",
                                 "severity": "high", "why": "everything",
                                 "ids": []}],
                    "missing_high_value_facts": []}
        return GOOD[pid]

    out = critic_calibration(corpus_or_mini(), client(handler), n_partial=6,
                             n_verified=6)
    assert out["false_positive_rate"] == 1.0
    assert out["gate_m2_met"] is False


def test_a_critic_that_never_fires_has_zero_recall():
    _require_sigma()
    out = critic_calibration(corpus_or_mini(), client(), n_partial=6, n_verified=6)
    assert out["false_positive_rate"] == 0.0
    assert all(v == 0.0 for v in out["recall_by_defect"].values())
    assert out["gate_m2_met"] is False


def test_recall_is_reported_per_defect_family_not_pooled():
    """`param_missing` outnumbers the rest three to one, so a single overall
    number would be almost entirely a measure of that one family."""
    _require_sigma()

    def handler(pid, req):
        if pid == "L2":
            return {"defects": [{"quote": "x", "verdict": "UNSUPPORTED",
                                 "severity": "low", "why": "y", "ids": []}],
                    "missing_high_value_facts": []}
        return GOOD[pid]

    out = critic_calibration(corpus_or_mini(), client(handler), n_partial=12,
                             n_verified=2)
    assert len(out["recall_by_defect"]) >= 2
    assert all(0.0 <= v <= 1.0 for v in out["recall_by_defect"].values())


def test_a_low_severity_flag_is_not_counted_as_a_false_positive():
    _require_sigma()

    def handler(pid, req):
        if pid == "L2":
            return {"defects": [{"quote": "x", "verdict": "UNSUPPORTED",
                                 "severity": "low", "why": "y", "ids": []}],
                    "missing_high_value_facts": []}
        return GOOD[pid]

    out = critic_calibration(corpus_or_mini(), client(handler), n_partial=2,
                             n_verified=6)
    assert out["false_positive_rate"] == 0.0


def test_the_calibration_keeps_a_sample_of_rows_for_inspection():
    _require_sigma()
    out = critic_calibration(corpus_or_mini(), client(), n_partial=4, n_verified=2)
    assert out["rows"] and set(out["rows"][0]) >= {"pair_id", "labelled",
                                                   "fired", "reported"}


# -- M5: verifier stability ------------------------------------------------

def test_a_stable_verifier_passes():
    out = verifier_stability(corpus_or_mini(), client(), n=4)
    if out["claims_compared"]:
        assert out["agreement"] == 1.0 and out["gate_met"] is True
        assert out["flips"] == []


def test_a_verifier_that_reads_position_is_caught():
    """If the verdict depends on where a claim sits in the list, the verifier
    is reading order rather than evidence — the documented failure mode for
    LLM judges."""
    def handler(pid, req):
        if pid != "L8":
            return GOOD[pid]
        claims = json.loads(req["messages"][-1]["content"].split(
            "[Claims asserted]")[1].strip().split("\n\nVerify")[0])
        return {"verdicts": [
            {"claim": c["text"], "rows": ["F1"],
             "verdict": "SUPPORTED" if i == 0 else "UNSUPPORTED"}
            for i, c in enumerate(claims)], "gate": "PASS"}

    out = verifier_stability(corpus_or_mini(), client(handler), n=4)
    if out["claims_compared"]:
        assert out["agreement"] < 1.0
        assert out["gate_met"] is False
        assert out["flips"]
