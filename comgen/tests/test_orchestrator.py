"""The loop: when it stops, which round wins, and what the record carries.

These are the decisions that are not visible in any prompt and cannot be read
off a run's output, so they are pinned here. Each test is named after the rule
it protects.
"""
from __future__ import annotations

import pytest

from natspec_corpus.errors import CorpusError

from comgen import orchestrator as O
from comgen.tests import fixtures as F


def run(script: F.Script, **kw) -> dict:
    return O.generate(F.ctx(), F.client(script), None, **kw)


# --------------------------------------------------------------------------
# stopping
# --------------------------------------------------------------------------

def test_a_clean_draft_ends_the_loop_immediately():
    """The deterministic critic is the exit guard, and it is free. A clean
    draft must cost no revision call at all."""
    s = F.default_script()
    rec = run(s)
    assert rec["rounds_used"] == 1
    assert rec["best_round"] == 0
    assert s.count("R1") == 0
    assert rec["gate_kept_draft"] is True


def test_a_draft_that_never_comes_clean_uses_its_whole_budget():
    s = F.default_script(drafts=(F.DIRTY,), revisions=(F.DIRTY,))
    rec = run(s, rounds=3)
    assert rec["rounds_used"] == 3
    assert s.count("R1") == 2, "three rounds of criticism, two revisions"
    assert s.count("L2") == 3


def test_the_loop_stops_the_round_it_comes_clean():
    s = F.default_script(drafts=(F.DIRTY,), revisions=(F.CLEAN,))
    rec = run(s, rounds=3)
    assert rec["rounds_used"] == 2
    assert s.count("R1") == 1
    assert rec["best_round"] == 1
    assert rec["gate_kept_draft"] is False


def test_a_semantic_defect_alone_keeps_the_loop_going():
    """Structural cleanliness is not sufficient. A comment solc is happy with
    can still assert something the evidence does not support, and that is the
    defect class this whole architecture exists for."""
    s = F.default_script(
        drafts=(F.CLEAN,), revisions=(F.CLEAN,),
        critiques=[F.critique(F.defect("only the owner may call",
                                       severity="high")),
                   F.critique()])
    rec = run(s, rounds=3)
    assert rec["rounds_used"] == 2
    assert s.count("R1") == 1


def test_rounds_of_one_is_v2s_single_pass():
    s = F.default_script(drafts=(F.DIRTY,))
    rec = run(s, rounds=1)
    assert rec["rounds_used"] == 1
    assert s.count("R1") == 0, "no budget for a revision"


def test_switching_off_revision_stops_after_the_first_critique():
    s = F.default_script(drafts=(F.DIRTY,))
    rec = run(s, rounds=3, stages=["I7", "G", "CD", "CS", "J"])
    assert rec["rounds_used"] == 1
    assert s.count("R1") == 0


def test_a_reviser_that_returns_nothing_does_not_poison_the_next_round():
    s = F.Script(L9=[F.contract_reply()], L7=[F.intent_reply()],
                 L1b=[F.draft(F.DIRTY)], L2=[F.critique()],
                 R1=[F.revision("")], L8=[F.judgement()])
    rec = run(s, rounds=3)
    assert rec["rounds_used"] == 1
    assert rec["rounds"][0]["reviser_returned_nothing"] is True
    assert rec["final"] == F.DIRTY, "the draft is kept, not an empty comment"


# --------------------------------------------------------------------------
# which round wins
# --------------------------------------------------------------------------

def test_the_best_round_wins_not_the_last():
    """A revision that answers one complaint by introducing two is not an
    improvement. Taking the last round on faith is exactly how v2's refiner
    made things worse about as often as better."""
    s = F.default_script(drafts=(F.ONE_DEFECT_A,), revisions=(F.DIRTY,))
    rec = run(s, rounds=2)
    assert rec["rounds_used"] == 2
    assert rec["best_round"] == 0
    assert rec["final"] == F.ONE_DEFECT_A
    assert rec["gate_kept_draft"] is True


def test_a_tie_on_defect_count_goes_to_the_later_round():
    """Equal counts are not equally good bets: the later candidate was written
    in answer to specific criticism of the earlier one, and its calls are spent
    either way. This is the v2 gate's `<=` rule, kept deliberately so the
    comparison against C1 is not confounded by a changed tie-break."""
    s = F.default_script(drafts=(F.ONE_DEFECT_A,), revisions=(F.ONE_DEFECT_B,))
    rec = run(s, rounds=2)
    assert rec["rounds"][0]["total"] == rec["rounds"][1]["total"] == 1
    assert rec["best_round"] == 1
    assert rec["final"] == F.ONE_DEFECT_B


def test_blocking_defects_outrank_total_defects():
    """A round with one structural defect and nothing else loses to a round
    with no structural defect and two semantic ones."""
    attempts = [{"round": 0, "blocking": 1, "total": 1},
                {"round": 1, "blocking": 0, "total": 2}]
    assert O.select(attempts)["round"] == 1


def test_select_refuses_an_empty_history():
    with pytest.raises(CorpusError):
        O.select([])


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------

def test_the_record_carries_every_round():
    s = F.default_script(drafts=(F.DIRTY,), revisions=(F.DIRTY,))
    rec = run(s, rounds=3)
    assert [r["round"] for r in rec["rounds"]] == [0, 1, 2]
    assert all(r["natspec"] for r in rec["rounds"])
    assert all("deterministic" in r and "semantic" in r for r in rec["rounds"])


def test_gate_defects_come_from_the_winning_round():
    """`gate.refined_defects` is what fills the defect-free column of the
    existing report, so it must describe the comment that was actually kept."""
    s = F.default_script(drafts=(F.DIRTY,), revisions=(F.CLEAN,))
    rec = run(s, rounds=3)
    assert rec["gate"]["refined_defects"] == []
    assert len(rec["gate"]["draft_defects"]) == 4
    assert rec["gate"]["passed"] is True


def test_calls_are_counted_because_cost_is_now_variable():
    s = F.default_script(drafts=(F.DIRTY,), revisions=(F.DIRTY,))
    rec = run(s, rounds=3)
    assert rec["calls"]["L1b"] == 1
    assert rec["calls"]["L2"] == 3
    assert rec["calls"]["R1"] == 2
    assert rec["calls"]["L9"] == 1
    assert rec["calls"]["total"] == 1 + 1 + 1 + 3 + 2 + 1   # L9 L7 L1b L2 R1 L8


def test_a_clean_function_costs_five_calls_then_four():
    """The floor of the cost distribution. L9, L7, L1b, L2, L8 on the first
    function of a file; the same minus L9 on every function after it, because
    the contract reading is memoised per file."""
    s = F.default_script()
    client = F.client(s)
    from comgen import intent as I
    memo = I.ContractIntent(client)
    first = O.generate(F.ctx(), client, None, contract=memo)
    second = O.generate(F.ctx(), client, None, contract=memo)
    assert first["calls"]["total"] == 5
    assert second["calls"]["total"] == 4


def test_the_judge_strips_a_contradicted_claim():
    s = F.default_script()
    s.q["L8"] = [F.judgement(verdict="CONTRADICTED", gate="FAIL")]
    rec = run(s)
    assert rec["gate_verdict"] == "FAIL"
    assert rec["support_rate"] == 0.0
    assert rec["stripped"] == ["Adds two unsigned integers."]


def test_the_generator_is_required():
    with pytest.raises(CorpusError):
        run(F.default_script(), stages=["I7", "CD", "CS", "J"])


def test_rounds_must_be_at_least_one():
    with pytest.raises(CorpusError):
        run(F.default_script(), rounds=0)


def test_contract_intent_is_asked_once_per_file_not_once_per_function():
    """The reason this agent is affordable at all. Two functions in one file
    must cost one L9 call between them."""
    from comgen import intent as I
    s = F.default_script()
    client = F.client(s)
    memo = I.ContractIntent(client)
    for _ in range(3):
        O.generate(F.ctx(), client, None, contract=memo)
    assert s.count("L9") == 1
    assert memo.stats() == {"files": 1, "reused": 2, "calls": 1}


def test_switching_off_contract_intent_issues_no_call():
    s = F.default_script()
    rec = run(s, stages=["I7", "G", "CD", "CS", "R", "J"])
    assert s.count("L9") == 0
    assert "contract_intent" not in rec


def test_the_contract_reading_reaches_the_generator():
    s = F.default_script()
    client = F.client(s)
    O.generate(F.ctx(), client, None)
    generator_request = [c for c in client.backend.calls
                         if c["_prompt_id"] == "L1b"][0]
    sent = generator_request["messages"][-1]["content"]
    assert "DO NOT CLAIM: no oracle" in sent
    assert "[Contract]" in sent and "[Function]" in sent
