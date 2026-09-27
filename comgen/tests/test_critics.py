"""The Critic Group: what each critic reports, and how the merge resolves them.

The merge is the one place in ComGen where two components can genuinely
contradict each other, so it is tested at the level of the rule rather than the
output: a deterministic finding survives, a semantic finding that would undo one
does not, and what is left is ordered worst-first.
"""
from __future__ import annotations

import pytest

from natspec_corpus.gate import judge_text

from comgen import critics as C
from comgen.tests import fixtures as F


# --------------------------------------------------------------------------
# the deterministic critic
# --------------------------------------------------------------------------

def test_clean_comment_has_no_deterministic_findings():
    v = C.deterministic(F.ctx(), F.CLEAN)
    assert v.findings == []
    assert v.ok


def test_dirty_comment_findings_match_the_corpus_scorer():
    """The critic's notion of a defect must be the corpus's notion of one.

    If these drift apart, ComGen is optimising against a different definition
    of "defective" than the one its training references were labelled with, and
    every comparison against C1 is measuring two things at once.
    """
    v = C.deterministic(F.ctx(), F.DIRTY)
    assert [f.text for f in v.findings] == judge_text(F.pair(), F.DIRTY)
    assert all(f.severity == "blocking" for f in v.findings)
    assert not v.ok


def test_every_deterministic_finding_carries_a_reason():
    v = C.deterministic(F.ctx(), F.DIRTY)
    assert all(f.why for f in v.findings), \
        "a MUST FIX item with no explanation is not actionable"


def test_no_compilation_unit_is_not_a_pass():
    """The most dangerous failure mode in the old gate, kept out by a test.

    With no compilation unit the compiler half cannot run. Reporting that as a
    pass would let a comment solc silently drops sail through as correct.
    """
    v = C.deterministic(F.ctx(), F.CLEAN)
    assert v.applicable is False
    assert v.detail["compiles"] is None
    assert v.detail["tags_emitted"] is None
    assert "no compilation unit" in v.detail["note"]


def test_a_critic_that_did_not_run_is_not_clean():
    v = C.not_run(C.SEMANTIC)
    assert not v.ok, "unknown must never read as clean"
    assert v.ran is False
    assert v.applicable is False


# --------------------------------------------------------------------------
# the semantic critic
# --------------------------------------------------------------------------

def test_semantic_findings_keep_verdict_and_severity():
    script = F.Script(L2=[F.critique(
        F.defect("only the owner may call", "UNSUPPORTED", "high"),
        F.defect("returns the fee in wei", "CONTRADICTED", "medium"))])
    v, call = C.semantic(F.ctx(), F.CLEAN, F.client(script))
    assert [f.severity for f in v.findings] == ["high", "medium"]
    assert v.findings[0].text.startswith("UNSUPPORTED")
    assert v.findings[1].text.startswith("CONTRADICTED")
    assert call.prompt_id == "L2"


def test_semantic_critic_reads_the_draft_not_a_revision():
    """Both critics see the generator's output directly. This asserts the
    candidate reaches the prompt, which is the whole structural change from
    v2 — there the deterministic check only ever saw the refiner's output."""
    script = F.Script(L2=[F.critique()])
    client = F.client(script)
    C.semantic(F.ctx(), F.DIRTY, client)
    sent = client.backend.calls[-1]["messages"][-1]["content"]
    assert F.DIRTY in sent


# --------------------------------------------------------------------------
# the merge
# --------------------------------------------------------------------------

def test_deterministic_findings_all_become_must_fix():
    det = C.deterministic(F.ctx(), F.DIRTY)
    g = C.merge(det, C.not_run(C.SEMANTIC))
    assert len(g.must_fix) == len(det.findings)
    assert g.should_fix == []
    assert not g.clean


def test_semantic_finding_that_would_undo_a_required_tag_is_suppressed():
    """The precedence rule, stated as a test.

    The deterministic critic says solc found no description for @param b. The
    semantic critic says the comment should not describe b at all. Following
    the semantic critic produces a comment that satisfies it and documents less
    than the declaration requires — so it is dropped, and the drop is recorded
    rather than silent.
    """
    det = C.deterministic(F.ctx(), F.TWO_DEFECTS)     # b and the return
    sem = C.CriticVerdict(name=C.SEMANTIC, ran=True, findings=[
        C.Finding(source=C.SEMANTIC, text="UNSUPPORTED: @param b the second "
                                          "addend", severity="high",
                  subject="b"),
        C.Finding(source=C.SEMANTIC, text="UNSUPPORTED: only the owner may "
                                          "call", severity="high")])
    g = C.merge(det, sem)
    kept = [f.text for f in g.should_fix]
    assert "UNSUPPORTED: only the owner may call" in kept
    assert not any("@param b" in t for t in kept)
    assert len(g.suppressed) == 1
    assert "deterministic critic requires" in g.suppressed[0]["dropped_because"]


def test_an_unrelated_semantic_finding_survives_the_merge():
    det = C.deterministic(F.ctx(), F.CLEAN)
    sem = C.CriticVerdict(name=C.SEMANTIC, ran=True, findings=[
        C.Finding(source=C.SEMANTIC, text="UNSUPPORTED: reverts on overflow",
                  severity="high")])
    g = C.merge(det, sem)
    assert g.must_fix == []
    assert len(g.should_fix) == 1
    assert not g.clean


def test_should_fix_is_ordered_worst_first():
    sem = C.CriticVerdict(name=C.SEMANTIC, ran=True, findings=[
        C.Finding(source=C.SEMANTIC, text="UNSUPPORTED: c", severity="low"),
        C.Finding(source=C.SEMANTIC, text="UNSUPPORTED: b", severity="high"),
        C.Finding(source=C.SEMANTIC, text="CONTRADICTED: a",
                  severity="medium")])
    g = C.merge(C.deterministic(F.ctx(), F.CLEAN), sem)
    assert [f.text[-1] for f in g.should_fix] == ["a", "b", "c"], \
        "contradicted first, then by severity"


def test_clean_group_is_clean_and_omissions_do_not_block():
    sem = C.CriticVerdict(
        name=C.SEMANTIC, ran=True, findings=[],
        detail={"missing_high_value_facts": [{"text": "reverts on zero",
                                             "ids": ["R1"]}]})
    g = C.merge(C.deterministic(F.ctx(), F.CLEAN), sem)
    assert g.clean, "an omission is a suggestion, not a defect"
    assert g.omissions


def test_feedback_puts_must_fix_first_and_labels_it_non_negotiable():
    det = C.deterministic(F.ctx(), F.DIRTY)
    sem = C.CriticVerdict(name=C.SEMANTIC, ran=True, findings=[
        C.Finding(source=C.SEMANTIC, text="UNSUPPORTED: only the owner",
                  severity="high", why="no R* row")])
    text = C.merge(det, sem).feedback()
    assert text.index("MUST FIX") < text.index("SHOULD FIX")
    assert "not opinions" in text
    assert "only the owner" in text


def test_feedback_says_so_when_there_is_nothing_to_fix():
    g = C.merge(C.deterministic(F.ctx(), F.CLEAN), C.not_run(C.SEMANTIC))
    assert "unchanged" in g.feedback()


# --------------------------------------------------------------------------
# the wording of a MUST FIX item
#
# From the first real run: `update()` takes no arguments and returns nothing.
# The draft documented a parameter and a return. The param complaint was fixed
# in one round; the return complaint — worded "more @return tags than the
# declaration returns" — was answered twice with `@return null` and the round
# budget ran out. A complaint the reviser cannot act on costs a whole round, so
# the declaration's own counts go into the wording.
# --------------------------------------------------------------------------

def nullary_pair() -> dict:
    """`update()` — no parameters, no return values."""
    p = dict(F.pair())
    p.update(name="update", signature="update()",
             code="function update() external { last = block.timestamp; }",
             params={}, returns=[])
    return p


def nullary_ctx():
    from natspec_corpus.runner import Context
    return Context(pair=nullary_pair(), table=F.table(), unit=None,
                   version=None)


def test_a_return_tag_on_a_void_function_is_told_to_delete_it():
    v = C.deterministic(nullary_ctx(),
                        "/// @notice Records the current timestamp now.\n"
                        "/// @return null")
    why = next(f.why for f in v.findings if f.text == "return_extra")
    assert "returns nothing" in why
    assert "delete" in why
    assert "placeholder" in why, \
        "the model answered the old wording with `@return null` twice"


def test_a_param_tag_on_a_nullary_function_says_it_takes_none():
    v = C.deterministic(nullary_ctx(),
                        "/// @notice Records the current timestamp now.\n"
                        "/// @param bandData the data from Band")
    why = next(f.why for f in v.findings
               if f.text == "param_unknown:bandData")
    assert "takes no parameters" in why
    assert "delete" in why


def test_an_unknown_param_names_the_ones_the_declaration_has():
    v = C.deterministic(F.ctx(),
                        "/// @notice Adds two unsigned integers and returns "
                        "the sum.\n"
                        "/// @param a the first addend\n"
                        "/// @param b the second addend\n"
                        "/// @param c a third one\n"
                        "/// @return the sum of the two addends")
    why = next(f.why for f in v.findings if f.text == "param_unknown:c")
    assert "`c`" in why
    assert "a, b" in why, "name the real parameters, not just the wrong one"


def test_a_missing_return_says_how_many_are_declared():
    v = C.deterministic(F.ctx(), F.ONE_DEFECT_A)
    why = next(f.why for f in v.findings if f.text == "return_missing")
    assert "returns 1 value" in why


def test_the_complaint_and_the_check_use_the_same_parameter_names():
    """The complaint names the declaration's parameters, and it must take them
    from the same place `judge_text` does — otherwise the critic can report a
    set of parameters the check never used."""
    from natspec_corpus.gate import _declared_params
    v = C.deterministic(F.ctx(), "/// @notice Adds them up somehow here.")
    missing = sorted(f.subject for f in v.findings
                     if f.text.startswith("param_missing"))
    assert missing == sorted(_declared_params(F.pair()))


def test_the_reviser_is_told_not_to_leave_a_placeholder():
    from comgen import prompts as P
    assert "@return null" in P.REVISER.system
    assert "DELETED, never filled in" in P.REVISER.system
