"""Recovering a reply that answered the question but skipped the paperwork.

Measured on the C-series: 8% of records were lost to `L3: no schema-valid JSON
after 3 attempts`, and the replies were not malformed. They carried a perfectly
good `natspec` and omitted `changed` or `claims`. C0 and C6 — the only two
configurations that never run L3 — had a zero error rate, which is what
identified the cause.

Throwing away a written comment because the model did not also report whether
it had changed anything is a bookkeeping failure charged to the experiment. But
the repair has to stay narrow, or it turns "the model failed" into "the model
succeeded with empty output", which is worse than losing the record. These
tests pin both edges.
"""
from __future__ import annotations

import json

import pytest

from natspec_corpus import prompts_v3 as P
from natspec_corpus.llm import Client, LLMError, MockBackend, repair

REFINER = P.REFINER.schema
GENERATOR = P.GENERATOR.schema


# --------------------------------------------------------------------------
# what gets repaired
# --------------------------------------------------------------------------

def test_a_reply_missing_only_bookkeeping_is_repaired():
    out = repair({"natspec": "/// @notice Adds two integers together."}, REFINER)
    assert out["natspec"] == "/// @notice Adds two integers together."
    assert out["changed"] is False
    assert out["claims"] == []


def test_the_model_s_own_values_are_never_overwritten():
    """`claims` is absent and filled; `changed` was reported and is kept."""
    out = repair({"natspec": "///", "changed": True}, REFINER)
    assert out["changed"] is True, "the model said True; the default is False"
    assert out["claims"] == []


def test_a_complete_reply_needs_no_repair():
    assert repair({"natspec": "///", "changed": False, "claims": []},
                  REFINER) is None, "nothing absent, so nothing to fill"


# --------------------------------------------------------------------------
# what does NOT get repaired — the edge that matters more
# --------------------------------------------------------------------------

def test_a_reply_without_the_substantive_field_is_still_a_failure():
    """This is the line. Filling in `natspec` would turn a failed call into a
    successful empty comment, which is worse than losing the record."""
    assert repair({"changed": True, "claims": []}, REFINER) is None


def test_an_unknown_missing_field_is_not_invented():
    schema = {"required": ["natspec", "something_we_have_no_default_for"]}
    assert repair({"natspec": "///"}, schema) is None


def test_a_non_dict_is_not_repaired():
    assert repair(None, REFINER) is None
    assert repair(["natspec"], REFINER) is None


def test_a_schema_with_no_substantive_field_is_not_repaired():
    assert repair({"a": 1}, {"required": ["a", "claims"]}) is None


def test_only_the_generator_and_refiner_are_repairable():
    """The critic, the judge and the intent reader are left strict on purpose.
    An empty `defects` list reads as "this comment is clean" and an empty
    `verdicts` list reads as "nothing to verify" — inventing either would
    corrupt a result, where losing the record merely shrinks the sample."""
    assert repair({"defects": []}, P.SEMANTIC_CRITIC.schema) is None
    assert repair({"verdicts": []}, P.CLAIM_VERIFIER.schema) is None
    assert repair({"purpose": "x"}, P.INTENT_REASONER.schema) is None


# --------------------------------------------------------------------------
# end to end through the client
# --------------------------------------------------------------------------

def scripted(reply):
    return Client(MockBackend(lambda pid, req: reply))


def test_the_client_recovers_the_call_and_flags_it():
    c = scripted({"natspec": "/// @notice Adds two integers together."})
    call = c.call(P.REFINER, source="f", sigma="s", candidate="c", critique="{}")
    assert call.data["natspec"] == "/// @notice Adds two integers together."
    assert call.repaired is True
    assert call.first_attempt_parsed is False, \
        "a repaired call must never count as a clean first-attempt parse"
    assert call.summary()["repaired"] is True


def test_a_clean_reply_is_not_flagged_as_repaired():
    c = scripted({"natspec": "///", "changed": False, "claims": []})
    call = c.call(P.REFINER, source="f", sigma="s", candidate="c", critique="{}")
    assert call.repaired is False
    assert call.first_attempt_parsed is True


def test_an_unusable_reply_still_raises():
    c = scripted({"changed": True, "claims": []})
    with pytest.raises(LLMError, match="no schema-valid JSON"):
        c.call(P.REFINER, source="f", sigma="s", candidate="c", critique="{}")


def test_prose_instead_of_json_still_raises():
    c = Client(MockBackend(lambda pid, req: "I cannot help with that."))
    with pytest.raises(LLMError):
        c.call(P.REFINER, source="f", sigma="s", candidate="c", critique="{}")


def test_the_repair_rate_is_visible_in_the_stats():
    c = scripted({"natspec": "/// @notice Adds two integers together."})
    for _ in range(3):
        c.call(P.REFINER, source="f", sigma="s", candidate="c", critique="{}")
    assert c.stats()["L3"]["repaired"] == 3
    assert c.stats()["L3"]["first_attempt_parse_rate"] == 0.0


def test_the_generator_is_covered_too():
    """L1b omits `used_examples` on the same model, and losing a draft costs
    the whole function rather than one refinement."""
    c = scripted({"natspec": "/// @notice Adds two integers together.",
                  "claims": []})
    call = c.call(P.GENERATOR, source="f", sigma="s", intent="{}", exemplars="")
    assert call.repaired is True
    assert call.data["used_examples"] is False


def test_comgen_s_reviser_is_covered_by_the_same_rule():
    from comgen import prompts as CP
    c = scripted({"natspec": "/// @notice Adds two integers together."})
    call = c.call(CP.REVISER, source="f", sigma="s", candidate="c",
                  feedback="none")
    assert call.repaired is True
    assert call.data["changed"] is False
