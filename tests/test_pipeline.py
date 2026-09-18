"""Tests for the stage-2 runtime: client, cache, gate, runner, evaluation.

No model is reachable here, so the model is a scripted mock. That is the
point: these tests are about the orchestration — retries, caching, the gate's
decision, claim stripping, per-field scoring — every part of which can be
wrong while the model is perfectly good.
"""
import json
from pathlib import Path

import pytest

from natspec_corpus import prompts_v3 as P
from natspec_corpus.cache import CallCache
from natspec_corpus.evaluate import (aggregate, bleu, fields, reference_fields,
                                     rouge_l, score_record)
from natspec_corpus.gate import attach, judge_text
from natspec_corpus.llm import Client, LLMError, MockBackend, extract_json
from natspec_corpus.runner import Context, generate, strip_claims

GOOD = {
    "L7": {"purpose": "Adds two unsigned integers.", "purpose_ids": ["F1"],
           "caller": None, "caller_ids": [], "preconditions": [],
           "effects": [], "returns_meaning": [], "unknowns": []},
    "L1b": {"natspec": "/// @notice Adds two unsigned integers.\n"
                       "/// @param a the first addend\n"
                       "/// @param b the second addend\n"
                       "/// @return the sum",
            "claims": [{"text": "Adds two unsigned integers.", "ids": ["F1"]}],
            "used_examples": False},
    "L2": {"defects": [], "missing_high_value_facts": []},
    "L3": {"natspec": "/// @notice Adds two unsigned integers.\n"
                      "/// @param a the first addend\n"
                      "/// @param b the second addend\n"
                      "/// @return the sum",
           "changed": False, "removed": [],
           "claims": [{"text": "Adds two unsigned integers.", "ids": ["F1"]}]},
    "L8": {"verdicts": [{"claim": "Adds two unsigned integers.",
                         "rows": ["F1"], "verdict": "SUPPORTED"}],
           "gate": "PASS"},
}


def scripted(script=None):
    script = {**GOOD, **(script or {})}
    return MockBackend(lambda pid, req: script[pid])


def pair_fixture():
    code = ("function add(uint256 a, uint256 b) public pure "
            "returns (uint256) { return a + b; }")
    return {"id": "p1", "file": "x/C.sol", "name": "add", "kind": "function",
            "signature": "add(uint256,uint256)", "container": "C",
            "container_kind": "contract", "split": "val",
            "visibility": "public", "mutability": "pure",
            "code": code, "code_start": 0, "doc_spans": [],
            "notice": "Adds two unsigned integers.", "dev": "",
            "params": {"a": "the first addend", "b": "the second addend"},
            "returns": [{"name": None, "text": "the sum"}],
            "doc_raw": "/// @notice Adds two unsigned integers."}


# -- JSON recovery ---------------------------------------------------------

def test_json_is_recovered_from_a_code_fence():
    assert extract_json('sure:\n```json\n{"a": 1}\n```') == {"a": 1}


def test_json_is_recovered_from_trailing_prose():
    assert extract_json('{"a": 1} hope that helps') == {"a": 1}


def test_unrecoverable_text_returns_none():
    assert extract_json("no object here") is None


def test_a_brace_inside_a_string_does_not_end_the_object():
    assert extract_json('{"a": "x } y"}') == {"a": "x } y"}


# -- client ----------------------------------------------------------------

def test_first_attempt_parse_is_recorded():
    c = Client(scripted())
    call = c.call(P.INTENT_REASONER, source="s", sigma="g")
    assert call.first_attempt_parsed and call.attempts == 1


def test_a_retry_is_recorded_as_a_first_attempt_failure():
    state = {"n": 0}

    def handler(pid, req):
        state["n"] += 1
        return "not json at all" if state["n"] == 1 else GOOD[pid]

    c = Client(MockBackend(handler))
    call = c.call(P.INTENT_REASONER, source="s", sigma="g")
    assert call.attempts == 2 and not call.first_attempt_parsed
    assert c.stats()["L7"]["first_attempt_parse_rate"] == 0.0


def test_the_repair_message_is_appended_only_on_retry():
    seen = []

    def handler(pid, req):
        seen.append(req["messages"][-1]["content"])
        return GOOD[pid] if len(seen) > 1 else "junk"

    Client(MockBackend(handler)).call(P.INTENT_REASONER, source="s", sigma="g")
    assert "not valid JSON" not in seen[0] and "not valid JSON" in seen[1]


def test_persistent_garbage_raises_rather_than_returning_nothing():
    c = Client(MockBackend(lambda pid, req: "never json"), max_attempts=2)
    with pytest.raises(LLMError, match="no schema-valid JSON"):
        c.call(P.INTENT_REASONER, source="s", sigma="g")


def test_a_response_missing_a_required_field_is_a_parse_failure():
    c = Client(MockBackend(lambda pid, req: {"purpose": "x"}), max_attempts=1)
    with pytest.raises(LLMError):
        c.call(P.INTENT_REASONER, source="s", sigma="g")


def test_model_slot_is_resolved():
    b = scripted()
    Client(b, models={"INTENT_REASONER_MODEL": "qwen2.5-coder:7b"}).call(
        P.INTENT_REASONER, source="s", sigma="g")
    assert b.calls[0]["model"] == "qwen2.5-coder:7b"


# -- cache -----------------------------------------------------------------

def test_cache_hit_avoids_a_second_backend_call(tmp_path):
    b = scripted()
    cache = CallCache(tmp_path)
    for _ in range(2):
        Client(b, cache=cache).call(P.INTENT_REASONER, source="s", sigma="g")
    assert len(b.calls) == 1
    assert cache.hits == 1 and cache.misses == 1


def test_changed_input_misses_the_cache(tmp_path):
    b = scripted()
    cache = CallCache(tmp_path)
    Client(b, cache=cache).call(P.INTENT_REASONER, source="s", sigma="g")
    Client(b, cache=cache).call(P.INTENT_REASONER, source="s2", sigma="g")
    assert len(b.calls) == 2


def test_a_cached_call_is_flagged_as_cached(tmp_path):
    cache = CallCache(tmp_path)
    b = scripted()
    Client(b, cache=cache).call(P.INTENT_REASONER, source="s", sigma="g")
    call = Client(b, cache=cache).call(P.INTENT_REASONER, source="s", sigma="g")
    assert call.cached


# -- attaching and judging -------------------------------------------------

def test_attach_replaces_the_old_comment_and_keeps_indentation():
    src = ("contract C {\n    /// @notice old\n"
           "    function f() public {}\n}\n")
    pair = {"code_start": src.index("function f"),
            "doc_spans": [[src.index("/// @notice old"),
                           src.index("function f")]]}
    out = attach(src, pair, "/// @notice new")
    assert "old" not in out
    assert "    /// @notice new\n    function f()" in out


def test_attach_on_a_function_with_no_existing_comment():
    src = "contract C {\n    function f() public {}\n}\n"
    pair = {"code_start": src.index("function f"), "doc_spans": []}
    assert "/// @notice new" in attach(src, pair, "/// @notice new")


def test_judge_text_finds_a_missing_param():
    pair = pair_fixture()
    d = judge_text(pair, "/// @notice Adds two unsigned integers.\n"
                         "/// @param a the first addend\n/// @return the sum")
    assert "param_missing:b" in d


def test_judge_text_accepts_a_complete_comment():
    pair = pair_fixture()
    assert judge_text(pair, GOOD["L1b"]["natspec"]) == []


# -- the runner ------------------------------------------------------------

def test_five_calls_are_made_in_order():
    b = scripted()
    c = Client(b)
    generate(Context(pair=pair_fixture(), table=None), c)
    assert [x.prompt_id for x in c.log] == ["L7", "L1b", "L2", "L3", "L8"]


def test_the_record_keeps_every_intermediate():
    c = Client(scripted())
    rec = generate(Context(pair=pair_fixture(), table=None), c)
    for k in ("intent", "draft", "critique", "refined", "gate",
              "verification", "final"):
        assert k in rec, k


def test_missing_fact_table_is_declared_in_the_evidence_block():
    ctx = Context(pair=pair_fixture(), table=None)
    assert "no fact table" in ctx.sigma_text()
    assert ctx.has_sigma is False


def test_ablation_can_switch_stages_off():
    c = Client(scripted())
    rec = generate(Context(pair=pair_fixture(), table=None), c,
                   stages=["L1b"])
    assert [x.prompt_id for x in c.log] == ["L1b"]
    assert rec["final"] == GOOD["L1b"]["natspec"]


def test_the_generator_cannot_be_switched_off():
    from natspec_corpus.errors import CorpusError
    with pytest.raises(CorpusError):
        generate(Context(pair=pair_fixture(), table=None),
                 Client(scripted()), stages=["L7", "L2"])


def test_a_worse_refinement_is_discarded():
    """The gate's whole purpose: a refinement that raises the defect count is
    thrown away and the draft kept, with no model asked to adjudicate."""
    worse = dict(GOOD["L3"])
    worse["natspec"] = "/// @notice Adds numbers.\n/// @param zzz ghost"
    c = Client(scripted({"L3": worse}))
    rec = generate(Context(pair=pair_fixture(), table=None), c)
    assert rec["gate_kept_draft"] is True
    assert rec["final"] == GOOD["L1b"]["natspec"]


def test_a_better_refinement_is_taken():
    draft = dict(GOOD["L1b"])
    draft["natspec"] = "/// @notice Adds two unsigned integers.\n/// @param a x"
    c = Client(scripted({"L1b": draft}))
    rec = generate(Context(pair=pair_fixture(), table=None), c)
    assert rec["gate_kept_draft"] is False
    assert rec["final"] == GOOD["L3"]["natspec"]


def test_a_contradicted_claim_is_stripped_from_the_output():
    v = {"verdicts": [{"claim": "Only the owner may call this.",
                       "rows": [], "verdict": "CONTRADICTED"}],
         "gate": "FAIL"}
    gen = dict(GOOD["L1b"])
    gen["natspec"] = ("/// @notice Adds two unsigned integers.\n"
                      "/// @dev Only the owner may call this.")
    ref = dict(GOOD["L3"])
    ref["natspec"] = gen["natspec"]
    rec = generate(Context(pair=pair_fixture(), table=None),
                   Client(scripted({"L1b": gen, "L3": ref, "L8": v})))
    assert "Only the owner" not in rec["final"]
    assert rec["gate_verdict"] == "FAIL"


def test_strip_leaves_an_unmatched_claim_alone():
    text = "/// @notice Does a thing."
    assert strip_claims(text, ["something never written"]) == text


def test_strip_never_returns_an_empty_comment():
    text = "/// @notice Does a thing."
    assert strip_claims(text, ["Does a thing."]).strip()


# -- evaluation ------------------------------------------------------------

def test_bleu_is_one_for_an_identical_field():
    assert bleu("adds two numbers", "adds two numbers") == pytest.approx(1.0)


def test_bleu_is_zero_with_no_unigram_overlap():
    assert bleu("completely different words", "adds two numbers") == 0.0


def test_bleu_is_smoothed_for_short_fields():
    """An unsmoothed BLEU-4 reports a hard zero for a good paraphrase of a
    one-sentence @notice, because no 4-gram matches."""
    assert 0.0 < bleu("adds two unsigned numbers", "adds two numbers") < 1.0


def test_rouge_l_rewards_subsequence_overlap():
    assert rouge_l("the sum of a and b", "sum of a and b") > 0.8
    assert rouge_l("unrelated text", "sum of a and b") == 0.0


def test_fields_are_split_per_tag():
    f = fields(GOOD["L1b"]["natspec"])
    assert set(f) == {"notice", "param:a", "param:b", "return:0"}


def test_reference_fields_come_from_the_pair():
    assert set(reference_fields(pair_fixture())) == {
        "notice", "param:a", "param:b", "return:0"}


def test_a_missing_param_shows_as_zero_coverage_not_a_missing_row():
    rec = {"pair_id": "p1", "has_sigma": True,
           "final": "/// @notice Adds two unsigned integers."}
    s = score_record(rec, pair_fixture())
    assert s["by_kind"]["param"]["coverage"] == 0.0
    assert s["by_kind"]["param"]["n"] == 2


def test_fluent_notice_with_invented_params_does_not_score_well_overall():
    """The reason scoring is per field: one blended number hides this."""
    good_notice = {"pair_id": "p", "has_sigma": True,
                   "final": "/// @notice Adds two unsigned integers.\n"
                            "/// @param a nonsense\n/// @param b nonsense\n"
                            "/// @return nonsense"}
    s = score_record(good_notice, pair_fixture())
    assert s["by_kind"]["notice"]["bleu"] > 0.9
    assert s["by_kind"]["param"]["bleu"] < 0.2


def test_aggregate_separates_functions_without_a_fact_table():
    a = score_record({"pair_id": "1", "has_sigma": True,
                      "final": GOOD["L1b"]["natspec"]}, pair_fixture())
    b = score_record({"pair_id": "2", "has_sigma": False,
                      "final": "/// @notice Adds."}, pair_fixture())
    agg = aggregate([a, b])
    assert agg["with_sigma"]["n"] == 1 and agg["without_sigma"]["n"] == 1
    assert agg["all"]["n"] == 2


def test_bleu_handles_a_field_shorter_than_four_tokens():
    """Orders above the sentence length have no n-grams; scoring them as zero
    made every short identical @notice score 0.0."""
    assert bleu("adds numbers", "adds numbers") == pytest.approx(1.0)
    assert bleu("x", "x") == pytest.approx(1.0)


def test_bleu_penalises_a_too_short_candidate():
    assert bleu("adds", "adds two unsigned numbers together") < 0.5


def test_return_name_is_not_counted_twice_when_scoring():
    """`@return depositID The ID` parses with the name still at the head of
    the text; joining both duplicated it and cost a perfect answer 0.13 BLEU."""
    f = fields("/// @return depositID The ID of the deposit")
    assert f["return:0"] == "depositID The ID of the deposit"


def test_oracle_replay_scores_one():
    """Feeding the reference back through scoring must give 1.0. If it does
    not, the metric is measuring a formatting difference between the two
    paths rather than the model."""
    pair = pair_fixture()
    gold = ("/// @notice Adds two unsigned integers.\n"
            "/// @param a the first addend\n/// @param b the second addend\n"
            "/// @return the sum")
    s = score_record({"pair_id": "p", "has_sigma": True, "final": gold}, pair)
    for kind in ("notice", "param", "return"):
        assert s["by_kind"][kind]["bleu"] == pytest.approx(1.0), kind


# -- the gate's agreement with solc ----------------------------------------

GATE_SRC = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

contract G {
    error Unauthorised(address who);
    event Moved(address indexed to, uint256 amount);

    uint256 internal total;

    /// @notice Builds the contract with a starting total.
    /// @param start the initial total
    constructor(uint256 start) { total = start; }

    /// @notice Raised when the caller is not allowed to act.
    /// @param who the rejected caller
    /// @dev carries the caller for off-chain triage

    /// @notice Requires a non-zero amount.
    /// @param amount the amount being checked
    modifier positive(uint256 amount) { require(amount > 0, "zero"); _; }

    /// @notice Adds to the running total.
    /// @param amount the amount to add
    function add(uint256 amount) external positive(amount) { total += amount; }

    /// @notice Halves a value.
    /// @param x the value to halve
    /// @return the value divided by two
    function _half(uint256 x) internal pure returns (uint256) { return x / 2; }
}
"""


@pytest.fixture(scope="module")
def gate_corpus():
    from natspec_corpus.compile import compile_source, installed_versions
    if not installed_versions():
        pytest.skip("no solc")
    from natspec_corpus.extract import build_file
    res = compile_source("G.sol", GATE_SRC)
    assert res.ok, res.error
    model = build_file("G.sol", GATE_SRC)
    by_name = {}
    for a in model.attachments:
        by_name[a.decl.name or a.decl.kind] = {
            "code_start": a.decl.header_start, "doc_spans": [
                [s.start, s.end] for s in a.unit.spans],
            "signature": a.decl.sig, "name": a.decl.name or a.decl.kind,
            "kind": a.decl.kind, "container": "G",
            "visibility": a.decl.visibility,
            "code": a.decl.header}
    return res, by_name


def emits(gate_corpus, name, text):
    from natspec_corpus.gate import solc_emits
    res, by_name = gate_corpus
    return solc_emits(res.unit, "G.sol", by_name[name], text, res.version)


def test_gate_finds_devdoc_for_a_constructor(gate_corpus):
    """solc writes a constructor as the bare word `constructor`, not
    `constructor()`. Matching our own signature dropped every one."""
    ok, tags, _ = emits(gate_corpus, "constructor",
                        "/// @notice Builds it.\n/// @param start the total")
    assert ok and tags


def test_gate_skips_a_modifier(gate_corpus):
    """solc emits no devdoc for modifiers at all, so there is nothing to
    compare and the check must not report a dropped tag."""
    ok, tags, detail = emits(gate_corpus, "positive",
                             "/// @notice Requires more than zero.\n"
                             "/// @param amount the amount")
    assert ok and tags and "modifier" in detail["note"]


def test_gate_skips_an_internal_function(gate_corpus):
    """solc only documents what it exposes; an internal function's NatSpec
    never reaches devdoc however good it is."""
    ok, tags, detail = emits(gate_corpus, "_half",
                             "/// @notice Halves it.\n/// @param x the value\n"
                             "/// @return half of x")
    assert ok and tags and "internal" in detail["note"]


def test_gate_still_catches_an_invented_param(gate_corpus):
    """solc rejects an undocumented parameter name outright rather than
    dropping the tag, so the compile half of the gate catches it first. The
    dropped-tag comparison stays as the safety net for the cases where solc
    is lenient."""
    ok, tags, detail = emits(gate_corpus, "add",
                             "/// @notice Adds to the total.\n"
                             "/// @param ghost not a parameter")
    assert not ok and not tags
    assert any("ghost" in e for e in detail["errors"])


def test_gate_accepts_a_correct_external_function(gate_corpus):
    ok, tags, _ = emits(gate_corpus, "add",
                        "/// @notice Adds to the total.\n"
                        "/// @param amount the amount to add")
    assert ok and tags


def test_gate_matches_a_function_with_a_struct_parameter(gate_corpus):
    """solc expands a struct into its ABI tuple in devdoc keys, so a naive
    comma count reports twenty parameters where the source has two. All twelve
    of Seaport's fulfil* functions were rejected by the gate for it."""
    from natspec_corpus.gate import _arity
    assert _arity("fulfillOrder(Order,address)") == 2
    assert _arity(
        "fulfillOrder((address,address,(uint8,address)[],uint256),bytes32)") == 2
    assert _arity("constructor") == 0
    assert _arity("f()") == 0
