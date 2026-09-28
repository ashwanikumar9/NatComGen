"""tools/audit_corpus.py — each hard invariant, made to fail on purpose.

An audit nobody has seen fail is not an audit. These build small corpora that
violate one invariant each and check that the tool says so, because a checker
that always prints PASS is worse than no checker: it is a false assurance
attached to every number downstream of it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import audit_corpus as A                              # noqa: E402


def pair(pid, *, split="train", project="p", params=None, returns=(),
         notice="Adds two unsigned integers and returns the sum.", dev="",
         name="add", kind="function",
         code="function add(uint256 a, uint256 b) public pure "
              "returns (uint256) { return a + b; }"):
    return {"id": pid, "project": project, "file": f"{project}/C.sol",
            "container": "C", "container_kind": "contract", "kind": kind,
            "name": name, "signature": "add(uint256,uint256)",
            "visibility": "public", "mutability": "pure", "split": split,
            "code": code, "notice": notice, "dev": dev,
            "params": {"a": "the first addend", "b": "the second addend"}
            if params is None else params,
            "returns": list(returns) or [{"name": None, "text": "the sum"}],
            "doc_raw": "/// @notice " + notice}


def corpus(tmp_path: Path, pairs, sigma_ids=()) -> Path:
    root = tmp_path / "C"
    root.mkdir(exist_ok=True)
    (root / "pairs.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
    (root / "sigma").mkdir(exist_ok=True)
    (root / "sigma" / "sigma.jsonl").write_text(
        "".join(json.dumps({"pair_id": i}) + "\n" for i in sigma_ids),
        encoding="utf-8")
    return root


def test_a_complete_corpus_passes_every_invariant(tmp_path):
    a = A.audit(corpus(tmp_path, [
        pair("a"),
        pair("b", split="val", project="q",
             notice="Returns the sum of the two arguments given.")]))
    assert a["gold_complete"]
    assert a["reconstruction_agrees"]
    assert a["returns_complete"]
    assert a["splits_disjoint"]
    assert a["no_leakage"]


def test_an_undocumented_parameter_fails_check_one(tmp_path):
    a = A.audit(corpus(tmp_path, [pair("a", params={"a": "only the first"})]))
    assert not a["gold_complete"]
    assert a["gold_defects"] == {"param_missing": 1}
    assert a["gold_offenders"][0]["defects"] == ["param_missing:b"]


def test_a_return_the_corpus_never_recorded_fails_check_two_b(tmp_path):
    """The question the audit was asked, and the one check 1 cannot answer.

    The scorer counts a declaration's returns from `pair["returns"]` — the same
    field the reference is built from — so the two always agree and a return the
    corpus simply never recorded is invisible. Parsing the signature is the only
    independent witness, and here the signature returns two values while the
    corpus records one.
    """
    a = A.audit(corpus(tmp_path, [pair(
        "a", returns=[{"name": None, "text": "the sum"}],
        code="function add(uint256 a, uint256 b) public pure "
             "returns (uint256, uint256) { return (a, b); }")]))
    assert a["gold_complete"], "check 1 sees nothing wrong — that is the point"
    assert not a["returns_complete"]
    m = a["return_mismatches"][0]
    assert (m["declared"], m["recorded"]) == (2, 1)


def test_an_unparsable_signature_is_reported_not_counted_as_agreement(tmp_path):
    a = A.audit(corpus(tmp_path, [pair("a", code="not solidity at all")]))
    assert a["returns_unparsed"] == 1
    assert a["returns_complete"], "unparsed is not a mismatch"


def test_a_gold_with_an_empty_return_description_fails_check_one(tmp_path):
    a = A.audit(corpus(tmp_path, [pair(
        "a", returns=[{"name": None, "text": ""}])]))
    assert not a["gold_complete"]
    assert "return_empty" in a["gold_defects"]


def test_a_gold_with_no_prose_fails_check_one(tmp_path):
    a = A.audit(corpus(tmp_path, [pair("a", notice="", dev="")]))
    assert not a["gold_complete"]
    assert "no_text" in a["gold_defects"]


def test_a_documented_param_the_code_does_not_declare_fails_check_two(tmp_path):
    """The invariant that underwrites every defect number in every table: if
    the critic's reconstruction cannot see a parameter the corpus recorded, the
    critic reports a defect that is not there."""
    a = A.audit(corpus(tmp_path, [pair(
        "a", params={"a": "first", "b": "second", "phantom": "not declared"})]))
    assert not a["reconstruction_agrees"]
    assert a["reconstruction_mismatches"][0]["recorded"] == \
        ["a", "b", "phantom"]


def test_a_project_in_two_splits_fails_check_three(tmp_path):
    a = A.audit(corpus(tmp_path, [pair("a", split="train", project="same"),
                                  pair("b", split="val", project="same")]))
    assert not a["splits_disjoint"]
    assert a["project_overlaps"]["train|val"] == ["same"]


def test_a_reference_copied_from_train_fails_check_four(tmp_path):
    """SmartDoc's published 47.39 rests on 434 of its 1,000 test references
    appearing verbatim in its own training file. This is the check that says
    whether our number is of the same kind."""
    leaked = "Adds two unsigned integers and returns the sum."
    a = A.audit(corpus(tmp_path, [pair("a", split="train", notice=leaked),
                                  pair("b", split="val", project="q",
                                       notice=leaked.upper())]))
    assert not a["no_leakage"], "the match is case- and whitespace-insensitive"
    assert a["leakage"]["val"]["n"] == 1


def test_sigma_coverage_is_reported_per_split_and_kind(tmp_path):
    root = corpus(tmp_path,
                  [pair("a", split="val", project="q"),
                   pair("b", split="val", project="q"),
                   pair("c", split="val", project="q", kind="event")],
                  sigma_ids=["a"])
    a = A.audit(root)
    assert a["sigma_coverage"]["val"]["function"]["coverage"] == 0.5
    assert a["sigma_coverage"]["val"]["event"]["coverage"] == 0.0
    assert a["sigma_coverage"]["val"]["ALL"]["with_sigma"] == 1


def test_the_exit_code_is_non_zero_when_an_invariant_fails(tmp_path):
    root = corpus(tmp_path, [pair("a", params={"a": "only the first"})])
    assert A.main(["--corpus", str(root)]) == 1
    root = corpus(tmp_path, [pair("a")])
    assert A.main(["--corpus", str(root)]) == 0
