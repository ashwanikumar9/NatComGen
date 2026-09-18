"""Tests for the three-view retriever."""
import json

import numpy as np
import pytest

from natspec_corpus.retrieve import (HashingEncoder, MultiViewIndex,
                                     RetrievalError, Unit, load_units,
                                     build_index, evaluate_loo, label_ceiling)
from natspec_corpus.views import VIEW_NAMES, ast_view, cfg_view, code_view


def unit(pid, sig, code, ast="", cfg="", group="g", proj="p", file="p/a.sol"):
    return Unit(pair_id=pid, file=file, project=proj, group_id=group,
                signature=sig, code=code, natspec="/// @notice x",
                views={"code": code, "ast": ast, "cfg": cfg})


# -- views -----------------------------------------------------------------

def test_code_view_strips_body_comments():
    """The generator sees comment-free source, so the query view must be
    comment-free too — otherwise one body comment matches another."""
    v = code_view({"code": "function f() public { // secret hint\n x = 1; }"})
    assert "secret hint" not in v and "x = 1;" in v


def test_code_view_keeps_string_literals():
    v = code_view({"code": 'function f() public { s = "// not a comment"; }'})
    assert "not a comment" in v


def test_ast_view_is_node_types_only():
    v = ast_view({"ast_types": ["FunctionDefinition", "Block", "Return"]})
    assert v == "FunctionDefinition Block Return"


def test_ast_view_of_a_table_without_one_is_empty():
    assert ast_view({}) == "" and ast_view(None) == ""


def test_cfg_view_carries_branch_labels_and_call_kinds():
    t = {"nodes": [{"id": "N1", "type": "IF"}, {"id": "N2", "type": "RETURN"}],
         "edges": [{"from": "N1", "to": "N2", "label": "true"}],
         "paths": [{"id": "P1", "nodes": ["N1", "N2"]}],
         "calls": [{"kind": "external", "target": "X.y()", "node": "N1"}],
         "reverts": [{"kind": "require", "node": "N1"}]}
    v = cfg_view(t)
    assert "IF [true] RETURN" in v
    assert "call:external" in v and "revert:require" in v


def test_cfg_view_never_leaks_identifiers():
    t = {"nodes": [{"id": "N1", "type": "EXPRESSION"}], "edges": [],
         "paths": [], "calls": [{"kind": "library", "target": "SafeMath.add",
                                 "node": "N1"}], "reverts": []}
    assert "SafeMath" not in cfg_view(t)


# -- encoder ---------------------------------------------------------------

def test_encoder_output_is_normalised_and_deterministic():
    e = HashingEncoder(dim=256)
    a = e.encode(["function f() public {}", "function g() public {}"])
    b = e.encode(["function f() public {}", "function g() public {}"])
    assert a.shape == (2, 256) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a, b)


def test_encoder_handles_an_empty_view():
    v = HashingEncoder(dim=64).encode(["", "x"])
    assert np.isfinite(v).all()


def test_similar_code_scores_above_unrelated_code():
    e = HashingEncoder(dim=512)
    v = e.encode(["function add(uint a, uint b) returns (uint) { return a+b; }",
                  "function sum(uint x, uint y) returns (uint) { return x+y; }",
                  "function transferOwnership(address o) { owner = o; }"])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


# -- fusion ----------------------------------------------------------------

@pytest.fixture(scope="module")
def toy():
    units = [
        unit("a", "add(uint256,uint256)", "function add(uint a, uint b){return a+b;}",
             "FunctionDefinition Block Return", "ENTRYPOINT RETURN |"),
        unit("b", "sub(uint256,uint256)", "function sub(uint a, uint b){return a-b;}",
             "FunctionDefinition Block Return", "ENTRYPOINT RETURN |"),
        unit("c", "setOwner(address)", "function setOwner(address o){owner=o;}",
             "FunctionDefinition Block ExpressionStatement",
             "ENTRYPOINT EXPRESSION |"),
    ]
    return MultiViewIndex(units, HashingEncoder(dim=512))


def test_all_three_indices_are_populated(toy):
    assert {v: toy.indices[v].ntotal for v in VIEW_NAMES} == {
        "code": 3, "ast": 3, "cfg": 3}


def test_search_returns_the_nearest_unit_first(toy):
    q = {"code": "function plus(uint x, uint y){return x+y;}",
         "ast": "FunctionDefinition Block Return",
         "cfg": "ENTRYPOINT RETURN |"}
    hits = toy.search(q, tau=None)
    assert hits[0].unit.pair_id in ("a", "b")


def test_each_hit_carries_per_view_scores(toy):
    hits = toy.search({"code": "function add(uint a, uint b){return a+b;}",
                       "ast": "", "cfg": ""}, tau=None)
    assert set(hits[0].scores) == set(VIEW_NAMES)


def test_an_absent_query_view_contributes_zero_not_nan(toy):
    hits = toy.search({"code": "function add(uint a,uint b){return a+b;}",
                       "ast": "", "cfg": ""}, tau=None)
    assert all(np.isfinite(h.score) for h in hits)
    assert all(h.scores["ast"] == 0.0 for h in hits)


def test_weights_are_equal_and_untuned(toy):
    assert set(toy.weights.values()) == {1 / 3}


def test_threshold_can_drop_every_hit(toy):
    assert toy.search({"code": "zzz", "ast": "", "cfg": ""}, tau=99.0) == []


def test_hit_renders_as_an_exemplar(toy):
    h = toy.search({"code": "function add(uint a,uint b){return a+b;}",
                    "ast": "", "cfg": ""}, tau=None)[0]
    ex = h.as_exemplar()
    assert set(ex) == {"code", "natspec", "scores", "pair_id"}


# -- the allowlist ---------------------------------------------------------

def test_a_never_index_file_is_refused_at_build_time(tmp_path):
    """Enforced when the index is built, never at query time: a vector that
    exists can be leaked by one bug in a filter."""
    (tmp_path / "index_allowlist.json").write_text(json.dumps(
        {"train": [], "val": [], "test": [], "never_index": ["p/dep.sol"]}))
    (tmp_path / "pairs.jsonl").write_text(json.dumps(
        {"id": "1", "file": "p/dep.sol", "project": "p", "group_id": "g",
         "signature": "f()", "code": "function f(){}", "doc_raw": "///x",
         "split": "train"}) + "\n")
    with pytest.raises(RetrievalError, match="never_index"):
        load_units(tmp_path, "train")


def test_a_file_outside_the_allowlist_is_refused(tmp_path):
    (tmp_path / "index_allowlist.json").write_text(json.dumps(
        {"train": ["p/other.sol"], "val": [], "test": [], "never_index": []}))
    (tmp_path / "pairs.jsonl").write_text(json.dumps(
        {"id": "1", "file": "p/a.sol", "project": "p", "group_id": "g",
         "signature": "f()", "code": "function f(){}", "doc_raw": "///x",
         "split": "train"}) + "\n")
    with pytest.raises(RetrievalError, match="allowlist"):
        load_units(tmp_path, "train")


# -- against the real corpus ----------------------------------------------

@pytest.fixture(scope="module")
def real():
    """The real corpus. These three tests are about numbers a toy corpus
    cannot produce — a recall figure over 20 probes, an allowlist covering
    several projects — so they skip rather than pretend when none is built."""
    from conftest import find_corpus
    root = find_corpus()
    if root is None:
        pytest.skip("no built corpus found; run ./run_pipeline.sh --only corpus")
    return root, build_index(root)


def test_index_holds_only_train_units(real):
    root, idx = real
    allow = json.loads((root / "index_allowlist.json").read_text())
    permitted, never = set(allow["train"]), set(allow["never_index"])
    files = {u.file for u in idx.units}
    assert files <= permitted
    assert not (files & never)
    assert not (files & (set(allow["val"]) | set(allow["test"])))


def test_leave_one_out_recall_is_not_broken(real):
    _, idx = real
    r = evaluate_loo(idx)
    assert r["probes"] >= 20
    assert r["recall@1"] > 0.30, r        # weak offline encoder; a real one
    assert r["recall@5"] > 0.60, r        # should do better, never worse


def test_cross_split_label_ceiling_is_recorded(real):
    """Guards against reporting a recall number whose ceiling is 2%."""
    root, idx = real
    c = label_ceiling(idx.units, load_units(root, "val"))
    assert c["signature_ceiling"] < 0.10
