"""The invariants, made to fail.

Every check in `checks.py` has been exercised on data that passes. That proves
it does not fire wrongly; it does not prove it fires at all. An invariant that
has never been made to fail is a comment with a function signature — and the
whole argument for shipping this corpus is that these checks would have caught
the bugs that shipped in v1.

So each test here constructs the defect the check exists for and asserts it is
caught, by name.
"""
import json
from pathlib import Path

import pytest

from natspec_corpus import checks
from natspec_corpus.errors import InvariantError

SRC = ("pragma solidity ^0.8.0;\ncontract C {\n"
       "    /// @notice Adds two unsigned integers.\n"
       "    /// @param a the first addend\n"
       "    /// @param b the second addend\n"
       "    function add(uint256 a, uint256 b) public pure "
       "returns (uint256) { return a + b; }\n}\n")


def pair(src=SRC, **over):
    start = src.index("function add")
    doc_lo = src.index("/// @notice")
    p = {"id": "p1", "project": "proj", "file": "proj/C.sol",
         "container": "C", "container_kind": "contract", "kind": "function",
         "name": "add", "signature": "add(uint256,uint256)",
         "arity_key": "add/2", "visibility": "public", "mutability": "pure",
         "is_virtual": False, "overrides": False, "group_id": "g1",
         "doc_start": doc_lo, "doc_end": start,
         "doc_spans": [[doc_lo, start]], "doc_raw": src[doc_lo:start],
         "notice": "Adds two unsigned integers.", "dev": "",
         "params": {"a": "the first addend", "b": "the second addend"},
         "returns": [], "inheritdoc": None, "resolved_from": None,
         "code_start": start, "code_end": len(src) - 2,
         "code": src[start:len(src) - 2],
         "verified": True, "defects": [], "split": "train"}
    p.update(over)
    return p


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "NatSpecGold"
    (root / "contracts" / "proj").mkdir(parents=True)
    (root / "contracts" / "proj" / "C.sol").write_text(SRC)
    for name, body in (("pairs.jsonl", json.dumps(pair()) + "\n"),
                       ("pairs_partial.jsonl", ""),
                       ("manifest.json", "{}"), ("splits.json", "{}"),
                       ("index_allowlist.json", "{}"),
                       ("README.md", "# corpus\n")):
        (root / name).write_text(body)
    return root


def manifest_for(root, role="scored"):
    import hashlib
    text = (root / "contracts" / "proj" / "C.sol").read_text()
    return {"proj/C.sol": {
        "role": role, "project": "proj",
        "sha1": hashlib.sha1(text.encode()).hexdigest(),
        "bytes": len(text.encode()), "pragma": "^0.8.0",
        "unresolved_imports": []}}


def fails(name, fn, *a, **kw):
    with pytest.raises(InvariantError) as e:
        fn(*a, **kw)
    assert f"[{name}]" in str(e.value), str(e.value)


# -- corpus invariants -----------------------------------------------------

def test_artifacts_exist_catches_a_missing_file(corpus):
    (corpus / "splits.json").unlink()
    fails("artifacts_exist", checks.artifacts_exist, corpus)


def test_artifacts_exist_catches_an_empty_required_file(corpus):
    (corpus / "splits.json").write_text("")
    fails("artifacts_exist", checks.artifacts_exist, corpus)


def test_artifacts_exist_allows_an_empty_partial_file(corpus):
    checks.artifacts_exist(corpus)          # pairs_partial.jsonl is empty


def test_artifacts_exist_catches_an_unsubstituted_placeholder(corpus):
    (corpus / "README.md").write_text("# corpus\n{{COUNTS}}\n")
    fails("artifacts_exist", checks.artifacts_exist, corpus)


def test_manifest_hashes_catches_an_edited_contract(corpus):
    m = manifest_for(corpus)
    (corpus / "contracts" / "proj" / "C.sol").write_text(SRC + "// edited\n")
    fails("manifest_hashes", checks.manifest_hashes, corpus, m)


def test_manifest_hashes_catches_a_file_missing_from_the_manifest(corpus):
    (corpus / "contracts" / "proj" / "D.sol").write_text("contract D {}")
    fails("manifest_hashes", checks.manifest_hashes, corpus,
          manifest_for(corpus))


def test_offsets_slice_back_catches_drifted_code_offsets(corpus):
    fails("offsets_slice_back", checks.offsets_slice_back, corpus,
          [pair(code_start=3)])


def test_offsets_slice_back_catches_doc_spans_that_do_not_reproduce_the_text(
        corpus):
    p = pair()
    p["doc_raw"] = "/// something else entirely"
    fails("offsets_slice_back", checks.offsets_slice_back, corpus, [p])


def test_offsets_slice_back_catches_ends_disagreeing_with_spans(corpus):
    p = pair()
    p["doc_end"] = p["doc_end"] - 1
    fails("offsets_slice_back", checks.offsets_slice_back, corpus, [p])


def test_no_overlapping_docs_catches_the_v1_merged_comment_bug():
    """The apostrophe bug produced exactly this: two declarations whose doc
    ranges overlap because a `*/` was blanked and two blocks merged."""
    a = pair(id="a", doc_start=0, doc_end=100, code_start=100)
    b = pair(id="b", doc_start=50, doc_end=150, code_start=150)
    fails("no_overlapping_docs", checks.no_overlapping_docs, [a, b])


def test_docs_are_single_comments_catches_two_blocks_glued_together():
    """88mph's section banner ending up inside a function's notice."""
    p = pair(doc_raw="/** banner */\n/** @notice real */",
             doc_spans=[[0, 13], [14, 34]])
    fails("docs_are_single_comments", checks.docs_are_single_comments, [p])


def test_docs_are_single_comments_catches_a_mixed_style_doc():
    p = pair(doc_raw="/** @notice x */\n/// @dev y", doc_spans=[[0, 26]])
    fails("docs_are_single_comments", checks.docs_are_single_comments, [p])


def test_no_duplicate_decls_catches_one_declaration_paired_twice():
    fails("no_duplicate_decls", checks.no_duplicate_decls,
          [pair(id="a"), pair(id="b")])


def test_no_duplicate_decls_catches_repeated_ids():
    fails("no_duplicate_decls", checks.no_duplicate_decls,
          [pair(), pair(code_start=1, doc_spans=[[0, 1]], doc_start=0,
                        doc_end=1, doc_raw=SRC[0:1], code=SRC[1:3],
                        code_end=3)])


def test_params_round_trip_catches_a_documented_parameter_that_is_not_declared(
        corpus):
    """The v1 truncated-`mapping` bug: the pair claims parameters the
    declaration does not have."""
    p = pair()
    p["params"] = {"a": "x", "ghost": "y"}
    fails("params_round_trip", checks.params_round_trip, corpus, [p])


def test_params_round_trip_catches_signature_drift(corpus):
    fails("params_round_trip", checks.params_round_trip, corpus,
          [pair(signature="add(uint256)")])


def test_dependencies_are_silent_catches_a_pair_from_a_dependency_file(corpus):
    fails("dependencies_are_silent", checks.dependencies_are_silent,
          [pair()], manifest_for(corpus, role="dependency"))


def test_groups_do_not_straddle_catches_a_twin_split_across_sets():
    """A signature documented in both an interface and its implementation is
    one item of knowledge; split across train and test it inflates the score."""
    fails("groups_do_not_straddle", checks.groups_do_not_straddle,
          [pair(id="a", split="train"), pair(id="b", split="test")])


def test_verified_have_prose_catches_a_verified_pair_with_nothing_in_it():
    fails("verified_have_prose", checks.verified_have_prose,
          [pair(notice="", dev="")])


def test_verified_have_prose_catches_a_verified_pair_carrying_defects():
    fails("verified_have_prose", checks.verified_have_prose,
          [pair(defects=["param_missing:b"])])


def test_imports_resolve_or_are_declared_catches_an_undeclared_import(corpus):
    src = 'import "@openzeppelin/x.sol";\n' + SRC
    (corpus / "contracts" / "proj" / "C.sol").write_text(src)
    fails("imports_resolve_or_are_declared",
          checks.imports_resolve_or_are_declared, corpus, manifest_for(corpus))


def test_allowlist_is_default_deny_catches_a_dependency_left_indexable(corpus):
    (corpus / "index_allowlist.json").write_text(json.dumps(
        {"train": ["proj/C.sol"], "val": [], "test": [], "never_index": []}))
    fails("allowlist_is_default_deny", checks.allowlist_is_default_deny,
          corpus, manifest_for(corpus, role="dependency"))


def test_allowlist_is_default_deny_catches_overlapping_splits(corpus):
    (corpus / "index_allowlist.json").write_text(json.dumps(
        {"train": ["proj/C.sol"], "val": ["proj/C.sol"], "test": [],
         "never_index": []}))
    fails("allowlist_is_default_deny", checks.allowlist_is_default_deny,
          corpus, manifest_for(corpus))


# -- Σ(f) invariants -------------------------------------------------------

def table(**over):
    t = {"file": "proj/C.sol", "contract": "C", "function": "add",
         "canonical_name": "C.add", "char_start": SRC.index("function add"),
         "char_end": len(SRC) - 2, "pair_id": "p1",
         "facts": [{"id": "F1", "kind": "visibility", "text": "public"}],
         "nodes": [{"id": "N1", "type": "ENTRYPOINT",
                    "src": [SRC.index("function add"), len(SRC) - 2],
                    "text": "", "reads": [], "writes": [], "state_reads": [],
                    "state_writes": []}],
         "edges": [], "paths": [{"id": "P1", "nodes": ["N1"], "length": 1}],
         "calls": [], "deps": [], "reverts": [], "events": [],
         "ast_types": ["FunctionDefinition"]}
    t.update(over)
    return t


def test_sigma_tables_join_catches_an_offset_that_is_not_a_declaration(corpus):
    """The byte-versus-character offset failure: a two-position drift attaches
    a function's control flow to its neighbour, silently."""
    fails("sigma_tables_join_to_declarations",
          checks.sigma_tables_join_to_declarations, corpus,
          [table(char_start=7)])


def test_sigma_offsets_are_sane_catches_a_node_outside_its_function(corpus):
    t = table()
    t["nodes"][0]["src"] = [0, 5]
    fails("sigma_offsets_are_sane", checks.sigma_offsets_are_sane, corpus, [t])


def test_sigma_rows_catch_duplicate_ids():
    t = table()
    t["facts"].append({"id": "F1", "kind": "other", "text": "x"})
    fails("sigma_rows_are_well_formed", checks.sigma_rows_are_well_formed, [t])


def test_sigma_rows_catch_a_dangling_edge():
    fails("sigma_rows_are_well_formed", checks.sigma_rows_are_well_formed,
          [table(edges=[{"from": "N1", "to": "N9", "label": ""}])])


def test_sigma_rows_catch_a_dangling_call():
    fails("sigma_rows_are_well_formed", checks.sigma_rows_are_well_formed,
          [table(calls=[{"id": "C1", "kind": "builtin", "target": "require",
                         "node": "N7"}])])


def test_sigma_rows_catch_raw_slithir_in_a_call_target():
    """The library call reported twice, once readable and once as raw IR."""
    fails("sigma_rows_are_well_formed", checks.sigma_rows_are_well_formed,
          [table(calls=[{"id": "C1", "kind": "external", "node": "N1",
                         "target": "TMP_63 = LIBRARY_CALL, dest:L"}])])


def test_sigma_rows_catch_a_path_that_revisits_a_node():
    fails("sigma_rows_are_well_formed", checks.sigma_rows_are_well_formed,
          [table(paths=[{"id": "P1", "nodes": ["N1", "N1"], "length": 2}])])


def test_sigma_rows_catch_slithir_temporaries_in_a_dependency():
    fails("sigma_rows_are_well_formed", checks.sigma_rows_are_well_formed,
          [table(deps=[{"id": "D1", "variable": "TMP_5",
                        "depends_on": ["a"]}])])


def test_sigma_tables_are_unique_catches_a_function_analysed_twice():
    """Slither's `contract.functions` includes inherited declarations, so an
    interface implemented in the same file produced two tables at one offset."""
    fails("sigma_tables_are_unique", checks.sigma_tables_are_unique,
          [table(), table()])


def test_sigma_pair_ids_catch_a_citation_to_a_pair_that_does_not_exist(corpus):
    fails("sigma_pair_ids_exist", checks.sigma_pair_ids_exist, corpus,
          [table(pair_id="nope")])


def test_sigma_pair_ids_catch_a_pair_pointing_at_another_offset(corpus):
    fails("sigma_pair_ids_exist", checks.sigma_pair_ids_exist, corpus,
          [table(char_start=SRC.index("function add"), pair_id="p1",
                 file="proj/other.sol")])


# -- and the passing case, so none of the above is vacuous ----------------

def test_every_check_passes_on_a_well_formed_corpus(corpus):
    m = manifest_for(corpus)
    (corpus / "index_allowlist.json").write_text(json.dumps(
        {"train": ["proj/C.sol"], "val": [], "test": [], "never_index": []}))
    checks.run_all(corpus, {}, [pair()], [], m)
    checks.run_sigma_checks(corpus, [table()])
