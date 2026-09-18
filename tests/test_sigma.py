"""Tests for the compile + Σ(f) stage.

These run a real compiler and a real Slither analysis, so they are slower than
the extraction tests. They are worth it: every failure mode in this stage is
silent — a wrong offset attaches facts to the neighbouring function, a
duplicated call row inflates the evidence, an unfiltered dependency row fills
the prompt with SlithIR temporaries.
"""
import pytest

from natspec_corpus.compile import (Unit, compile_source, compile_unit,
                                    installed_versions, unit_for)
from natspec_corpus.extract import build_file
from natspec_corpus.sigma import OffsetMap, analyze

pytestmark = pytest.mark.skipif(not installed_versions(),
                                reason="no solc available")

BRANCHY = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library Math {
    function double(uint256 x) internal pure returns (uint256) {
        return x * 2;
    }
}

contract Vault {
    uint256 public total;
    mapping(address => uint256) public balance;

    event Deposited(address indexed who, uint256 amount);

    modifier positive(uint256 amount) {
        require(amount > 0, "zero");
        _;
    }

    /// @notice Deposits `amount` for the caller.
    function deposit(uint256 amount) external positive(amount) returns (uint256 newTotal) {
        uint256 doubled = Math.double(amount);
        if (doubled > 100) {
            total += doubled;
        } else {
            total += amount;
        }
        balance[msg.sender] += amount;
        emit Deposited(msg.sender, amount);
        newTotal = total;
    }

    function peek() external view returns (uint256) {
        return total;
    }
}
"""


@pytest.fixture(scope="module")
def analysis():
    res = compile_source("Vault.sol", BRANCHY)
    assert res.ok, res.error
    return analyze(res.unit, res.version), res


def table(analysis, name):
    tables, _ = analysis
    return next(t for t in tables if t.function == name)


# -- offsets ---------------------------------------------------------------

def test_offset_map_is_identity_for_ascii():
    m = OffsetMap("contract C { }")
    assert m.to_char(7) == 7 and m.to_byte(7) == 7


def test_offset_map_handles_a_multibyte_character():
    """An em-dash is one character and three bytes. Uniswap's Tick.sol has
    exactly one, at char 834, which shifts every later byte offset by two —
    enough to attach a function's facts to the wrong declaration."""
    src = "/// a — dash\ncontract C { }"
    m = OffsetMap(src)
    c = src.index("contract")
    assert m.to_byte(c) == c + 2
    assert m.to_char(m.to_byte(c)) == c


def test_offset_map_round_trips_every_position():
    src = "// héllo — wörld\ncontract C { function f() public {} }"
    m = OffsetMap(src)
    assert all(m.to_char(m.to_byte(i)) == i for i in range(len(src)))


# -- compile ---------------------------------------------------------------

def test_unresolved_import_is_reported_not_raised():
    src = 'import "@openzeppelin/contracts/token/ERC20/ERC20.sol";\ncontract C {}'
    res = compile_source("C.sol", src)
    assert not res.ok
    assert "unresolved" in (res.error or "")
    assert res.unit.unresolved


def test_multi_file_unit_is_assembled_from_relative_imports():
    lib = "pragma solidity ^0.8.0;\nlibrary L { function f() internal pure {} }"
    main = ('pragma solidity ^0.8.0;\nimport "./lib/L.sol";\n'
            "contract C { function g() public pure { L.f(); } }")
    u = unit_for("C.sol", {"C.sol": main, "lib/L.sol": lib}.get)
    assert set(u.sources) == {"C.sol", "lib/L.sol"} and u.complete


def test_version_search_picks_a_working_compiler():
    res = compile_source("C.sol", "pragma solidity ^0.8.0;\ncontract C {}")
    assert res.ok and res.version.startswith("0.8")


def test_compile_is_cached_by_content(tmp_path):
    src = "pragma solidity ^0.8.0;\ncontract C {}"
    a = compile_source("C.sol", src, cache_dir=tmp_path)
    b = compile_source("C.sol", src, cache_dir=tmp_path)
    assert a.ok and b.ok and a.version == b.version
    assert len(list(tmp_path.glob("*.json"))) == 1


# -- the join --------------------------------------------------------------

def test_sigma_offsets_match_the_extractor(analysis):
    """The join key between Σ(f) and the corpus. If this drifts, facts are
    silently attached to the wrong function."""
    tables, _ = analysis
    decls = {d.name: d.header_start for d in build_file("Vault.sol", BRANCHY).decls}
    for t in tables:
        if t.function in decls:
            assert t.char_start == decls[t.function], t.function
    assert {"deposit", "peek"} <= {t.function for t in tables}


def test_offsets_slice_back_to_the_declaration(analysis):
    tables, _ = analysis
    t = table(analysis, "deposit")
    assert BRANCHY[t.char_start:].startswith("function deposit")


# -- facts -----------------------------------------------------------------

def test_facts_capture_visibility_params_returns_and_modifiers(analysis):
    t = table(analysis, "deposit")
    kinds = {f["kind"]: f["text"] for f in t.facts}
    assert kinds["visibility"] == "external"
    assert any(f["kind"] == "modifier" and f["text"] == "positive"
               for f in t.facts)
    assert any(f["kind"] == "parameter" and f["name"] == "amount"
               for f in t.facts)
    assert any(f["kind"] == "return" and f["name"] == "newTotal"
               for f in t.facts)


def test_state_reads_and_writes_are_recorded(analysis):
    t = table(analysis, "deposit")
    written = next(f for f in t.facts if f["kind"] == "state_written")
    assert "total" in written["variables"]
    assert "view" not in {f["text"] for f in t.facts}


def test_view_function_is_marked(analysis):
    t = table(analysis, "peek")
    assert any(f["kind"] == "mutability" and f["text"] == "view"
               for f in t.facts)


# -- cfg -------------------------------------------------------------------

def test_branch_produces_labelled_true_and_false_edges(analysis):
    t = table(analysis, "deposit")
    labels = {e["label"] for e in t.edges}
    assert "true" in labels and "false" in labels
    assert any(n["type"] == "IF" for n in t.nodes)


def test_every_edge_points_at_a_real_node(analysis):
    t = table(analysis, "deposit")
    ids = {n["id"] for n in t.nodes}
    assert all(e["from"] in ids and e["to"] in ids for e in t.edges)


def test_node_offsets_lie_inside_the_function(analysis):
    t = table(analysis, "deposit")
    for n in t.nodes:
        if n["src"][0] >= 0:
            assert t.char_start <= n["src"][0] <= t.char_end, n


def test_paths_start_at_the_entry_node(analysis):
    t = table(analysis, "deposit")
    assert t.paths
    assert all(p["nodes"][0] == t.paths[0]["nodes"][0] for p in t.paths)
    assert all(len(set(p["nodes"])) == len(p["nodes"]) for p in t.paths)


# -- calls -----------------------------------------------------------------

def test_library_call_is_named_and_not_duplicated_as_external(analysis):
    """Slither reports a library call under both `library_calls` and
    `high_level_calls`; the second stringifies to a whole SlithIR line."""
    t = table(analysis, "deposit")
    lib = [c for c in t.calls if c["kind"] == "library"]
    assert any("double" in c["target"] for c in lib)
    # `=>` appears legitimately in a canonical name with a mapping parameter,
    # so the marker for leaked IR is the operation syntax itself.
    assert not any(m in c["target"] for c in t.calls
                   for m in (" = ", "dest:", "LIBRARY_CALL"))
    targets = [(c["kind"], c["target"], c["node"]) for c in t.calls]
    assert len(targets) == len(set(targets))


def test_builtin_call_is_not_also_reported_as_internal(analysis):
    tables, _ = analysis
    t = next(t for t in tables if t.function == "positive")
    kinds = {c["kind"] for c in t.calls if "require" in c["target"]}
    assert kinds == {"builtin"}


def test_every_call_row_points_at_a_real_node(analysis):
    t = table(analysis, "deposit")
    ids = {n["id"] for n in t.nodes}
    assert all(c["node"] in ids for c in t.calls if c["node"])


# -- data dependency -------------------------------------------------------

def test_dependencies_are_in_function_scope(analysis):
    t = table(analysis, "deposit")
    scope = {"amount", "doubled", "newTotal", "total", "balance"}
    for d in t.deps:
        assert d["variable"] in scope, d
        assert set(d["depends_on"]) <= scope, d


def test_slithir_temporaries_never_appear(analysis):
    tables, _ = analysis
    blob = repr([t.deps for t in tables])
    assert "TMP_" not in blob and "REF_" not in blob


def test_the_return_value_depends_on_the_input(analysis):
    t = table(analysis, "deposit")
    dep = {d["variable"]: set(d["depends_on"]) for d in t.deps}
    assert "amount" in dep.get("total", set()) | dep.get("newTotal", set())


# -- row ids ---------------------------------------------------------------

def test_row_ids_are_unique_and_prefixed(analysis):
    t = table(analysis, "deposit")
    ids = t.row_ids
    assert len(ids) == len(set(ids))
    assert all(i[0] in "FNPCDRE" for i in ids)


def test_name_lists_are_sorted_for_reproducibility(analysis):
    """Slither returns some of these as sets, so iteration order varies with
    PYTHONHASHSEED and two runs of the same build differ byte for byte."""
    tables, _ = analysis
    for t in tables:
        for n in t.nodes:
            for key in ("reads", "writes", "state_reads", "state_writes"):
                assert n[key] == sorted(n[key]), (t.function, n["id"], key)


def test_analysis_is_reproducible(analysis):
    tables, res = analysis
    import json
    again = analyze(res.unit, res.version)
    assert ([json.dumps(t.to_dict(), sort_keys=True) for t in tables]
            == [json.dumps(t.to_dict(), sort_keys=True) for t in again])


def test_call_rows_are_sorted_within_each_node(analysis):
    """Slither's call collections are sets; without a sort the same facts get
    different C-numbers per run and a cited id stops meaning anything."""
    from natspec_corpus.sigma import _CALL_PRIORITY
    tables, _ = analysis
    for t in tables:
        by_node = {}
        for c in t.calls:
            by_node.setdefault(c["node"], []).append(
                (_CALL_PRIORITY.get(c["kind"], 9), c["target"]))
        for node, rows in by_node.items():
            assert rows == sorted(rows), (t.function, node)


def test_dependency_relation_is_transitively_closed(analysis):
    """Slither's raw table is not a fixpoint and its contents vary with
    PYTHONHASHSEED; closing it makes the rows reproducible and complete."""
    t = table(analysis, "deposit")
    dep = {d["variable"]: set(d["depends_on"]) for d in t.deps}
    for var, srcs in dep.items():
        for s in list(srcs):
            assert dep.get(s, set()) <= srcs | {var}, (var, s)


def test_dependency_survives_a_slithir_temporary():
    """`total += Math.double(amount)` reaches `amount` only through a TMP."""
    res = compile_source("Vault.sol", BRANCHY)
    t = next(x for x in analyze(res.unit, res.version) if x.function == "deposit")
    dep = {d["variable"]: set(d["depends_on"]) for d in t.deps}
    assert "amount" in dep.get("total", set())


def test_compilers_are_identified_by_what_they_report(tmp_path, monkeypatch):
    """A compiler installed into a wrongly-named directory must be registered
    under its real version. In this container solc 0.7.6 sat in the `0.8.7`
    folder, so every file pinned to `=0.7.6` was skipped as uncompilable."""
    import natspec_corpus.compile as C
    d = tmp_path / "9.9.9"
    d.mkdir()
    exe = d / "solc"
    exe.write_text("#!/bin/sh\necho 'Version: 0.6.11+commit.deadbeef'\n")
    exe.chmod(0o755)
    monkeypatch.setattr(C, "SHIM_ROOT", tmp_path)
    monkeypatch.setattr(C, "_probed", None)
    assert C.installed_versions() == ["0.6.11"]
    assert C.solc_path("0.6.11") == str(exe)
    with pytest.raises(Exception):
        C.solc_path("9.9.9")
    monkeypatch.setattr(C, "_probed", None)


# -- reverts, events, interaction order ------------------------------------

REENTRANT = """pragma solidity ^0.8.0;
interface IToken { function transfer(address to, uint256 v) external returns (bool); }
contract V {
  mapping(address => uint256) bal; address owner; IToken tok;
  event Paid(address indexed to, uint256 amount);
  function withdraw(uint256 amount) external {
    require(msg.sender == owner, "not owner");
    require(amount > 0 && amount <= bal[msg.sender], "bad amount");
    tok.transfer(msg.sender, amount);
    bal[msg.sender] -= amount;
    emit Paid(msg.sender, amount);
  }
  function safeWithdraw(uint256 amount) external {
    bal[msg.sender] -= amount;
    tok.transfer(msg.sender, amount);
  }
}
"""


@pytest.fixture(scope="module")
def reentrant():
    res = compile_source("V.sol", REENTRANT)
    assert res.ok, res.error
    return {t.function: t for t in analyze(res.unit, res.version)}


def test_revert_conditions_are_captured_as_written(reentrant):
    """Slither prints the lowered `require(bool,string)(cond,msg)`; the first
    paren group is solc's type signature, so a naive parse records every
    condition as the word "bool"."""
    rs = reentrant["withdraw"].reverts
    assert [r["condition"] for r in rs] == [
        "msg.sender == owner", "amount > 0 && amount <= bal[msg.sender]"]
    assert [r["message"] for r in rs] == ["not owner", "bad amount"]


def test_revert_condition_keeps_a_nested_comma(reentrant):
    r = reentrant["withdraw"].reverts[1]
    assert "bal[msg.sender]" in r["condition"] and "," not in r["message"]


def test_events_are_captured_with_arguments(reentrant):
    e = reentrant["withdraw"].events
    assert len(e) == 1 and e[0]["event"] == "Paid"
    assert e[0]["args"] == ["msg.sender", "amount"]


def test_state_write_after_an_external_call_is_flagged(reentrant):
    f = [x for x in reentrant["withdraw"].facts
         if x["kind"] == "interaction_order"]
    assert f and "re-enter" in f[0]["text"]


def test_correct_ordering_is_reported_as_safe(reentrant):
    f = [x for x in reentrant["safeWithdraw"].facts
         if x["kind"] == "interaction_order"]
    assert f and "precedes" in f[0]["text"]


def test_no_external_call_means_no_interaction_row(analysis):
    t = table(analysis, "deposit")
    assert not [x for x in t.facts if x["kind"] == "interaction_order"]


def test_revert_and_event_rows_point_at_real_nodes(reentrant):
    t = reentrant["withdraw"]
    ids = {n["id"] for n in t.nodes}
    assert all(r["node"] in ids for r in t.reverts)
    assert all(e["node"] in ids for e in t.events)


def test_compile_error_is_a_real_exception_class():
    """It is raised in four places; an earlier refactor deleted the class
    definition and left every raise site as a latent NameError, because no
    test had ever made a solc subprocess fail."""
    from natspec_corpus.compile import CompileError
    from natspec_corpus.errors import CorpusError
    assert issubclass(CompileError, CorpusError)
    with pytest.raises(CompileError):
        raise CompileError("boom")


def test_unknown_version_raises_rather_than_crashing():
    import natspec_corpus.compile as C
    with pytest.raises(C.CompileError, match="not installed"):
        C.solc_path("0.0.1")


INHERITING = """pragma solidity ^0.8.0;
interface IThing {
    /// @notice Does the thing.
    function act(uint256 x) external returns (uint256);
}
contract Thing is IThing {
    function act(uint256 x) external returns (uint256) { return x + 1; }
}
"""


def test_an_interface_in_the_same_file_is_not_analysed_twice():
    """Slither's `contract.functions` includes inherited declarations, so a
    contract implementing an interface declared in the same file yielded that
    interface's function a second time at the same source offset."""
    res = compile_source("T.sol", INHERITING)
    assert res.ok, res.error
    tables = analyze(res.unit, res.version, ast_root=res.ast("T.sol"))
    offsets = [t.char_start for t in tables]
    assert len(offsets) == len(set(offsets)), [
        (t.contract, t.function, t.char_start) for t in tables]
    assert {(t.contract, t.function) for t in tables} == {
        ("IThing", "act"), ("Thing", "act")}


# -- resumable analysis ----------------------------------------------------
# A killed sigma run used to lose every file it had analysed, because the
# fact tables were only written at the very end. These cover the per-file
# shards that make an interrupted run resume where it stopped, and — just as
# important — the cases where a shard must NOT be reused.

def _unit(sources, rel="C.sol"):
    return unit_for(rel, sources.get)


def test_shard_key_is_stable_for_the_same_inputs():
    from natspec_corpus.sigma_build import shard_key
    u = _unit({"C.sol": "contract C {}"})
    assert shard_key("C.sol", u, "0.8.13") == shard_key("C.sol", u, "0.8.13")


def test_shard_key_changes_when_the_source_changes():
    from natspec_corpus.sigma_build import shard_key
    a = _unit({"C.sol": "contract C {}"})
    b = _unit({"C.sol": "contract C { uint x; }"})
    assert shard_key("C.sol", a, "0.8.13") != shard_key("C.sol", b, "0.8.13")


def test_shard_key_changes_when_an_imported_file_changes():
    """The shard covers the whole compilation unit, so editing a library the
    file imports has to invalidate it too."""
    from natspec_corpus.sigma_build import shard_key
    main = ('pragma solidity ^0.8.0;\nimport "./L.sol";\n'
            "contract C { function g() public pure { L.f(); } }")
    a = _unit({"C.sol": main,
               "L.sol": "library L { function f() internal pure {} }"})
    b = _unit({"C.sol": main,
               "L.sol": "library L { function f() internal pure { } }"})
    assert shard_key("C.sol", a, "0.8.13") != shard_key("C.sol", b, "0.8.13")


def test_shard_key_changes_with_the_compiler():
    from natspec_corpus.sigma_build import shard_key
    u = _unit({"C.sol": "contract C {}"})
    assert shard_key("C.sol", u, "0.8.13") != shard_key("C.sol", u, "0.7.6")


def test_shard_key_changes_when_the_analysis_code_changes(monkeypatch):
    """Otherwise a rebuild after editing sigma.py would serve fact tables
    produced by the previous version of the analysis."""
    import natspec_corpus.sigma_build as sb
    u = _unit({"C.sol": "contract C {}"})
    before = sb.shard_key("C.sol", u, "0.8.13")
    monkeypatch.setattr(sb, "_ANALYSIS_VERSION", "deadbeefdeadbeef")
    assert sb.shard_key("C.sol", u, "0.8.13") != before


def test_a_truncated_shard_is_ignored_rather_than_crashing(tmp_path):
    """This is the shard a kill lands on. It must read as 'not cached'."""
    from natspec_corpus.sigma_build import _read_shard
    p = tmp_path / "s.json"
    p.write_text('[{"id": "F1", "cha')
    assert _read_shard(p) is None


def test_a_missing_shard_reads_as_not_cached(tmp_path):
    from natspec_corpus.sigma_build import _read_shard
    assert _read_shard(tmp_path / "absent.json") is None


def test_a_shard_holding_something_other_than_rows_is_ignored(tmp_path):
    from natspec_corpus.sigma_build import _read_shard
    p = tmp_path / "s.json"
    p.write_text('{"oops": true}')
    assert _read_shard(p) is None


def test_shard_round_trips_and_leaves_no_temporary(tmp_path):
    from natspec_corpus.sigma_build import _read_shard, _write_shard
    p = tmp_path / "s.json"
    _write_shard(p, [{"id": "F1", "name": "f"}])
    assert _read_shard(p) == [{"id": "F1", "name": "f"}]
    assert not list(tmp_path.glob("*.tmp"))


def test_shard_write_replaces_the_previous_one(tmp_path):
    from natspec_corpus.sigma_build import _read_shard, _write_shard
    p = tmp_path / "s.json"
    _write_shard(p, [{"id": "F1"}])
    _write_shard(p, [{"id": "F2"}])
    assert _read_shard(p) == [{"id": "F2"}]
