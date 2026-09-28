"""Dependency-only stand-ins: resolution, reading, and what must not change.

The risk this module carries is not that it fails loudly — a bad substitution
makes solc error and the file stays uncovered, exactly as today. The risk is
that it quietly widens what counts as corpus data. So alongside the resolution
mechanics, these tests pin the boundary: vendored files are dependencies and
nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from natspec_corpus.closure import resolve_import
from natspec_corpus.compile import unit_for
from natspec_corpus.vendor import PREFIX, Vendor, sources_from_vendor


def tree(tmp_path: Path, prefixes, files) -> Vendor:
    root = tmp_path / "vendor"
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (root / "index.json").write_text(
        json.dumps({"prefixes": prefixes}), encoding="utf-8")
    return Vendor.load(root, {"proj/A.sol", "proj/lib/B.sol"})


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def test_no_vendor_tree_is_a_normal_state(tmp_path):
    assert Vendor.load(tmp_path / "nothing") is None


def test_a_prefix_may_be_a_bare_string_or_a_list(tmp_path):
    v = tree(tmp_path, {"@oz/": "a/", "@up/": ["b/", "c/"]}, {})
    assert v.prefixes == {"@oz/": ["a/"], "@up/": ["b/", "c/"]}


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def test_a_relative_import_behaves_exactly_as_before(tmp_path):
    v = tree(tmp_path, {}, {})
    for frm, spec in (("proj/A.sol", "./lib/B.sol"),
                      ("proj/x/A.sol", "../lib/B.sol"),
                      ("A.sol", "./B.sol")):
        assert v.resolve(frm, spec) == resolve_import(frm, spec)


def test_a_non_relative_import_naming_a_corpus_file_now_resolves(tmp_path):
    """These are real and were being written off. `element-finance/contracts/
    .../SafeMath.sol` is in the corpus and was treated as an npm package purely
    because it does not begin with a dot."""
    v = tree(tmp_path, {}, {})
    assert v.resolve("proj/A.sol", "proj/lib/B.sol") == "proj/lib/B.sol"
    assert v.resolve("proj/A.sol", "proj/absent.sol") is None


def test_a_package_import_keeps_its_specifier_as_the_source_key(tmp_path):
    """solc resolves imports by matching the literal import string against
    its own `sources` keys. Key a vendored file as `_vendor/...` and solc
    answers `Source "@oz/..." not found` — which is exactly how the first
    version of this failed, 51 compile errors deep."""
    v = tree(tmp_path, {"@oz/contracts/": ["oz4/"]},
             {"oz4/token/IERC20.sol": "interface IERC20 {}"})
    spec = "@oz/contracts/token/IERC20.sol"
    assert v.resolve("proj/A.sol", spec) == spec


def test_a_package_file_the_tree_does_not_hold_stays_unresolved(tmp_path):
    v = tree(tmp_path, {"@oz/contracts/": ["oz4/"]},
             {"oz4/token/IERC20.sol": "interface IERC20 {}"})
    assert v.resolve("proj/A.sol", "@oz/contracts/absent.sol") is None


def test_the_longest_prefix_wins(tmp_path):
    """`@oz/contracts-upgradeable/` must not be swallowed by `@oz/contracts/`,
    which is a prefix of it as a string but a different package."""
    v = tree(tmp_path, {"@oz/contracts/": ["v4/"],
                        "@oz/contracts-upgradeable/": ["up/"]},
             {"v4/x.sol": "// v4", "up/x.sol": "// upgradeable"})
    assert v.resolve("A.sol", "@oz/contracts-upgradeable/x.sol") == \
        "@oz/contracts-upgradeable/x.sol"
    assert v.read("@oz/contracts-upgradeable/x.sol") == "// upgradeable"
    assert v.read("@oz/contracts/x.sol") == "// v4"


def test_trees_are_tried_in_order_so_v3_and_v4_can_coexist(tmp_path):
    """OpenZeppelin moved files between major releases — `math/SafeMath.sol`
    in v3 is `utils/math/SafeMath.sol` in v4 — and this corpus spans both."""
    v = tree(tmp_path, {"@oz/contracts/": ["v4/", "v3/"]},
             {"v4/utils/math/SafeMath.sol": "// v4",
              "v3/math/SafeMath.sol": "// v3"})
    assert v.read("@oz/contracts/utils/math/SafeMath.sol") == "// v4"
    assert v.read("@oz/contracts/math/SafeMath.sol") == "// v3"


def test_resolution_records_which_specs_were_substituted(tmp_path):
    v = tree(tmp_path, {"@oz/contracts/": ["oz4/"]},
             {"oz4/token/IERC20.sol": "interface IERC20 {}"})
    v.resolve("A.sol", "@oz/contracts/token/IERC20.sol")
    v.resolve("A.sol", "./B.sol")
    assert v.report()["specs_remapped"] == ["@oz/contracts/token/IERC20.sol"]


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def test_read_serves_only_what_a_prefix_covers(tmp_path):
    v = tree(tmp_path, {"@oz/": ["oz/"]}, {"oz/A.sol": "contract A {}"})
    assert v.read("@oz/A.sol") == "contract A {}"
    assert v.read("proj/A.sol") is None
    assert v.read("@oz/absent.sol") is None


def test_read_does_not_translate_newlines(tmp_path):
    """`Path.read_text` opens in universal-newline mode, turns \\r\\n into \\n
    and shortens the string — and every byte offset downstream is then wrong.
    This project has been bitten by exactly that before."""
    root = tmp_path / "vendor"
    (root / "oz").mkdir(parents=True)
    (root / "oz" / "A.sol").write_bytes(b"contract A {\r\n}\r\n")
    (root / "index.json").write_text('{"prefixes":{"@oz/":["oz/"]}}',
                                     encoding="utf-8")
    v = Vendor.load(root)
    assert v.read("@oz/A.sol") == "contract A {\r\n}\r\n"


def test_wrap_serves_the_corpus_first(tmp_path):
    v = tree(tmp_path, {"@oz/": ["oz/"]}, {"oz/A.sol": "vendored"})
    read = v.wrap(lambda rel: "from corpus" if rel == "proj/A.sol" else None)
    assert read("proj/A.sol") == "from corpus"
    assert read("@oz/A.sol") == "vendored"
    assert read("nowhere.sol") is None


# --------------------------------------------------------------------------
# the whole unit
# --------------------------------------------------------------------------

def test_a_unit_completes_through_the_vendor_tree(tmp_path):
    v = tree(tmp_path, {"@oz/contracts/": ["oz4/"]},
             {"oz4/token/IERC20.sol":
                  'import "../utils/Context.sol";\ninterface IERC20 {}',
              "oz4/utils/Context.sol": "abstract contract Context {}"})
    sources = {"proj/A.sol":
               'import "@oz/contracts/token/IERC20.sol";\ncontract A {}'}
    unit = unit_for("proj/A.sol", v.wrap(sources.get), v.resolve)
    assert unit.unresolved == [], "the package and its own imports resolved"
    # The keys are the specifiers solc will look for, and the vendored file's
    # own `../../utils/Context.sol` landed on the right package path.
    assert set(unit.sources) == {
        "proj/A.sol", "@oz/contracts/token/IERC20.sol",
        "@oz/contracts/utils/Context.sol"}
    assert sources_from_vendor(unit, v) == [
        "@oz/contracts/token/IERC20.sol", "@oz/contracts/utils/Context.sol"]


def test_without_a_vendor_the_same_unit_is_incomplete(tmp_path):
    sources = {"proj/A.sol":
               'import "@oz/contracts/token/IERC20.sol";\ncontract A {}'}
    unit = unit_for("proj/A.sol", sources.get)
    assert unit.unresolved == ["@oz/contracts/token/IERC20.sol"]


def test_unit_for_still_defaults_to_the_plain_resolver():
    """The default path must be untouched, or every existing fact table's
    shard key changes and the whole Σ(f) build re-runs for no reason."""
    import inspect
    assert inspect.signature(unit_for).parameters["resolve"].default is None
    sources = {"a.sol": 'import "./b.sol";', "b.sol": "// b"}
    assert unit_for("a.sol", sources.get).unresolved == []


def test_vendored_sources_are_identifiable_in_any_unit(tmp_path):
    """Provenance: a fact table built against stand-ins must be distinguishable
    from one built against a real dependency closure."""
    v = tree(tmp_path, {"@oz/": ["oz/"]}, {"oz/A.sol": "// a"})
    unit = unit_for("a.sol", {"a.sol": "// a"}.get)
    assert sources_from_vendor(unit, v) == []


# --------------------------------------------------------------------------
# picking a tree of the right Solidity version
#
# The first attempt selected donor trees by file count alone, picked
# OpenZeppelin 4.2.0 (^0.8.0) for every project, and mixed it into 0.6 and 0.7
# codebases. A compilation unit must satisfy every pragma in it with one
# compiler, so the result was 12 pragma errors, 35 syntax errors and not one
# new fact table.
# --------------------------------------------------------------------------

def versioned(tmp_path):
    root = tmp_path / "vendor"
    for name, pragma in (("v8", "^0.8.0"), ("v6", "^0.6.0")):
        d = root / name
        d.mkdir(parents=True)
        (d / "IERC20.sol").write_text(
            f"pragma solidity {pragma};\ninterface IERC20 {{}}", encoding="utf-8")
    root.joinpath("index.json").write_text(json.dumps(
        {"prefixes": {"@oz/": ["v8/", "v6/"]},
         "versions": {"v8/": [[0, 8]], "v6/": [[0, 6]]}}), encoding="utf-8")
    return Vendor.load(root)


def test_a_project_gets_a_tree_of_its_own_solidity_version(tmp_path):
    v = versioned(tmp_path)
    assert "0.6" in v.for_pragma("pragma solidity 0.6.12;").read("@oz/IERC20.sol")
    assert "0.8" in v.for_pragma("pragma solidity ^0.8.4;").read("@oz/IERC20.sol")


def test_with_no_pragma_the_declared_order_stands(tmp_path):
    v = versioned(tmp_path)
    assert "0.8" in v.for_pragma("").read("@oz/IERC20.sol")


def test_an_unmatched_version_falls_back_rather_than_failing(tmp_path):
    """A tree of unknown or non-matching version is still tried — it may work,
    and no table at all is the alternative."""
    v = versioned(tmp_path)
    assert v.for_pragma("pragma solidity 0.4.24;").read("@oz/IERC20.sol")


# --------------------------------------------------------------------------
# generation must see the same dependencies Σ(f) was built against
#
# The two are built in different places — `sigma_build.build` and
# `runner.load_contexts` — and if only one of them knows about the stand-ins
# they disagree: a function can have a fact table built with vendored
# dependencies and still arrive at the critic with `unit=None`, so the
# compiler half sits out on exactly the functions the vendoring was meant to
# reach. Silent, and it would look like the vendoring simply did not work.
# --------------------------------------------------------------------------

def test_both_unit_builders_consult_the_vendor_tree():
    import inspect
    from natspec_corpus import runner, sigma_build
    for mod in (runner.load_contexts, sigma_build.build):
        src = inspect.getsource(mod)
        assert "Vendor" in src or "_Vendor" in src, \
            f"{mod.__name__} builds units without the vendor stand-ins"
        assert "unit_for(rel, read, " in src, \
            f"{mod.__name__} calls unit_for without passing a resolver"


def test_a_resolver_is_threaded_all_the_way_into_the_unit(tmp_path):
    """End to end at the level that matters: a package import becomes a real
    source in the unit, which is what gives the deterministic critic something
    to compile."""
    v = tree(tmp_path, {"@oz/": ["oz/"]}, {"oz/IERC20.sol": "interface IERC20 {}"})
    src = {"proj/A.sol": 'import "@oz/IERC20.sol";\ncontract A {}'}
    without = unit_for("proj/A.sol", src.get)
    with_ = unit_for("proj/A.sol", v.wrap(src.get), v.resolve)
    assert without.unresolved and not without.complete
    assert not with_.unresolved and with_.complete
