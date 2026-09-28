"""The Σ(f) join invariant: what it tolerates and what it must still catch.

Separate from test_sigma.py because that module is skipped where no compiler is
installed, and none of this needs one — it is pure offset arithmetic, and it is
the guard standing between the corpus and fact tables attached to the wrong
functions.
"""
from __future__ import annotations

from pathlib import Path


# --------------------------------------------------------------------------
# a fact table that joins to no declaration
#
# Solidity permits a modifier with no parameter list — `modifier foo { ... }` —
# and `find_decls` requires one, so Slither reports six modifiers in
# opyn-gamma's Controller.sol where the extractor finds two. That surfaced only
# once vendored dependencies let the file compile for the first time, and it
# aborted the whole build. The table is inert — the join is by exact offset, so
# with no matching declaration it never reaches a pair — but the check is the
# only guard against genuine offset drift, so tolerance has to be bounded.
# --------------------------------------------------------------------------

import pytest as _pytest

from natspec_corpus import checks as _checks
from natspec_corpus.errors import InvariantError as _InvariantError


def _corpus(tmp_path, source: str):
    root = tmp_path / "corpus"
    (root / "contracts" / "p").mkdir(parents=True)
    (root / "contracts" / "p" / "C.sol").write_text(source, encoding="utf-8")
    return root


_SRC = """pragma solidity 0.6.12;
contract C {
    modifier notPaused { _; }
    function add(uint256 a) public pure returns (uint256) { return a; }
}
"""


def _table(char_start, function="x"):
    return {"file": "p/C.sol", "function": function, "char_start": char_start}


def test_a_parser_gap_is_tolerated_and_reported(tmp_path):
    root = _corpus(tmp_path, _SRC)
    at = _SRC.index("modifier notPaused")
    good = _SRC.index("function add")
    tables = [_table(good, "add")] + [_table(at, "notPaused")] * 1
    # 1 miss in a large table set: under the rate, and on a declaration.
    tables += [_table(good, "add") for _ in range(40)]
    misses = _checks.sigma_tables_join_to_declarations(root, tables)
    assert len(misses) == 1
    assert misses[0]["function"] == "notPaused"
    assert misses[0]["on_a_declaration"] is True


def test_an_offset_that_lands_mid_token_is_still_a_failure(tmp_path):
    """The guard that must not be lost: a drift lands inside a token, and one
    such miss is enough to stop the build."""
    root = _corpus(tmp_path, _SRC)
    good = _SRC.index("function add")
    with _pytest.raises(_InvariantError, match="offset drift"):
        _checks.sigma_tables_join_to_declarations(
            root, [_table(good, "add")] * 40 + [_table(good + 3, "drifted")])


def test_too_many_misses_is_a_failure_even_on_declarations(tmp_path):
    """A parser gap affects a handful of one shape. A quarter of the corpus
    failing to join is something else, whatever it lands on."""
    root = _corpus(tmp_path, _SRC)
    at = _SRC.index("modifier notPaused")
    good = _SRC.index("function add")
    with _pytest.raises(_InvariantError, match="offset drift"):
        _checks.sigma_tables_join_to_declarations(
            root, [_table(good, "add")] * 6 + [_table(at, "notPaused")] * 4)


def test_everything_joining_returns_no_misses(tmp_path):
    root = _corpus(tmp_path, _SRC)
    good = _SRC.index("function add")
    assert _checks.sigma_tables_join_to_declarations(
        root, [_table(good, "add")]) == []
