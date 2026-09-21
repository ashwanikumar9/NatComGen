"""Stage 3: rendering comments, assembling them into a file, verifying it.

None of this needs a model. Every check here is either a structural property
of the output or the compiler's own opinion of it.
"""
import json
from pathlib import Path

import pytest

from natspec_corpus.assemble import (AssemblyError, document_file, plan,
                                     choose_inheritdoc, inheritdoc_targets)
from natspec_corpus.compile import compile_source, installed_versions, unit_for
from natspec_corpus.emit import (Comment, detect_style, from_doc, from_pair,
                                 render, render_for, tags_in)
from natspec_corpus.extract import build_file
from natspec_corpus.natspec import parse as parse_doc
from natspec_corpus.verify_file import (devdoc_diff, devdoc_equal, devdoc_of,
                                        exposed_members, strip_comments,
                                        strip_doc_comments, verify)

FILE = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

/// @title A small vault
/// @author Someone
contract Vault {
    error Unauthorised(address who);
    event Moved(address indexed to, uint256 amount);

    uint256 public total;

    /// ****************
    /// *****ACTIONS****
    /// ****************

    function deposit(uint256 amount) external returns (uint256 newTotal) {
        require(amount > 0, "zero");
        total += amount;
        newTotal = total;
    }

    /// @notice Reads the total.
    /// @return the running total
    function peek() external view returns (uint256) { return total; }

    function _half(uint256 x) internal pure returns (uint256) { return x / 2; }
}
"""


def decl(src, name):
    return next(d for d in build_file("t.sol", src).decls if d.name == name)


# -- rendering -------------------------------------------------------------

def test_tags_come_out_in_solc_order():
    c = Comment(notice="N", dev="D", params={"b": "second", "a": "first"},
                returns=["r"], title="T", author="A")
    text = render(c, param_order=["a", "b"], return_count=1,
                  return_names=[None])
    assert tags_in(text) == ["title", "author", "notice", "dev", "param",
                             "param", "return"]


def test_params_follow_declaration_order_not_the_models():
    """solc matches @param by name but a reviewer reads them in order; @return
    is matched positionally, so order there is meaning, not style."""
    c = Comment(params={"b": "second", "a": "first"})
    text = render(c, param_order=["a", "b"])
    assert text.index("@param a") < text.index("@param b")


def test_returns_are_positional_and_named_returns_lead_with_their_name():
    c = Comment(returns=["the sum", "the carry"])
    text = render(c, return_count=2, return_names=["sum", "carry"])
    assert "@return sum the sum" in text
    assert "@return carry the carry" in text


def test_a_param_the_declaration_does_not_have_is_dropped():
    """solc rejects the file outright for an undocumented parameter name, so
    emitting one turns a good comment into a broken build."""
    c = Comment(params={"a": "first", "ghost": "invented"})
    text = render(c, param_order=["a"])
    assert "ghost" not in text


def test_inheritdoc_is_exclusive_except_for_dev():
    c = Comment(inheritdoc="IFoo", notice="ignored", dev="local note",
                params={"a": "x"})
    assert tags_in(render(c, param_order=["a"])) == ["inheritdoc", "dev"]


def test_repeated_tags_are_kept_separate():
    """solc concatenates repeated @dev with no separator; merging them and
    re-emitting one tag inserts a space solc never had."""
    text = render(Comment(dev=["first para", "second para"]))
    assert tags_in(text) == ["dev", "dev"]


def test_untagged_prose_is_preserved_as_prose():
    doc = parse_doc("/**\n * SPDX-License-Identifier: X\n * @notice Does it.\n */")
    c = from_doc(doc)
    assert c.preamble.startswith("SPDX")
    text = render(c)
    assert text.splitlines()[0].endswith("SPDX-License-Identifier: X")
    assert tags_in(text) == ["notice"]


def test_style_is_detected_from_the_file():
    assert detect_style("/// @notice x\ncontract C {}") == "line"
    assert detect_style("/**\n * @notice x\n * @dev y\n */\ncontract C {}") == "block"
    assert detect_style("contract C {}") == "line"


def test_block_style_renders_a_well_formed_block():
    text = render(Comment(notice="N"), style="block", indent="    ")
    assert text.startswith("    /**") and text.endswith("    */")


def test_long_text_wraps_and_stays_inside_the_comment():
    c = Comment(notice="word " * 60)
    for line in render(c).splitlines():
        assert line.startswith("/// ") and len(line) <= 101


def test_an_empty_comment_renders_to_nothing():
    assert render(Comment()) == ""


# -- assembly --------------------------------------------------------------

def test_many_comments_land_on_the_right_declarations():
    src = FILE
    m = build_file("t.sol", src)
    by = {d.name: d.header_start for d in m.decls}
    out = document_file(src, {
        by["deposit"]: Comment(notice="Adds to the total.",
                               params={"amount": "how much"},
                               returns=["newTotal the new total"]),
        by["_half"]: Comment(notice="Halves a value.",
                             params={"x": "the value"}, returns=["half"]),
    }, mode="fill_gaps")
    m2 = build_file("t.sol", out)
    got = {a.decl.name: a.doc.notice for a in m2.attachments}
    assert got["deposit"] == "Adds to the total."
    assert got["_half"] == "Halves a value."
    assert got["peek"] == "Reads the total."      # untouched


def test_fill_gaps_leaves_an_existing_comment_alone():
    m = build_file("t.sol", FILE)
    by = {d.name: d.header_start for d in m.decls}
    out = document_file(FILE, {by["peek"]: Comment(notice="Replaced!")},
                        mode="fill_gaps")
    assert "Replaced!" not in out


def test_replace_overwrites_it():
    m = build_file("t.sol", FILE)
    by = {d.name: d.header_start for d in m.decls}
    out = document_file(FILE, {by["peek"]: Comment(notice="Replaced.")},
                        mode="replace")
    assert "Replaced." in out and "Reads the total." not in out


def test_a_decorative_banner_does_not_count_as_documentation():
    """Lens separates sections with a /// banner. solc reads untagged prose as
    @notice, so the banner IS the next function's documentation — and in
    fill-gaps mode it makes an undocumented function look documented."""
    m = build_file("t.sol", FILE)
    by = {d.name: d.header_start for d in m.decls}
    placements = plan(FILE, {by["deposit"]: Comment(notice="Adds to it.")},
                      mode="fill_gaps")
    assert len(placements) == 1


def test_contract_level_comments_are_placed_above_the_contract():
    m = build_file("t.sol", FILE)
    c = next(c for c in m.containers if c.name == "Vault")
    out = document_file(FILE, {c.header_start: Comment(title="New title")},
                        mode="replace")
    assert out.index("@title New title") < out.index("contract Vault")


def test_emission_is_idempotent():
    src = strip_doc_comments(FILE, rel="t.sol")
    m = build_file("t.sol", src)
    by = {d.name: d.header_start for d in m.decls}
    comments = {by["deposit"]: Comment(notice="Adds to the total.",
                                       params={"amount": "how much"})}
    once = document_file(src, comments, mode="replace")
    m2 = build_file("t.sol", once)
    by2 = {d.name: d.header_start for d in m2.decls}
    twice = document_file(once, {by2["deposit"]: comments[by["deposit"]]},
                          mode="replace")
    assert once == twice


def test_emission_changes_nothing_but_comments():
    m = build_file("t.sol", FILE)
    by = {d.name: d.header_start for d in m.decls}
    out = document_file(FILE, {by["deposit"]: Comment(notice="Adds.")},
                        mode="fill_gaps")
    assert strip_comments(FILE) == strip_comments(out)


def test_an_offset_with_no_declaration_is_refused():
    with pytest.raises(AssemblyError, match="no declaration"):
        document_file(FILE, {7: Comment(notice="x")})


def test_inheritdoc_is_chosen_only_when_the_base_is_documented():
    d = decl("contract C is B { function f() public override {} }", "f")
    assert choose_inheritdoc(d, ["B"], {"B": {"f()"}}) == "B"
    assert choose_inheritdoc(d, ["B"], {"B": set()}) is None
    assert choose_inheritdoc(d, [], {}) is None


# -- stripping -------------------------------------------------------------

def test_stripping_docs_keeps_formatting_and_plain_comments():
    src = "contract C {\n    // keep me\n    /// @notice go\n    function f() public {}\n}\n"
    out = strip_doc_comments(src, rel="t.sol")
    assert "// keep me" in out and "@notice" not in out
    assert "    function f() public {}" in out


def test_stripping_keeps_documentation_it_cannot_place():
    """NatSpec on a public state variable produces a getter's notice in
    devdoc, and the extractor cannot attach it — so removing it would destroy
    documentation emission has no way to write back."""
    src = ("contract C {\n    /// @notice the running total\n"
           "    uint256 public total;\n\n    /// @notice does it\n"
           "    function f() public {}\n}\n")
    out = strip_doc_comments(src, rel="t.sol")
    assert "the running total" in out
    assert "does it" not in out


# -- whole-file verification ----------------------------------------------

def test_exposed_members_excludes_what_solc_never_documents():
    names = dict(exposed_members("t.sol", FILE))
    kinds = {k for k, _ in exposed_members("t.sol", FILE)}
    assert "error" in kinds and "event" in kinds and "function" in kinds
    assert not any(sig.startswith("_half") for _, sig in
                   exposed_members("t.sol", FILE))


def test_devdoc_equal_ignores_whitespace_only_differences():
    a = {"C": {"devdoc": {"methods": {"f()": {"details": "a  b"}}}}}
    b = {"C": {"devdoc": {"methods": {"f()": {"details": "a b"}}}}}
    assert devdoc_equal(a, b) and devdoc_diff(a, b) == []


def test_devdoc_diff_reports_a_contract_level_loss():
    a = {"C": {"devdoc": {"title": "T", "methods": {}}}}
    b = {"C": {"devdoc": {"methods": {}}}}
    assert not devdoc_equal(a, b)
    assert "C.devdoc.title" in devdoc_diff(a, b)


@pytest.mark.skipif(not installed_versions(), reason="no solc")
def test_verify_accepts_a_good_emission_and_reports_completeness():
    m = build_file("t.sol", FILE)
    by = {d.name: d.header_start for d in m.decls}
    out = document_file(FILE, {
        by["deposit"]: Comment(notice="Adds to the total.",
                               params={"amount": "how much"},
                               returns=["newTotal the new total"])},
        mode="fill_gaps")
    rep = verify("t.sol", FILE, out, {"t.sol": FILE})
    assert rep.ok and rep.compiles and rep.code_unchanged
    assert rep.exposed >= 4 and rep.documented >= 2


@pytest.mark.skipif(not installed_versions(), reason="no solc")
def test_the_round_trip_on_a_synthetic_file():
    """E4 in miniature: strip, re-emit from the file's own comments, and
    require solc to read back exactly the same documentation."""
    res = compile_source("t.sol", FILE)
    assert res.ok, res.error
    before = devdoc_of("t.sol", {"t.sol": FILE}, res.version)

    orig = build_file("t.sol", FILE)
    want = {(a.decl.container, a.decl.sig): from_doc(a.doc)
            for a in orig.attachments}
    cdocs = {n: from_doc(d) for n, d in orig.contract_docs.items()}

    stripped = strip_doc_comments(FILE, rel="t.sol")
    m = build_file("t.sol", stripped)
    by = {}
    for d in m.decls:
        by.setdefault((d.container, d.sig), d.header_start)
    cm = {by[k]: v for k, v in want.items() if k in by}
    for c in m.containers:
        if c.name in cdocs:
            cm[c.header_start] = cdocs[c.name]

    out = document_file(stripped, cm, mode="fill_gaps")
    after = devdoc_of("t.sol", {"t.sol": out}, res.version)
    assert devdoc_equal(before, after), devdoc_diff(before, after)
    assert strip_comments(FILE) == strip_comments(out)


@pytest.mark.skipif(not installed_versions(), reason="no solc")
def test_a_preexisting_orphan_comment_is_not_a_placement_failure():
    """NatSpec on a public state variable is an orphan to the extractor and is
    deliberately preserved. Counting it as a misplacement failed three real
    files that emission had handled correctly."""
    src = ("pragma solidity ^0.8.4;\ncontract C {\n"
           "    /// @notice the running total\n    uint256 public total;\n\n"
           "    function f() external {}\n}\n")
    m = build_file("t.sol", src)
    by = {d.name: d.header_start for d in m.decls}
    out = document_file(src, {by["f"]: Comment(notice="Does it.")},
                        mode="fill_gaps")
    rep = verify("t.sol", src, out, {"t.sol": src})
    assert rep.placement_ok and rep.ok, rep.misplaced
    assert "the running total" in out


# -- normalising what a model actually returns -----------------------------
# In a real run 71 of 110 refined comments failed to compile and 77 emitted no
# tags — not because they were wrong, but because the model returned NatSpec
# content with no comment markers and nobody turned it into a comment. These
# cover the shapes that run produced.

def test_bare_tags_become_a_real_comment():
    from natspec_corpus.emit import normalise
    out = normalise("@notice Not implemented, always reverts.\n"
                    "@dev Intended to be overridden.")
    assert out.startswith("/// @notice")
    assert "/// @dev" in out


def test_double_slash_is_promoted_to_natspec():
    """`//` is invisible to solc: it compiles and emits nothing at all."""
    from natspec_corpus.emit import normalise
    out = normalise("// @notice Allocates liquidity.\n// @param id The pool.")
    assert out.count("///") == 2 and "//  " not in out


def test_normalising_keeps_every_parameter_the_model_wrote():
    from natspec_corpus.emit import normalise
    from natspec_corpus.evaluate import fields
    out = normalise("// @notice N\n// @param a The a\n// @param b The b")
    assert sorted(fields(out)) == ["notice", "param:a", "param:b"]


def test_returns_typo_is_corrected_rather_than_dropped():
    from natspec_corpus.emit import normalise
    assert "@return " in normalise("@returns the new total")


def test_tags_solc_rejects_are_dropped():
    """`@throws` has no NatSpec equivalent and makes solc reject the file."""
    from natspec_corpus.emit import normalise
    assert "throws" not in normalise("@notice N\n@throws SomeError() when called")


def test_normalising_is_idempotent():
    from natspec_corpus.emit import normalise
    once = normalise("@notice N\n@param a The a\n@return r")
    assert normalise(once) == once


def test_normalising_invents_nothing():
    """It reformats. A comment with no parameters must not gain any."""
    from natspec_corpus.emit import normalise
    from natspec_corpus.evaluate import fields
    assert sorted(fields(normalise("@notice Just a notice."))) == ["notice"]
    assert normalise("") == ""
