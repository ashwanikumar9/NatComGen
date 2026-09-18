"""Regression tests. Every case here is a bug that shipped in v1 or a Solidity
construct that looks like another one.
"""
import pytest

from natspec_corpus.errors import LexError, ParseError
from natspec_corpus.extract import build_file, doc_units
from natspec_corpus.masking import CODE_ONLY, Kind, code_view, mask, scan
from natspec_corpus.natspec import parse as parse_doc
from natspec_corpus import solidity as S
from natspec_corpus.score import judge


def model(src):
    return build_file("t.sol", src)


def kinds(src):
    return [(sp.kind.value, sp.text(src)) for sp in scan(src)]


# -- lexer -----------------------------------------------------------------

def test_apostrophe_in_comment_does_not_open_a_string():
    """The v1 bug: `don't` opened a phantom string that swallowed the `*/`."""
    src = ("/** @notice We don't check this.\n * @param x a value\n */\n"
           "function f(uint x) public {}\n"
           "/** @notice Second block. */\nfunction g() public {}\n")
    spans = scan(src)
    docs = [sp for sp in spans if sp.kind is Kind.DOC_BLOCK]
    assert len(docs) == 2
    m = model(src)
    assert {a.decl.name for a in m.attachments} == {"f", "g"}
    f = next(a for a in m.attachments if a.decl.name == "f")
    assert list(f.doc.params) == ["x"]


def test_slashes_inside_a_string_are_not_a_comment():
    src = 'function f() public { s = "// not a comment"; }'
    assert not any(sp.kind is Kind.LINE_COMMENT for sp in scan(src))


def test_block_comment_markers_inside_a_string():
    src = 'function f() public { s = "/* not a comment */"; }'
    assert not any(sp.kind is Kind.BLOCK_COMMENT for sp in scan(src))


def test_four_slashes_is_not_natspec():
    assert scan("//// divider\n")[0].kind is Kind.LINE_COMMENT
    assert scan("/// doc\n")[0].kind is Kind.DOC_LINE


def test_empty_block_comment_is_not_a_doc_block():
    assert scan("/**/\n")[0].kind is Kind.BLOCK_COMMENT
    assert scan("/** doc */\n")[0].kind is Kind.DOC_BLOCK


def test_prefixed_string_literals():
    for src in ['x = unicode"héllo";', 'x = hex"deadbeef";']:
        sp = next(s for s in scan(src) if s.kind is Kind.STRING)
        assert sp.text(src).startswith(("unicode", "hex"))


def test_escaped_quote_does_not_end_the_string():
    src = r'x = "a \" b"; // after'
    assert any(sp.kind is Kind.LINE_COMMENT for sp in scan(src))


def test_unterminated_constructs_raise_in_strict_mode():
    with pytest.raises(LexError):
        scan('x = "oops')
    with pytest.raises(LexError):
        scan("/* oops")
    assert scan('x = "oops', strict=False)


def test_masking_preserves_offsets_and_newlines():
    src = "a;\n// c\nb;\n"
    out = mask(src, scan(src), CODE_ONLY)
    assert len(out) == len(src)
    assert out.count("\n") == src.count("\n")
    assert out.index("b;") == src.index("b;")


# -- structure -------------------------------------------------------------

def test_mapping_parameter_is_not_truncated():
    """v1's non-greedy `\\(.*?\\)` cut this at the first `)`."""
    src = ("library L {\n function f(mapping(int16 => uint256) storage self,"
           " int16 wordPos) internal {}\n}")
    d = model(src).decls[0]
    assert [p.name for p in d.params] == ["self", "wordPos"]
    assert d.params[0].type == "mapping(int16 => uint256) storage"


def test_unnamed_and_named_parameters():
    src = "contract C { function f(uint256, address to) external {} }"
    d = model(src).decls[0]
    assert [p.name for p in d.params] == [None, "to"]


def test_named_returns_are_counted():
    src = ("contract C { function f() external view "
           "returns (uint256 a, uint256 b) {} }")
    d = model(src).decls[0]
    assert [p.name for p in d.returns] == ["a", "b"]


def test_function_type_parameter():
    src = ("contract C { function f(function (uint) external returns (uint) cb)"
           " internal {} }")
    d = model(src).decls[0]
    assert len(d.params) == 1 and d.params[0].name == "cb"


def test_multiline_parameter_list_with_interleaved_comments():
    src = ("contract C {\n function f(\n  bytes32, // poolId\n"
           "  address recipient,\n  uint256[] calldata bal\n ) external {}\n}")
    d = model(src).decls[0]
    assert [p.name for p in d.params] == [None, "recipient", "bal"]


def test_container_and_bases():
    src = "abstract contract C is A, B(1) { function f() public {} }"
    c = model(src).containers[0]
    assert (c.kind, c.name, c.abstract, c.bases) == ("contract", "C", True,
                                                     ["A", "B"])
    assert model(src).decls[0].container == "C"


def test_visibility_mutability_virtual_override():
    src = ("contract C { function f() public view virtual override "
           "returns (uint) {} }")
    d = model(src).decls[0]
    assert (d.visibility, d.mutability, d.is_virtual, d.overrides) == \
           ("public", "view", True, True)


def test_signature_ignores_data_location():
    src = "contract C { function f(bytes calldata a, uint[] memory b) external {} }"
    assert model(src).decls[0].sig == "f(bytes,uint[])"


def test_constructor_fallback_receive_modifier_event_error():
    src = ("contract C { constructor() {} fallback() external {} "
           "receive() external payable {} modifier m() { _; } "
           "event E(uint a); error Bad(uint a); }")
    assert {d.kind for d in model(src).decls} == {
        "constructor", "fallback", "receive", "modifier", "event", "error"}


def test_unbalanced_parens_raise():
    with pytest.raises(ParseError):
        S.match_delim("f(a, b", 1)


# -- attachment ------------------------------------------------------------

def test_doc_attaches_only_to_the_next_declaration():
    src = ("/// @notice A\nfunction a() public {}\n"
           "function b() public {}\n")
    m = model(src)
    assert [(a.decl.name, a.doc.notice) for a in m.attachments] == [("a", "A")]


def test_doc_before_a_contract_is_not_a_function_doc():
    src = "/// @title T\ncontract C { function f() public {} }"
    m = model(src)
    assert m.attachments == []
    assert "C" in m.contract_docs


def test_orphaned_doc_is_reported_not_guessed():
    src = "contract C { /// @notice dangling\n uint public x; }"
    m = model(src)
    assert m.attachments == [] and len(m.orphans) == 1


def test_consecutive_doc_lines_merge_into_one_unit():
    src = "/// @notice A\n/// @dev B\nfunction f() public {}"
    assert len(doc_units(src, scan(src))) == 1


def test_plain_comment_does_not_split_a_doc_run():
    """Element Finance mixes `///` and `//` inside one comment. solc ignores
    the `//` lines; they must not split the run into three docs."""
    src = ("/// @notice N\n// @param ignored solc never sees this\n"
           "/// @param x real\nfunction f(uint x) public {}")
    m = model(src)
    assert len(m.attachments) == 1
    doc = m.attachments[0].doc
    assert list(doc.params) == ["x"]
    assert "ignored" not in doc.raw


def test_blank_line_between_doc_and_declaration_still_attaches():
    src = "/// @notice A\n\n\nfunction f() public {}"
    assert model(src).attachments[0].decl.name == "f"


# -- natspec ---------------------------------------------------------------

def test_untagged_prose_is_the_notice():
    d = parse_doc("/// Transfers tokens to a recipient.\n")
    assert d.notice == "Transfers tokens to a recipient."


def test_block_markers_are_stripped():
    d = parse_doc("/**\n * @notice A\n * @dev B\n */")
    assert (d.notice, d.dev) == ("A", "B")


def test_continuation_lines_join_the_current_tag():
    d = parse_doc("/// @param x the first\n///          and second line\n")
    assert d.params["x"] == "the first and second line"


def test_at_inside_prose_does_not_open_a_tag():
    d = parse_doc("/// @notice Contact a@b.com for more\n")
    assert d.notice.endswith("for more") and len(d.of("notice")) == 1


def test_inheritdoc_target():
    assert parse_doc("/// @inheritdoc IFoo\n").inheritdoc == "IFoo"


def test_custom_tags_survive():
    d = parse_doc("/// @custom:security contact x\n")
    assert d.of("custom:security")[0].text == "contact x"


# -- scoring ---------------------------------------------------------------

def _judge(src):
    m = model(src)
    return judge(m.attachments[0])


def test_complete_doc_is_verified():
    v = _judge("contract C {\n/// @notice Adds two numbers together.\n"
               "/// @param a first\n/// @param b second\n"
               "/// @return the sum\nfunction add(uint a, uint b) public "
               "returns (uint) {}\n}")
    assert v.verified and not v.defects


def test_missing_param_is_labelled():
    v = _judge("contract C {\n/// @notice Adds two numbers together.\n"
               "/// @param a first\nfunction add(uint a, uint b) public {}\n}")
    assert not v.verified and "param_missing:b" in v.defects


def test_unknown_param_is_labelled():
    v = _judge("contract C {\n/// @notice Adds two numbers together.\n"
               "/// @param a first\n/// @param z ghost\n"
               "function add(uint a) public {}\n}")
    assert "param_unknown:z" in v.defects


def test_return_count_must_match():
    v = _judge("contract C {\n/// @notice Returns a pair of values.\n"
               "/// @return the first\nfunction f() public "
               "returns (uint, uint) {}\n}")
    assert "return_missing" in v.defects


def test_unnamed_parameters_need_no_param_tag():
    v = _judge("contract C {\n/// @notice Accepts an ignored argument.\n"
               "function f(uint256) public {}\n}")
    assert v.verified


def test_placeholder_and_short_text_are_labelled():
    assert "placeholder" in _judge(
        "contract C {\n/// @notice TODO write this later\n"
        "function f() public {}\n}").defects
    assert "text_short" in _judge(
        "contract C {\n/// @notice Adds.\nfunction f() public {}\n}").defects


def test_name_echo_is_labelled():
    v = _judge("contract C {\n/// @notice Get total supply\n"
               "function getTotalSupply() public {}\n}")
    assert "name_echo" in v.defects


# -- regressions found while documenting the output ------------------------

def test_adjacent_doc_blocks_do_not_merge():
    """88mph puts a section banner right above a function's own doc block."""
    src = ("contract C {\n/**\n   Public action functions\n */\n"
           "/**\n @notice Creates a deposit for the caller.\n"
           " @param amount how much\n */\n"
           "function deposit(uint256 amount) external {}\n}")
    m = model(src)
    assert len(m.attachments) == 1
    doc = m.attachments[0].doc
    assert "Public action functions" not in doc.notice
    assert doc.notice.startswith("Creates a deposit")
    assert len(m.orphans) == 1          # the banner, reported not merged


def test_nearest_comment_wins():
    src = ("contract C {\n/// @notice Banner text here.\n"
           "/**\n @notice The real one.\n */\n"
           "function f() external {}\n}")
    m = model(src)
    assert len(m.attachments) == 1
    assert m.attachments[0].doc.notice == "The real one."


def test_return_name_is_not_duplicated_into_the_text():
    from natspec_corpus.build import _returns
    m = model("contract C {\n/// @notice Does a thing with two results.\n"
              "/// @return depositID The ID of the deposit\n"
              "/// @return interestAmount The interest\n"
              "function f() external returns (uint256 depositID,"
              " uint256 interestAmount) {}\n}")
    a = m.attachments[0]
    r = _returns(a.doc, a.decl)
    assert r[0] == {"name": "depositID", "text": "The ID of the deposit"}
    assert r[1] == {"name": "interestAmount", "text": "The interest"}


def test_unnamed_return_keeps_its_whole_text():
    from natspec_corpus.build import _returns
    m = model("contract C {\n/// @notice Adds two numbers together.\n"
              "/// @return the sum of the inputs\n"
              "function f() external returns (uint256) {}\n}")
    a = m.attachments[0]
    assert _returns(a.doc, a.decl) == [{"name": None,
                                        "text": "the sum of the inputs"}]


def test_a_qualified_type_is_not_split_into_type_and_name():
    """`DataTypes.PubType` as an unnamed return was parsed as type
    `DataTypes.` plus name `PubType`, which invents a parameter that does not
    exist and truncates the type."""
    assert S.parse_param("DataTypes.PubType") == S.Param(
        type="DataTypes.PubType", name=None, raw="DataTypes.PubType")
    assert S.parse_param("DataTypes.PubType kind").name == "kind"


def test_qualified_return_type_has_no_name():
    src = ("contract C { function f() external view "
           "returns (DataTypes.PubType) {} }")
    d = model(src).decls[0]
    assert [p.name for p in d.returns] == [None]
    assert d.returns[0].type == "DataTypes.PubType"
