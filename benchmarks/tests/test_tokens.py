"""The matcher's alphabet. Every bug in here produces a *wrong* match rather
than a missing one, which is the failure mode this benchmark cannot survive.
"""
import pytest

from benchmarks.smartdoc import tokens as T

SRC = """function updateDates(uint8 _tierId, uint256 _start) public onlyOwner {
    if (_start != 0) { tiers[_tierId].startDate = _start; }
}"""
TOKENISED = ("function updateDates ( uint8 _tierId , uint256 _start ) public "
             "onlyOwner { if ( _start != 0 ) { tiers [ _tierId ] . startDate "
             "= _start ; } }")


def test_exact_key_ignores_whitespace_only():
    assert T.exact_key(SRC) == T.exact_key(TOKENISED)


def test_exact_key_is_not_blind_to_content():
    assert T.exact_key(SRC) != T.exact_key(SRC.replace("_start", "_stop"))


def test_squeeze_removes_newlines_and_tabs():
    assert T.squeeze(" a\t b\r\n c ") == "abc"


def test_header_counts_top_level_parameters_only():
    # The comma inside `mapping(...)` belongs to the type, not the list. A
    # naive split on "," reports three parameters and puts the function in a
    # bucket where its real twin will never be compared against it.
    line = ("function set ( mapping ( uint256 => uint256 ) storage self , "
            "uint256 k ) internal { }")
    assert T.header(line) == ("set", 2)


def test_header_empty_parameter_list():
    assert T.header("function f ( ) public { }") == ("f", 0)


def test_header_rejects_non_functions():
    assert T.header("constructor ( uint x ) public { }") is None
    assert T.header("modifier onlyOwner ( ) { _ ; }") is None
    assert T.header("") is None


def test_bucket_key_shape():
    assert T.bucket_key(TOKENISED) == "updateDates/2"


def test_tokenise_keeps_string_literals_whole():
    toks = T.tokenise('require ( ok , "not started yet" ) ;')
    assert '"not started yet"' in toks


def test_tokenise_keeps_multi_character_operators():
    assert ">>=" in T.tokenise("a >>= b ;")
    assert "=>" in T.tokenise("mapping ( uint => uint ) m ;")


def test_detokenise_preserves_the_exact_key():
    assert T.exact_key(T.detokenise(TOKENISED)) == T.exact_key(TOKENISED)


def test_best_match_refuses_a_near_tie():
    """Two candidates within the margin means no answer.

    Deployed Solidity is mostly copies of the same few contracts, so a bucket
    routinely holds several bodies that all score above the threshold.
    Returning the highest of those would attach a real fact table to somebody
    else's reference.
    """
    q = "function f ( uint a ) public { uint b = a + 1 ; return b ; }"
    left = q.replace("+ 1", "+ 2")
    right = q.replace("+ 1", "+ 3")
    # Equidistant from the query, so neither can win by the margin.
    assert T.jaccard(T.shingles(q), T.shingles(left)) == \
        T.jaccard(T.shingles(q), T.shingles(right))
    assert T.best_match(q, [(1, left), (2, right)], threshold=0.5) is None


def test_best_match_accepts_a_clear_winner():
    q = "function f ( uint a ) public { return a + 1 ; }"
    far = "function f ( uint a ) public { while ( a > 0 ) { a -- ; } }"
    hit = T.best_match(q, [(1, q), (2, far)])
    assert hit is not None and hit[0] == 1


def test_best_match_respects_the_threshold():
    q = "function f ( uint a ) public { return a + 1 ; }"
    far = "function f ( uint a ) public { while ( a > 0 ) { a -- ; } }"
    assert T.best_match(q, [(2, far)], threshold=0.99) is None


def test_jaccard_edges():
    assert T.jaccard(set(), set()) == 1.0
    assert T.jaccard({1}, set()) == 0.0
    assert T.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
