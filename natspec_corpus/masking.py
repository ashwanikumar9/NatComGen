"""Single-pass Solidity lexer producing character spans.

Everything downstream derives from `scan()`. Regexes are never run against raw
source, because raw source contains three things that look like each other:

    "don't"          an apostrophe inside a comment is not a string
    "// not a comment"   a slash-slash inside a string is not a comment
    "/* nope */"         likewise

The v1 corpus builder ran a string-masking regex over raw text and an
apostrophe inside a NatSpec comment opened a phantom string literal, which
blanked the `*/` that closed the block. Two doc blocks then merged and a
function was documented with its neighbour's @param tags. One lexer removes
that whole class of bug.
"""
from enum import Enum
from typing import List, NamedTuple

from .errors import LexError


class Kind(str, Enum):
    CODE = "code"
    LINE_COMMENT = "line_comment"      # // ...
    DOC_LINE = "doc_line"              # /// ...
    BLOCK_COMMENT = "block_comment"    # /* ... */
    DOC_BLOCK = "doc_block"            # /** ... */
    STRING = "string"


class Span(NamedTuple):
    kind: Kind
    start: int
    end: int      # exclusive

    def text(self, src: str) -> str:
        return src[self.start:self.end]


_STR_PREFIXES = ("unicode", "hex")


def _string_prefix_start(src: str, quote_at: int) -> int:
    """Solidity allows unicode"..." and hex"..." — the prefix is part of the
    literal, so a scanner that starts at the quote would leave the prefix in
    CODE. Harmless for us, but we record the true start for correct offsets."""
    for p in _STR_PREFIXES:
        s = quote_at - len(p)
        if s >= 0 and src[s:quote_at] == p:
            if s == 0 or not (src[s - 1].isalnum() or src[s - 1] == "_"):
                return s
    return quote_at


def scan(src: str, *, strict: bool = True) -> List[Span]:
    """Classify every character of `src`. Spans tile the input with no gaps.

    strict=True raises on an unterminated string or block comment; that is a
    file we must not silently half-parse. strict=False closes them at EOF, for
    callers that would rather skip the file than abort a whole build.
    """
    spans: List[Span] = []
    n = len(src)
    i = 0
    code_start = 0

    def flush_code(upto: int) -> None:
        if upto > code_start:
            spans.append(Span(Kind.CODE, code_start, upto))

    while i < n:
        c = src[i]

        # ---- comments
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            flush_code(i)
            is_doc = i + 2 < n and src[i + 2] == "/" and not (
                i + 3 < n and src[i + 3] == "/")      # //// is not NatSpec
            j = src.find("\n", i)
            j = n if j == -1 else j + 1               # keep the newline
            spans.append(Span(Kind.DOC_LINE if is_doc else Kind.LINE_COMMENT, i, j))
            i = code_start = j
            continue

        if c == "/" and i + 1 < n and src[i + 1] == "*":
            flush_code(i)
            is_doc = i + 2 < n and src[i + 2] == "*" and not (
                i + 3 < n and src[i + 3] == "/")      # /**/ is an empty block
            j = src.find("*/", i + 2)
            if j == -1:
                if strict:
                    raise LexError(f"unterminated block comment at offset {i}")
                j = n
            else:
                j += 2
            spans.append(Span(Kind.DOC_BLOCK if is_doc else Kind.BLOCK_COMMENT, i, j))
            i = code_start = j
            continue

        # ---- string literals
        if c in "\"'":
            start = _string_prefix_start(src, i)
            flush_code(start)
            quote = c
            j = i + 1
            closed = False
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == quote:
                    j += 1
                    closed = True
                    break
                if src[j] == "\n":                     # Solidity has no raw
                    break                              # multi-line strings
                j += 1
            if not closed:
                if strict:
                    raise LexError(f"unterminated string at offset {i}")
                j = min(j, n)
            spans.append(Span(Kind.STRING, start, j))
            i = code_start = j
            continue

        i += 1

    flush_code(n)
    return spans


def mask(src: str, spans: List[Span], keep: frozenset) -> str:
    """Return `src` with every span whose kind is not in `keep` blanked.

    Blanking preserves length and newlines, so every offset computed on the
    masked text is valid in the original. That property is relied on
    throughout; do not replace it with deletion.
    """
    out = list(src)
    for sp in spans:
        if sp.kind in keep:
            continue
        for k in range(sp.start, sp.end):
            if out[k] != "\n":
                out[k] = " "
    return "".join(out)


CODE_ONLY = frozenset({Kind.CODE})
DOC_KINDS = frozenset({Kind.DOC_LINE, Kind.DOC_BLOCK})


def code_view(src: str, spans: List[Span]) -> str:
    """Source with comments and string contents blanked. Safe for declaration
    matching: no comment or literal can masquerade as code."""
    out = list(src)
    for sp in spans:
        if sp.kind is Kind.CODE:
            continue
        lo, hi = sp.start, sp.end
        if sp.kind is Kind.STRING:            # keep the quotes, blank inside
            lo, hi = sp.start + 1, max(sp.start + 1, sp.end - 1)
        for k in range(lo, hi):
            if out[k] != "\n":
                out[k] = " "
    return "".join(out)
