"""Turning a tokenised SmartDoc line and a real source function into keys
that can be compared.

Two keys per function.

`exact_key` is the sha1 of the text with **every whitespace character
removed**. SmartDoc's tokenisation only ever inserts, removes or normalises
whitespace — it does not rename identifiers, reformat numbers or drop
punctuation — so a function that survived the tokeniser unchanged in every
other respect has the same exact key as its original. This is what carries
the bulk of the matching, and it is exact: no threshold, no tuning.

`bucket_key` is `(name, parameter count)`, parsed out of the tokenised
header. It is deliberately coarse, because its only job is to keep the fuzzy
pass from comparing every function against every other one.

The fuzzy pass exists for the minority that *were* changed: a corpus copy
with a different pragma-era spelling, a trailing semicolon the tokeniser ate,
a `uint` written `uint256`. It scores Jaccard over token trigrams and demands
both a high score and a clear margin over the runner-up, because a confident
wrong match is worse here than no match at all — it would attach the wrong
fact table to a published reference and quietly corrupt the comparison.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Optional, Sequence, Set, Tuple

_WS = re.compile(r"\s+")

# Solidity's lexical atoms, longest-first so `>>=` beats `>>` beats `>`.
_TOKEN = re.compile(
    r"""
      (?: unicode | hex ) ? " (?: \\. | [^"\\] )* "   # string, with prefix
    | (?: unicode | hex ) ? ' (?: \\. | [^'\\] )* '
    | 0[xX][0-9a-fA-F_]+
    | \d[\d_]* (?: \.[\d_]+ )? (?: [eE][+-]?\d+ )?
    | [A-Za-z_$][A-Za-z0-9_$]*
    | >>>= | <<= | >>= | \*\*= | \|\| | && | == | != | <= | >= | << | >>
    | \+\+ | -- | \+= | -= | \*= | /= | %= | \|= | &= | \^= | =>
    | [{}()\[\].,;:?~!+\-*/%<>=&|^]
    """,
    re.X,
)


def squeeze(text: str) -> str:
    """Every whitespace character removed."""
    return _WS.sub("", text)


def exact_key(text: str) -> str:
    return hashlib.sha1(squeeze(text).encode("utf-8", "replace")).hexdigest()


def tokenise(text: str) -> List[str]:
    """Solidity source -> tokens, in source order.

    Applied to an already-tokenised SmartDoc line this is close to the
    identity, which is the point: one function turns both sides of a
    comparison into the same alphabet.
    """
    return _TOKEN.findall(text)


def detokenise(line: str) -> str:
    """A tokenised line back to something a Solidity parser will accept.

    Not pretty-printing — just collapsing the inserted spaces so the result
    is syntactically the same function. Only used for diagnostics and for the
    header parse; matching never relies on it.
    """
    toks = tokenise(line)
    out: List[str] = []
    no_space_before = set(")]},;.")
    no_space_after = set("([.")
    for t in toks:
        if out and (t[0] in no_space_before or out[-1][-1] in no_space_after):
            out.append(t)
        else:
            out.append(" " + t if out else t)
    return "".join(out)


# --------------------------------------------------------------------------
# the coarse bucket
# --------------------------------------------------------------------------

def _balanced(toks: Sequence[str], open_at: int) -> int:
    """Index of the `)` matching the `(` at `open_at`, or -1."""
    depth = 0
    for i in range(open_at, len(toks)):
        if toks[i] == "(":
            depth += 1
        elif toks[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _count_params(inner: Sequence[str]) -> int:
    """Top-level comma count + 1, or 0 for an empty list.

    Top-level matters: `mapping(uint => uint) storage self, uint x` has one
    comma inside the mapping and one between the parameters.
    """
    if not [t for t in inner if t.strip()]:
        return 0
    depth, n = 0, 1
    for t in inner:
        if t in "([{":
            depth += 1
        elif t in ")]}":
            depth -= 1
        elif t == "," and depth == 0:
            n += 1
    return n


def header(text: str) -> Optional[Tuple[str, int]]:
    """`(function name, parameter count)` for a function body, else None.

    Returns None for a constructor, a modifier, a fallback, or anything that
    does not begin with the `function` keyword — SmartDoc's test set is all
    named functions, so anything else is not a candidate and is cheaper to
    drop here than to compare later.
    """
    toks = tokenise(text)
    try:
        kw = toks.index("function")
    except ValueError:
        return None
    if kw + 2 >= len(toks):
        return None
    name = toks[kw + 1]
    if not re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", name) or toks[kw + 2] != "(":
        return None
    close = _balanced(toks, kw + 2)
    if close < 0:
        return None
    return name, _count_params(toks[kw + 3:close])


def bucket_key(text: str) -> Optional[str]:
    h = header(text)
    return None if h is None else f"{h[0]}/{h[1]}"


# --------------------------------------------------------------------------
# the fuzzy pass
# --------------------------------------------------------------------------

def shingles(text: str, n: int = 3) -> Set[Tuple[str, ...]]:
    toks = tokenise(text)
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def best_match(query: str, candidates: Iterable[Tuple[int, str]], *,
               threshold: float = 0.85, margin: float = 0.05):
    """The one candidate that is both similar enough and clearly the best.

    Returns `(candidate_id, score)` or None. Two rules, both necessary:
    the winner must clear `threshold`, and it must beat the runner-up by
    `margin`. Near-duplicate functions are extremely common in deployed
    Solidity — half of Etherscan is the same ERC-20 — so a bucket routinely
    contains several candidates that all score 0.9. Picking one of those at
    random would attach a real fact table to the wrong reference, which is
    the single failure this whole exercise cannot afford.
    """
    q = shingles(query)
    scored = sorted(((jaccard(q, shingles(t)), cid) for cid, t in candidates),
                    reverse=True)
    if not scored or scored[0][0] < threshold:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
        return None
    return scored[0][1], scored[0][0]
