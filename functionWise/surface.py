"""Turning a NatSpec comment into the one string a metric sees.

Two decisions live here, and both change the numbers:

**Field order.** A comment is scored as its field values joined together. If
the reference joins them in declaration order and the hypothesis in whatever
order the model emitted, identical content scores as a partial match. So both
sides are joined by the same canonical key -- notice, dev, params by name,
returns by position -- and neither side gets to choose.

**Normalisation.** The existing tables split on whitespace, which means
`amount.` and `amount` are different tokens, `Returns` and `returns` are
different tokens, and every backtick in the corpus costs a match. The
code-summarisation literature lowercases and strips punctuation before
scoring. Both are available; the mode is named in every report, because a
BLEU without its tokenisation stated is not a number anyone can reuse.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from natspec_corpus.evaluate import fields as hypothesis_fields   # noqa: E402
from natspec_corpus.evaluate import reference_fields              # noqa: E402

MODES = ("none", "paper", "identifiers")

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_PIECE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z]+|[0-9]+")

_GROUP = {"notice": 0, "dev": 1, "param": 2, "return": 3}


def _order(key: str):
    head, _, rest = key.partition(":")
    group = _GROUP.get(head, 9)
    return (group, int(rest) if rest.isdigit() else 0, rest)


def join(values: Dict[str, str]) -> str:
    """Field map -> one string, canonical order, both sides alike."""
    return " ".join(values[k].strip()
                    for k in sorted(values, key=_order)
                    if values.get(k) and values[k].strip()).strip()


def split_identifier(word: str) -> List[str]:
    """`depositAmount` -> deposit amount; `MAX_FEE` -> max fee."""
    pieces = _PIECE.findall(word)
    return [p.lower() for p in pieces] or [word.lower()]


def normalize(text: str, mode: str = "paper") -> str:
    """Whitespace-joined tokens under the named scheme."""
    if mode not in MODES:
        raise SystemExit(f"unknown normalisation {mode!r}; "
                         f"choose from {', '.join(MODES)}")
    if mode == "none":
        return (text or "").strip()
    words = _TOKEN.findall(text or "")
    if mode == "identifiers":
        out: List[str] = []
        for w in words:
            out.extend(split_identifier(w))
        return " ".join(out)
    return " ".join(w.lower() for w in words)


# --------------------------------------------------------------------------
# the two views a paper comparison needs
# --------------------------------------------------------------------------

def reference_view(pair: dict, target: str) -> Dict[str, str]:
    got = reference_fields(pair)
    return _restrict(got, target)


def hypothesis_view(natspec: str, target: str) -> Dict[str, str]:
    got = hypothesis_fields(natspec or "")
    return _restrict(got, target)


def _restrict(values: Dict[str, str], target: str) -> Dict[str, str]:
    """`whole` keeps every field; `notice` keeps only @notice.

    `notice` exists because SmartDoc, CCGIR and SCCLLM emit a user notice and
    nothing else. Scoring a four-tag block against their one-line references
    loses precision for a reason that is not quality, and it is the single
    most common way a cross-paper table ends up meaningless.
    """
    if target == "whole":
        return dict(values)
    if target == "notice":
        return {k: v for k, v in values.items() if k == "notice"}
    if target in ("dev", "param", "return"):
        return {k: v for k, v in values.items() if k.split(":", 1)[0] == target}
    raise SystemExit(f"unknown target {target!r}")


def pair_strings(pair: dict, natspec: str, *, target: str = "whole",
                 mode: str = "paper") -> tuple:
    """(reference, hypothesis) ready for a metric."""
    ref = normalize(join(reference_view(pair, target)), mode)
    hyp = normalize(join(hypothesis_view(natspec, target)), mode)
    return ref, hyp
