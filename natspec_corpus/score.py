"""Decide whether a (declaration, doc) pair is gold.

The taxonomy is explicit and every defect is recorded by code, so a pair that
fails is still usable — `pairs_partial.jsonl` is the labelled defect set the
semantic critic is evaluated against, and it is only worth anything if the
labels are principled rather than the residue of a regex.
"""
from __future__ import annotations

import re
from typing import List, NamedTuple, Set

from .extract import Attachment
from . import solidity as S

# defect codes ------------------------------------------------------------
D_NO_TEXT        = "no_text"           # neither notice nor dev prose
D_PARAM_MISSING  = "param_missing"     # a named parameter has no @param
D_PARAM_UNKNOWN  = "param_unknown"     # @param names a non-parameter
D_PARAM_EMPTY    = "param_empty"       # @param with no description
D_RETURN_MISSING = "return_missing"
D_RETURN_EXTRA   = "return_extra"
D_RETURN_EMPTY   = "return_empty"
D_TEXT_SHORT     = "text_short"        # fewer than 3 words of prose
D_PLACEHOLDER    = "placeholder"       # TODO / TBD / FIXME / ???
D_NAME_ECHO      = "name_echo"         # prose is only the identifier restated

_PLACEHOLDER_RE = re.compile(r"\b(TODO|TBD|FIXME|XXX|WIP)\b|\?\?\?", re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _split_ident(name: str) -> Set[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ")
    return {w.lower() for w in _WORD_RE.findall(parts)}


class Verdict(NamedTuple):
    verified: bool
    defects: List[str]
    prose: str

    @property
    def partial(self) -> bool:
        return not self.verified


def judge(att: Attachment, *, resolved: bool = False) -> Verdict:
    doc, decl = att.doc, att.decl
    defects: List[str] = []

    prose = " ".join(x for x in (doc.notice, doc.dev) if x).strip()
    if not prose:
        defects.append(D_NO_TEXT)
    else:
        if _PLACEHOLDER_RE.search(prose):
            defects.append(D_PLACEHOLDER)
        words = _WORD_RE.findall(prose)
        if len(words) < 3:
            defects.append(D_TEXT_SHORT)
        elif decl.name and _split_ident(decl.name) and \
                {w.lower() for w in words} <= _split_ident(decl.name):
            defects.append(D_NAME_ECHO)

    documented = doc.params
    actual = [p.name for p in decl.params if p.name]
    for n in actual:
        if n not in documented:
            defects.append(f"{D_PARAM_MISSING}:{n}")
        elif not documented[n].strip():
            defects.append(f"{D_PARAM_EMPTY}:{n}")
    for n in documented:
        if n not in actual:
            defects.append(f"{D_PARAM_UNKNOWN}:{n}")

    rets = doc.returns
    n_ret = len(decl.returns)
    if n_ret:
        if len(rets) < n_ret:
            defects.append(D_RETURN_MISSING)
        elif len(rets) > n_ret:
            defects.append(D_RETURN_EXTRA)
        for t in rets:
            body = " ".join(x for x in (t.arg or "", t.text) if x).strip()
            if not body:
                defects.append(D_RETURN_EMPTY)
                break
    elif rets:
        defects.append(D_RETURN_EXTRA)

    return Verdict(verified=not defects, defects=defects, prose=prose)


def is_candidate(decl: S.Decl) -> bool:
    """Which declarations are eligible to become pairs at all.

    Functions, constructors, modifiers, events and errors carry NatSpec.
    solc only *interprets* tags on public/external members of a contract, but
    library internals are the documented API of a library, so visibility is
    not a filter here; it is recorded on the pair instead.
    """
    return decl.kind in ("function", "constructor", "fallback", "receive",
                         "modifier", "event", "error")
