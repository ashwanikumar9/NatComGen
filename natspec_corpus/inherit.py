"""Resolve `@inheritdoc IFoo`.

In audited modern Solidity the real NatSpec lives in `interfaces/` and the
implementation carries one line: `/// @inheritdoc IPool`. Without resolution
those functions look undocumented, which is how a corpus ends up teaching a
model that real code is uncommented.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

from .errors import ResolutionError
from .extract import Attachment, FileModel
from . import solidity as S


class Index(NamedTuple):
    # (container name, signature) -> attachment, and an arity fallback for
    # cases where a base declares the parameter types slightly differently
    by_sig: Dict[Tuple[str, str], Attachment]
    by_arity: Dict[Tuple[str, str], List[Attachment]]
    bases: Dict[str, List[str]]


def index(models: List[FileModel]) -> Index:
    by_sig: Dict[Tuple[str, str], Attachment] = {}
    by_arity: Dict[Tuple[str, str], List[Attachment]] = {}
    bases: Dict[str, List[str]] = {}
    for m in models:
        for c in m.containers:
            bases.setdefault(c.name, list(c.bases))
        for a in m.attachments:
            if not a.decl.container:
                continue
            by_sig.setdefault((a.decl.container, a.decl.sig), a)
            by_arity.setdefault((a.decl.container, a.decl.arity_key),
                                []).append(a)
    return Index(by_sig=by_sig, by_arity=by_arity, bases=bases)


def resolve(att: Attachment, idx: Index, *, max_depth: int = 8
            ) -> Optional[Attachment]:
    """Follow @inheritdoc to the attachment that holds the real prose.

    Follows chains (`@inheritdoc A` where A itself inherits) and refuses to
    loop. Returns None when the target is outside the corpus — an unresolved
    reference is reported, never silently treated as documentation.
    """
    seen = set()
    cur = att
    for _ in range(max_depth):
        target = cur.doc.inheritdoc
        if target is None:
            return cur if cur is not att else None
        key = (target, cur.decl.sig)
        if key in seen:
            raise ResolutionError(f"@inheritdoc cycle at {cur.decl.qualified}")
        seen.add(key)
        nxt = idx.by_sig.get(key)
        if nxt is None:
            cands = idx.by_arity.get((target, cur.decl.arity_key), [])
            if len(cands) == 1:
                nxt = cands[0]
        if nxt is None:
            return None
        cur = nxt
    raise ResolutionError(f"@inheritdoc too deep at {att.decl.qualified}")
