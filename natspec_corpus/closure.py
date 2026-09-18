"""File roles and the relative-import closure.

`scored` files are the ones whose comments become gold pairs. `dependency`
files are dragged in only so that the scored files compile; they are never
scored, never retrieved as exemplars, and never indexed. Keeping the two
apart is what lets the corpus be a compilation unit and an evaluation set at
the same time.
"""
from __future__ import annotations

import posixpath
import re
from typing import Dict, List, Set, Tuple

VENDORED = re.compile(
    r"(^|/)(node_modules|packages|external|vendor|lib|mocks?|tests?|"
    r"test|echidna|scripts)(/|$)", re.I)
VENDORED_NAME = re.compile(r"(^|/)(Mock|Test)[A-Z0-9_]", )


def is_vendored(rel: str) -> bool:
    return bool(VENDORED.search(rel)) or bool(VENDORED_NAME.search(rel))


def resolve_import(from_rel: str, spec: str) -> str | None:
    """Corpus-relative target of a *relative* import, or None for a package
    import (`@openzeppelin/...`), which cannot be satisfied from the corpus."""
    if not spec.startswith("."):
        return None
    return posixpath.normpath(posixpath.join(posixpath.dirname(from_rel), spec))


def closure(seeds: Set[str], imports_of: Dict[str, List[str]],
            present: Set[str]) -> Tuple[Set[str], Dict[str, List[str]]]:
    """Files reachable from `seeds` by relative imports, plus the unresolved
    package imports encountered, per file."""
    seen: Set[str] = set()
    unresolved: Dict[str, List[str]] = {}
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        if cur in seen or cur not in present:
            continue
        seen.add(cur)
        for spec in imports_of.get(cur, []):
            tgt = resolve_import(cur, spec)
            if tgt is None or tgt not in present:
                unresolved.setdefault(cur, []).append(spec)
                continue
            stack.append(tgt)
    return seen, unresolved
