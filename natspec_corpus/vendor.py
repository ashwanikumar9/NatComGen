"""Dependency-only sources, so a file can compile without its npm packages.

THE PROBLEM. 64 of the 135 files that carry gold pairs import something the
corpus does not contain — 282 of 362 missing edges are `@openzeppelin/...`.
solc cannot build an AST without them, Slither cannot run without solc, and so
those files get no fact table. That is why 43% of the validation split has no
Σ(f) and why the test split sits at 11% coverage. It is not a limitation of the
analysis; it is a missing dependency closure.

THE FIX. Supply a stand-in. We are not compiling to produce bytecode, we are
compiling to obtain an AST and a control-flow graph for one function, and any
declaration that type-checks yields the same CFG for the function under
analysis — an external call is `external` whichever IERC20 it was checked
against. DAppSCAN ships whole versioned package trees (openzeppelin-contracts-
4.2.0, openzeppelin-contracts-upgradeable-master, …), so the stand-in is a real
package at a real version rather than a synthesised stub.

THE RULE THAT KEEPS THIS HONEST. Vendored files are dependencies and nothing
else. They are never scored, never become pairs, never enter the retrieval
index, never supply reference text, and — this one matters most — they must
never take part in @inheritdoc resolution, because that would change which
documentation a val-split function is measured against. `pairs.jsonl` is frozen
before any of this runs. The only thing that changes is how many pairs have a
fact table.

Failures stay loud. If a vendored package's pragma does not match the importing
file, solc errors and that file stays uncovered exactly as it is today. The
downside of a bad substitution is no table, which is the status quo, so nothing
is silently corrupted.
"""
from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

#: Vendored files live under this prefix in a unit's source map, so a glance at
#: a unit says which of its sources came from the corpus and which did not.
PREFIX = "_vendor/"

INDEX = "index.json"


@dataclass
class Vendor:
    """A remapping from package specifiers onto local dependency trees."""
    root: Path
    #: "@openzeppelin/contracts/" -> ["openzeppelin-contracts-4.2.0/contracts/",
    #:                                  "openzeppelin-contracts-3.4/contracts/"]
    #: A list, in order, because the corpus spans OpenZeppelin v3 and v4 and
    #: they moved files between releases: `math/SafeMath.sol` in v3 is
    #: `utils/math/SafeMath.sol` in v4. A single tree leaves every project on
    #: the other major version uncompilable. Trying trees in order is safe
    #: because a given file imports one version's paths throughout, so the
    #: fallback picks a consistent tree per compilation unit rather than
    #: mixing releases inside one.
    prefixes: Dict[str, List[str]] = field(default_factory=dict)
    #: Corpus-relative paths, so a non-relative import that happens to name a
    #: corpus file resolves instead of being written off as a package.
    corpus_files: Set[str] = field(default_factory=set)
    used: Set[str] = field(default_factory=set)
    provenance: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, vendor_root: Path,
             corpus_files: Optional[Set[str]] = None) -> Optional["Vendor"]:
        """The vendor tree, or None when there isn't one. Absence is normal."""
        vendor_root = Path(vendor_root)
        index = vendor_root / INDEX
        if not index.exists():
            return None
        data = json.loads(index.read_text(encoding="utf-8"))
        prefixes = {k: ([v] if isinstance(v, str) else list(v))
                    for k, v in (data.get("prefixes") or {}).items()}
        return cls(root=vendor_root, prefixes=prefixes,
                   corpus_files=set(corpus_files or ()),
                   provenance=dict(data.get("provenance") or {}))

    # -- resolution ------------------------------------------------------
    def resolve(self, from_rel: str, spec: str) -> Optional[str]:
        """Drop-in for `closure.resolve_import`, with two additions.

        A relative import behaves exactly as before — including from inside a
        vendored file, where the arithmetic stays within that package's own
        tree because the tree's directory structure is preserved.
        """
        if spec.startswith("."):
            return posixpath.normpath(
                posixpath.join(posixpath.dirname(from_rel), spec))

        # 1. A non-relative import that names a corpus file. These are real:
        #    `element-finance/contracts/.../SafeMath.sol` is in the corpus and
        #    was being written off as a package import purely because it does
        #    not begin with a dot.
        if spec in self.corpus_files:
            return spec

        # 2. A package import with a remapping. Longest prefix wins, so
        #    `@openzeppelin/contracts-upgradeable/` is not shadowed by
        #    `@openzeppelin/contracts/`.
        for prefix in sorted(self.prefixes, key=len, reverse=True):
            if not spec.startswith(prefix):
                continue
            rest = spec[len(prefix):]
            for tree in self.prefixes[prefix]:
                target = PREFIX + tree + rest
                if (self.root / target[len(PREFIX):]).is_file():
                    self.used.add(spec)
                    return target
        return None

    # -- reading ---------------------------------------------------------
    def read(self, rel: str) -> Optional[str]:
        if not rel.startswith(PREFIX):
            return None
        path = self.root / rel[len(PREFIX):]
        if not path.is_file():
            return None
        # Bytes, decoded without newline translation: `Path.read_text` opens in
        # universal-newline mode, which turns \r\n into \n and shortens the
        # string, and every byte offset downstream is then wrong.
        return path.read_bytes().decode("utf-8", "replace")

    def wrap(self, base_read: Callable[[str], Optional[str]]
             ) -> Callable[[str], Optional[str]]:
        """A reader that serves the corpus first and the vendor tree second."""
        def read(rel: str) -> Optional[str]:
            got = base_read(rel)
            return got if got is not None else self.read(rel)
        return read

    # -- reporting -------------------------------------------------------
    def report(self) -> dict:
        return {"root": str(self.root),
                "prefixes": dict(self.prefixes),
                "specs_remapped": sorted(self.used),
                "provenance": self.provenance}


def sources_from_vendor(unit) -> List[str]:
    """Which of a unit's sources are stand-ins. Provenance for one table."""
    return sorted(r for r in unit.sources if r.startswith(PREFIX))
