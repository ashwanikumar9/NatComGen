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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

#: Retained only so an older index or caller does not break; vendored sources
#: are NOT keyed by it. See `resolve` for why.
PREFIX = "_vendor/"

INDEX = "index.json"


def floor(text: str) -> Optional[tuple]:
    """The (major, minor) a pragma admits, for matching a tree to a project.

    Crude on purpose. `^0.8.0`, `>=0.8.4 <0.9.0` and `0.8.13` all give (0, 8),
    which is the only distinction that matters here: OpenZeppelin v3 is a
    0.6/0.7 library and v4 is a 0.8 one, and mixing them into one compilation
    unit produces exactly the syntax errors this was written to stop.
    """
    m = re.search(r"pragma\s+solidity[^;]*?(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


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
    #: tree -> the (major, minor) its own sources declare.
    versions: Dict[str, list] = field(default_factory=dict)
    #: The (major, minor) of the file currently being resolved for, set by
    #: `for_pragma`. Trees matching it are tried first.
    target: Optional[tuple] = None

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
                   provenance=dict(data.get("provenance") or {}),
                   versions={k: [tuple(v) for v in vs]
                             for k, vs in (data.get("versions") or {}).items()})

    def for_pragma(self, source: str) -> "Vendor":
        """This same vendor, preferring trees written for `source`'s pragma.

        A compilation unit has to satisfy every pragma in it with one compiler.
        A 0.6 project handed OpenZeppelin v4 has no such compiler, and the
        result is a page of syntax errors rather than an honest version
        complaint — which is exactly how the first attempt at this failed.
        """
        self.target = floor(source)
        return self

    def _trees(self, prefix: str) -> List[str]:
        trees = self.prefixes[prefix]
        if self.target is None:
            return trees
        fit = [t for t in trees if self.target in (self.versions.get(t) or [])]
        # Trees of unknown version keep their place behind the matching ones:
        # an unknown may still work, a known mismatch will not.
        rest = [t for t in trees if t not in fit]
        return fit + rest

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
        # 3. A package import the vendor tree can satisfy. The specifier is
        #    returned UNCHANGED, so it becomes the key under which the file
        #    appears in the unit's source map — and solc resolves imports by
        #    matching that literal string against its own `sources` keys. Key
        #    it as `_vendor/...` instead and solc reports `Source
        #    "@openzeppelin/..." not found`, which is how the first version of
        #    this failed. Returning the specifier also makes a vendored file's
        #    own relative imports land on the right package path:
        #    `@oz/contracts/token/ERC20/IERC20.sol` + `../../utils/Context.sol`
        #    normalises to `@oz/contracts/utils/Context.sol`.
        if self._file_for(spec) is not None:
            self.used.add(spec)
            return spec
        return None

    def _file_for(self, spec: str) -> Optional[Path]:
        """The vendored file backing a package specifier, if any."""
        for prefix in sorted(self.prefixes, key=len, reverse=True):
            if not spec.startswith(prefix):
                continue
            rest = spec[len(prefix):]
            for tree in self._trees(prefix):
                path = self.root / tree / rest
                if path.is_file():
                    return path
        return None

    # -- reading ---------------------------------------------------------
    def read(self, rel: str) -> Optional[str]:
        path = self._file_for(rel)
        if path is None:
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
                "prefixes": {k: list(v) for k, v in self.prefixes.items()},
                "versions": {k: [list(x) for x in v]
                             for k, v in self.versions.items()},
                "specs_remapped": sorted(self.used),
                "provenance": self.provenance}

    def is_vendored(self, rel: str) -> bool:
        return (rel not in self.corpus_files
                and any(rel.startswith(p) for p in self.prefixes))


def sources_from_vendor(unit, vendor: Optional[Vendor] = None) -> List[str]:
    """Which of a unit's sources are stand-ins. Provenance for one table."""
    if vendor is not None:
        return sorted(r for r in unit.sources if vendor.is_vendored(r))
    return sorted(r for r in unit.sources if r.startswith(PREFIX))
