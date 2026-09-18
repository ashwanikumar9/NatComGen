"""Turn a corpus file into a compilation unit and compile it.

Two things make this harder than `solc file.sol`:

1. **The unit is not the file.** A Solidity file only compiles together with
   everything it imports. `closure.py` already computes that set, and the
   manifest records it; this module turns it into a standard-json input with
   every source inlined by content, so the compiler never touches the
   filesystem and the unit is reproducible from the corpus alone.

2. **The version is not the pragma.** v1 picked a compiler from the entry
   file's pragma and 64 files failed, because a unit must satisfy *every*
   pragma in it, and `^0.8.0` does not mean "use 0.8.0". The rule here is
   empirical: try the installed compilers newest-first and keep the first one
   that produces an AST with no errors. That raised coverage from 149 to 176
   files, and it cannot silently pick a wrong answer — a compiler either
   produces the AST or it does not.

Results are cached by content hash, so re-running costs nothing and the batch
driver and the single-file library path share one cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .closure import resolve_import
from .errors import CorpusError

SHIM_ROOT = Path(os.environ.get("SOLC_SHIM_ROOT", "/opt/solcshim"))


class CompileError(CorpusError):
    """The unit could not be compiled by any installed compiler."""

_probed = None
_VERSION_RE = re.compile(r"Version:\s*(\d+\.\d+\.\d+)")


def probe():
    """Ask every installed compiler what version it actually is.

    Never trust the directory name. In this container `solcjs@0.7.6` had been
    installed into the `0.8.7` folder and vice versa, so the search skipped
    every file pinned to `=0.7.6` — four of Uniswap v3-core's contracts,
    the pool among them — and mislabelled ten others in the report. The same
    thing happens with a hand-managed solc-select tree. Probing costs one
    subprocess per compiler, once per process.
    """
    global _probed
    if _probed is not None:
        return _probed
    found = {}
    if SHIM_ROOT.is_dir():
        for d in sorted(SHIM_ROOT.iterdir()):
            exe = d / "solc"
            if not exe.is_file():
                continue
            try:
                out = subprocess.run([str(exe), "--version"],
                                     capture_output=True, text=True,
                                     timeout=60).stdout
            except Exception:                        # noqa: BLE001
                continue
            m = _VERSION_RE.search(out)
            if m:
                found.setdefault(m.group(1), str(exe))
    _probed = found
    return found


def _order(v: str):
    return tuple(int(x) for x in v.split("."))


def installed_versions() -> List[str]:
    """Compilers actually present, newest first, by *reported* version."""
    return sorted(probe(), key=_order, reverse=True)


def solc_path(version: str) -> str:
    p = probe().get(version)
    if p is None:
        raise CompileError(
            f"solc {version} is not installed under {SHIM_ROOT} "
            f"(found: {', '.join(installed_versions()) or 'none'})")
    return str(p)

# --------------------------------------------------------------------------
# compilation units
# --------------------------------------------------------------------------

@dataclass
class Unit:
    """A file plus the transitive closure of its relative imports."""
    entry: str
    sources: Dict[str, str]            # corpus-relative path -> content
    unresolved: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.unresolved

    def digest(self) -> str:
        h = hashlib.sha1()
        h.update(self.entry.encode())
        for rel in sorted(self.sources):
            h.update(rel.encode())
            h.update(hashlib.sha1(self.sources[rel].encode()).digest())
        return h.hexdigest()

    def standard_json(self, *, ast_only: bool = True) -> dict:
        out = ["ast"] if ast_only else ["ast", "abi", "evm.bytecode"]
        return {
            "language": "Solidity",
            "sources": {rel: {"content": c} for rel, c in self.sources.items()},
            "settings": {"outputSelection": {"*": {"": ["ast"],
                                                   "*": out[1:]}}},
        }


def unit_for(entry: str, read: "callable") -> Unit:
    """Build the unit for `entry`. `read(rel) -> str | None` supplies content;
    returning None means the file is not in the corpus, which makes the import
    unresolved rather than raising — an npm package is a normal condition."""
    sources: Dict[str, str] = {}
    unresolved: List[str] = []
    stack = [entry]
    while stack:
        rel = stack.pop()
        if rel in sources:
            continue
        content = read(rel)
        if content is None:
            unresolved.append(rel)
            continue
        sources[rel] = content
        for spec in _imports(content):
            tgt = resolve_import(rel, spec)
            if tgt is None:
                unresolved.append(spec)
            else:
                stack.append(tgt)
    return Unit(entry=entry, sources=sources, unresolved=sorted(set(unresolved)))


def _imports(src: str) -> List[str]:
    # imports are read off the code view so a commented-out import is ignored
    from .masking import code_view, scan
    from .solidity import imports
    spans = scan(src, strict=False)
    return imports(code_view(src, spans), src)


# --------------------------------------------------------------------------
# compilation
# --------------------------------------------------------------------------

@dataclass
class CompileResult:
    unit: Unit
    version: Optional[str]
    output: Optional[dict]
    error: Optional[str]

    @property
    def ok(self) -> bool:
        return self.output is not None

    def ast(self, rel: str) -> Optional[dict]:
        if not self.ok:
            return None
        return (self.output.get("sources", {}).get(rel) or {}).get("ast")


def _run(version: str, payload: dict, timeout: int = 180) -> dict:
    proc = subprocess.run([solc_path(version), "--standard-json"],
                          input=json.dumps(payload), capture_output=True,
                          text=True, timeout=timeout)
    if proc.returncode != 0:
        raise CompileError(f"solc {version} exited {proc.returncode}: "
                           f"{proc.stderr[:200]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise CompileError(f"solc {version} produced non-JSON output: {e}")


_WRONG_VERSION = ("different compiler version", "requires different")


def compile_unit(unit: Unit, *, cache_dir: Optional[Path] = None,
                 versions: Optional[List[str]] = None) -> CompileResult:
    """Compile `unit`, trying installed compilers newest-first."""
    if not unit.complete:
        return CompileResult(unit, None, None,
                             f"{len(unit.unresolved)} unresolved imports")
    key = unit.digest()
    cache = (cache_dir / f"{key}.json") if cache_dir else None
    if cache and cache.exists():
        blob = json.loads(cache.read_text())
        return CompileResult(unit, blob.get("version"), blob.get("output"),
                             blob.get("error"))

    payload = unit.standard_json()
    last = "no compatible compiler"
    for v in (versions or installed_versions()):
        try:
            out = _run(v, payload)
        except CompileError as e:
            last = str(e)[:160]
            continue
        errs = [e for e in out.get("errors", [])
                if e.get("severity") == "error"]
        if not errs and (out.get("sources", {}).get(unit.entry) or {}).get("ast"):
            res = CompileResult(unit, v, out, None)
            break
        if not any(w in (e.get("message") or "") for e in errs
                   for w in _WRONG_VERSION):
            last = (errs[0].get("message", "no ast").splitlines() or ["?"])[0][:160]
    else:
        res = CompileResult(unit, None, None, last)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"version": res.version,
                                     "output": res.output,
                                     "error": res.error}))
    return res


def compile_corpus_file(rel: str, contracts_root: Path,
                        cache_dir: Optional[Path] = None) -> CompileResult:
    """The batch path: resolve the unit against a built corpus and compile."""
    def read(r: str) -> Optional[str]:
        p = contracts_root / r
        return p.read_text(encoding="utf-8") if p.is_file() else None
    return compile_unit(unit_for(rel, read), cache_dir=cache_dir)


def compile_source(path: str, source: str,
                   deps: Optional[Dict[str, str]] = None,
                   cache_dir: Optional[Path] = None) -> CompileResult:
    """The inference path: one contract in hand, no corpus on disk."""
    pool = dict(deps or {})
    pool[path] = source
    return compile_unit(unit_for(path, pool.get), cache_dir=cache_dir)
