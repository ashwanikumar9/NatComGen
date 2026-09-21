"""Batch Σ(f) over a built corpus.

Reads the corpus written by `build.py`, compiles every scored file it can,
runs Slither, and writes one fact table per function, joined to the corpus
pairs by character offset.

    python -m natspec_corpus.sigma_build [CORPUS_ROOT]

Files whose imports leave the corpus (npm packages) are reported, not failed:
they light up unchanged once `npm i` is available, with no code change here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from . import sigma as _sigma
from .compile import compile_unit, unit_for
from .errors import CorpusError
from .sigma import FactTable, SigmaError, analyze

#: Shards are keyed by the analysis code as well as the source, so editing
#: sigma.py invalidates every shard rather than silently serving fact tables
#: built by the previous version of the analysis.
_ANALYSIS_VERSION = hashlib.sha1(
    Path(_sigma.__file__).read_bytes()).hexdigest()[:16]


def shard_key(rel: str, unit, version: Optional[str]) -> str:
    """A content address for one file's fact tables.

    Everything that could change the answer goes in: the file, every source
    in its compilation unit, the compiler that produced the AST, and the
    analysis code itself.
    """
    h = hashlib.sha1()
    h.update(_ANALYSIS_VERSION.encode())
    h.update(b"\0" + rel.encode())
    h.update(b"\0" + (version or "").encode())
    for name in sorted(unit.sources):
        h.update(b"\0" + name.encode())
        h.update(b"\0" + unit.sources[name].encode("utf-8"))
    return h.hexdigest()


def _read_shard(path: Path) -> Optional[List[dict]]:
    """None when the shard is absent or was truncated by a kill mid-write."""
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rows if isinstance(rows, list) else None


def _write_shard(path: Path, rows: List[dict]) -> None:
    """Written to a temporary name and renamed, so an interrupted run leaves
    either the previous shard or the new one — never half of one."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_pairs(root: Path) -> Dict[str, Dict[int, dict]]:
    """file -> {char_start: pair}. Verified and partial together, because
    Σ(f) is needed for every function the system will ever describe, not only
    the ones with gold comments."""
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for name in ("pairs.jsonl", "pairs_partial.jsonl"):
        p = root / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["file"]][rec["code_start"]] = rec
    return out


def build(corpus_root: Path, *, limit: Optional[int] = None,
          cache_dir: Optional[Path] = None, resume: bool = True) -> dict:
    """Σ(f) for every scored file in the corpus.

    `resume` keeps one shard per analysed file under `sigma/parts/`, so a run
    that is killed halfway — a dropped connection, a closed laptop — picks up
    at the file it was on instead of at the first one. The shards are content
    addressed, so this is a cache and not a stale-results trap: change a
    contract, its compiler or the analysis code and that file is analysed
    again.
    """
    contracts = corpus_root / "contracts"
    manifest = json.loads((corpus_root / "manifest.json").read_text())
    pairs = load_pairs(corpus_root)
    out_dir = corpus_root / "sigma"
    out_dir.mkdir(exist_ok=True)
    parts = out_dir / "parts"
    parts.mkdir(exist_ok=True)
    cache_dir = cache_dir or (corpus_root / ".compile-cache")

    def read(rel: str) -> Optional[str]:
        p = contracts / rel
        return p.read_text(encoding="utf-8") if p.is_file() else None

    scored = sorted(r for r, m in manifest.items() if m["role"] == "scored")
    if limit:
        scored = scored[:limit]

    report = {
        "scored_files": len(scored), "compiled": 0, "analysed": 0,
        "unresolved_imports": 0, "compile_failed": 0, "slither_failed": 0,
        "functions": 0, "joined": 0, "unjoined": 0, "reused": 0,
        "pairs_total": sum(len(v) for f, v in pairs.items() if f in set(scored)),
        "pairs_with_sigma": 0,
        "failures": {}, "compilers": {},
    }
    written: List[dict] = []

    for i, rel in enumerate(scored, 1):
        unit = unit_for(rel, read)
        if not unit.complete:
            report["unresolved_imports"] += 1
            continue
        res = compile_unit(unit, cache_dir=cache_dir)
        if not res.ok:
            report["compile_failed"] += 1
            report["failures"][rel] = f"compile: {res.error}"
            continue
        report["compiled"] += 1
        report["compilers"][res.version] = \
            report["compilers"].get(res.version, 0) + 1

        shard = parts / f"{shard_key(rel, unit, res.version)}.json"
        rows = _read_shard(shard) if resume else None
        cached = rows is not None
        if rows is None:
            try:
                tables = analyze(unit, res.version, ast_root=res.ast(rel))
            except (SigmaError, CorpusError) as e:
                report["slither_failed"] += 1
                report["failures"][rel] = str(e)[:200]
                continue
            rows = [t.to_dict() for t in tables]
            if resume:
                _write_shard(shard, rows)
        else:
            report["reused"] += 1
        report["analysed"] += 1

        by_off = pairs.get(rel, {})
        for row in rows:
            report["functions"] += 1
            pair = by_off.get(row["char_start"])
            row["pair_id"] = pair["id"] if pair is not None else None
            if pair is not None:
                report["joined"] += 1
            else:
                report["unjoined"] += 1
            written.append(row)
        print(f"  [{i}/{len(scored)}] {rel} -> {len(rows)} functions"
              + (" (cached)" if cached else ""), file=sys.stderr)

    from . import checks
    checks.run_sigma_checks(corpus_root, written)
    report["invariants"] = "all hold"

    # Never overwrite a populated table set with an empty one. Twice now a
    # run has failed on every file — once because no compiler was installed,
    # once because a solc-select shim hijacked the one that was — and written
    # zero rows over 546 good ones without complaint. Everything downstream
    # then reads an empty Σ(f): the ablations go vacuous, the retrieval views
    # collapse to code-only, and nothing anywhere says so. A stage that
    # destroys its own output on failure is worse than a stage that fails.
    target = out_dir / "sigma.jsonl"
    if not written and target.exists() and target.stat().st_size > 0:
        raise SigmaError(
            f"analysis produced no fact tables, but {target} already holds "
            f"{sum(1 for _ in target.open())} of them — refusing to overwrite. "
            f"{report['slither_failed']} files failed. First failure: "
            + (next(iter(report['failures'].values()), 'none'))[:200])
    target.write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in written),
        encoding="utf-8")
    report["pairs_with_sigma"] = len({t["pair_id"] for t in written
                                      if t["pair_id"]})
    report["node_type_histogram"] = dict(Counter(
        n["type"] for t in written for n in t["nodes"]).most_common(12))
    report["call_kind_histogram"] = dict(Counter(
        c["kind"] for t in written for c in t["calls"]).most_common())
    (corpus_root / "sigma_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv=None):
    argv = argv or sys.argv[1:]
    root = Path(argv[0]) if argv else Path("out/NatSpecGold")
    limit = int(argv[1]) if len(argv) > 1 else None
    rep = build(root, limit=limit)
    rep.pop("failures", None)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
