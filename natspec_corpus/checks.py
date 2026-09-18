"""Post-build invariants.

These are the guard rails the v1 builder did not have. Every one of them would
have caught a bug that actually shipped: the merged-doc-block bug shows up in
`no_overlapping_docs`, the truncated `mapping(...)` parameter in
`params_round_trip`, the silently-dropped README in `artifacts_exist`.

`run_all` raises InvariantError on the first failure. A corpus that fails a
check is not written off as "mostly fine" — it is not shipped.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .closure import resolve_import
from .errors import InvariantError
from .extract import build_file
from .score import is_candidate


def _fail(name: str, detail: str):
    raise InvariantError(f"[{name}] {detail}")


def artifacts_exist(root: Path):
    # pairs_partial.jsonl may legitimately be empty: a corpus in which every
    # comment is complete has no partial pairs. Requiring content there fails
    # the cleanest possible corpus, which is the wrong way round.
    may_be_empty = {"pairs_partial.jsonl"}
    for f in ("pairs.jsonl", "pairs_partial.jsonl", "manifest.json",
              "splits.json", "index_allowlist.json", "README.md"):
        p = root / f
        if not p.exists():
            _fail("artifacts_exist", f"{f} missing")
        if f not in may_be_empty and not p.read_text().strip():
            _fail("artifacts_exist", f"{f} is empty")
        if "{{" in p.read_text():
            _fail("artifacts_exist", f"{f} has an unsubstituted placeholder")


def manifest_hashes(root: Path, manifest: Dict[str, dict]):
    for rel, meta in manifest.items():
        p = root / "contracts" / rel
        if not p.exists():
            _fail("manifest_hashes", f"{rel} in manifest but not on disk")
        got = hashlib.sha1(p.read_text(encoding="utf-8").encode()).hexdigest()
        if got != meta["sha1"]:
            _fail("manifest_hashes", f"{rel} sha1 {got} != {meta['sha1']}")
    on_disk = {str(p.relative_to(root / "contracts").as_posix())
               for p in (root / "contracts").rglob("*.sol")}
    extra = on_disk - set(manifest)
    if extra:
        _fail("manifest_hashes", f"{len(extra)} files on disk not in manifest: "
                                 f"{sorted(extra)[:3]}")


def offsets_slice_back(root: Path, pairs: List[dict]):
    cache: Dict[str, str] = {}
    for p in pairs:
        src = cache.setdefault(
            p["file"], (root / "contracts" / p["file"]).read_text(encoding="utf-8"))
        if src[p["code_start"]:p["code_end"]] != p["code"]:
            _fail("offsets_slice_back",
                  f"{p['file']}:{p['name']} code offsets do not match the file")
        joined = "".join(src[a:b] for a, b in p["doc_spans"])
        if joined != p["doc_raw"]:
            _fail("offsets_slice_back",
                  f"{p['file']}:{p['name']} doc spans do not reproduce doc_raw")
        if p["doc_spans"][0][0] != p["doc_start"] or \
                p["doc_spans"][-1][1] != p["doc_end"]:
            _fail("offsets_slice_back",
                  f"{p['file']}:{p['name']} doc_start/doc_end disagree with "
                  f"doc_spans")


def no_overlapping_docs(pairs: List[dict]):
    """The v1 apostrophe bug produced exactly this: two declarations whose doc
    ranges overlapped, because a `*/` had been blanked and two blocks merged."""
    by_file = defaultdict(list)
    for p in pairs:
        by_file[p["file"]].append(p)
    for f, ps in by_file.items():
        ps.sort(key=lambda p: p["doc_start"])
        for a, b in zip(ps, ps[1:]):
            if b["doc_start"] < a["doc_end"]:
                _fail("no_overlapping_docs",
                      f"{f}: {a['name']} and {b['name']} share doc text "
                      f"[{b['doc_start']},{a['doc_end']})")


def docs_are_single_comments(pairs: List[dict]):
    """A pair's doc must be ONE comment: either a single `/** */` block or a
    run of `///` lines. Two blocks glued together is how 88mph's section
    banner ended up inside a function's notice."""
    for p in pairs:
        raw = p["doc_raw"].strip()
        n_blocks = raw.count("/**")
        if n_blocks > 1 or (n_blocks and "///" in raw):
            _fail("docs_are_single_comments",
                  f"{p['file']}:{p['name']} doc is {n_blocks} block(s) "
                  f"and {'some' if '///' in raw else 'no'} line comments")
        if n_blocks == 1:
            if len(p["doc_spans"]) != 1:
                _fail("docs_are_single_comments",
                      f"{p['file']}:{p['name']} block doc spans "
                      f"{len(p['doc_spans'])} ranges")
            if "*/" in raw[:-2]:
                _fail("docs_are_single_comments",
                      f"{p['file']}:{p['name']} block doc contains an inner */")


def no_duplicate_decls(pairs: List[dict]):
    seen = set()
    for p in pairs:
        key = (p["file"], p["code_start"])
        if key in seen:
            _fail("no_duplicate_decls", f"{p['file']}@{p['code_start']} twice")
        seen.add(key)
    ids = [p["id"] for p in pairs]
    if len(set(ids)) != len(ids):
        _fail("no_duplicate_decls", "pair ids are not unique")


def params_round_trip(root: Path, pairs: List[dict]):
    """Re-parse each shipped file from disk and confirm the declaration a
    verified pair claims really has the parameters the pair documents."""
    cache = {}
    for p in pairs:
        if not p["verified"]:
            continue
        f = p["file"]
        if f not in cache:
            src = (root / "contracts" / f).read_text(encoding="utf-8")
            m = build_file(f, src)
            cache[f] = {d.header_start: d for d in m.decls}
        d = cache[f].get(p["code_start"])
        if d is None:
            _fail("params_round_trip",
                  f"{f}: no declaration at offset {p['code_start']}")
        actual = [x.name for x in d.params if x.name]
        if sorted(actual) != sorted(p["params"]):
            _fail("params_round_trip",
                  f"{f}:{p['name']} documents {sorted(p['params'])} "
                  f"but declares {sorted(actual)}")
        if d.sig != p["signature"]:
            _fail("params_round_trip",
                  f"{f}:{p['name']} signature drift {d.sig} != {p['signature']}")


def dependencies_are_silent(pairs: List[dict], manifest: Dict[str, dict]):
    for p in pairs:
        role = manifest.get(p["file"], {}).get("role")
        if role != "scored":
            _fail("dependencies_are_silent",
                  f"{p['file']} has role {role!r} but produced a pair")


def groups_do_not_straddle(pairs: List[dict]):
    """A signature documented in both an interface and its implementation is
    one item of knowledge. If its twin sits in train while it sits in test,
    the test score is inflated by memorisation."""
    where = defaultdict(set)
    for p in pairs:
        where[p["group_id"]].add(p["split"])
    bad = {g: s for g, s in where.items() if len(s) > 1}
    if bad:
        _fail("groups_do_not_straddle",
              f"{len(bad)} signature groups appear in more than one split, "
              f"e.g. {list(bad.items())[:3]}")


def verified_have_prose(pairs: List[dict]):
    for p in pairs:
        if not (p["notice"].strip() or p["dev"].strip()):
            _fail("verified_have_prose", f"{p['file']}:{p['name']} has no prose")
        if p["defects"]:
            _fail("verified_have_prose",
                  f"{p['file']}:{p['name']} is verified but carries defects "
                  f"{p['defects']}")


def imports_resolve_or_are_declared(root: Path, manifest: Dict[str, dict]):
    present = set(manifest)
    for rel, meta in manifest.items():
        src = (root / "contracts" / rel).read_text(encoding="utf-8")
        m = build_file(rel, src)
        declared = set(meta["unresolved_imports"])
        for spec in m.imports:
            tgt = resolve_import(rel, spec)
            if tgt in present:
                continue
            if spec not in declared:
                _fail("imports_resolve_or_are_declared",
                      f"{rel} imports {spec!r}, which is neither shipped nor "
                      f"declared unresolved")


def allowlist_is_default_deny(root: Path, manifest: Dict[str, dict]):
    a = json.loads((root / "index_allowlist.json").read_text())
    deps = {r for r, m in manifest.items() if m["role"] == "dependency"}
    if set(a["never_index"]) != deps:
        _fail("allowlist_is_default_deny",
              "never_index does not equal the dependency set")
    overlap = (set(a["train"]) & set(a["val"])) | (set(a["train"]) & set(a["test"]))
    if overlap:
        _fail("allowlist_is_default_deny", f"splits overlap: {sorted(overlap)[:3]}")
    if set(a["train"]) & deps:
        _fail("allowlist_is_default_deny", "a dependency file is indexable")


def run_all(root: Path, models, verified: List[dict], partial: List[dict],
            manifest: Dict[str, dict]):
    allp = verified + partial
    artifacts_exist(root)
    manifest_hashes(root, manifest)
    offsets_slice_back(root, allp)
    no_overlapping_docs(allp)
    no_duplicate_decls(allp)
    docs_are_single_comments(allp)
    dependencies_are_silent(allp, manifest)
    verified_have_prose(verified)
    params_round_trip(root, verified)
    groups_do_not_straddle(verified)
    imports_resolve_or_are_declared(root, manifest)
    allowlist_is_default_deny(root, manifest)


# ==========================================================================
# Σ(f) invariants
# ==========================================================================

def sigma_tables_join_to_declarations(root: Path, tables: List[dict]):
    """Every fact table must sit exactly on a declaration the extractor found.

    This is the load-bearing check of the whole stage: Slither reports byte
    offsets, the corpus records character offsets, and a file with a single
    em-dash shifts everything after it by two. A silent two-character drift
    attaches a function's control flow to its neighbour with no error
    anywhere."""
    from .extract import build_file
    cache: Dict[str, Dict[int, object]] = {}
    for t in tables:
        rel = t["file"]
        if rel not in cache:
            src = (root / "contracts" / rel).read_text(encoding="utf-8")
            cache[rel] = {d.header_start: d for d in build_file(rel, src).decls}
        if t["char_start"] not in cache[rel]:
            _fail("sigma_tables_join_to_declarations",
                  f"{rel}:{t['function']} at {t['char_start']} matches no "
                  f"declaration found by the extractor")


def sigma_offsets_are_sane(root: Path, tables: List[dict]):
    cache: Dict[str, str] = {}
    for t in tables:
        src = cache.setdefault(
            t["file"], (root / "contracts" / t["file"]).read_text(encoding="utf-8"))
        head = src[t["char_start"]:t["char_start"] + 40]
        if not head.lstrip().startswith(("function", "constructor", "fallback",
                                         "receive", "modifier")):
            _fail("sigma_offsets_are_sane",
                  f"{t['file']}:{t['function']} starts with {head[:24]!r}")
        for n in t["nodes"]:
            lo, hi = n["src"]
            if lo >= 0 and not (t["char_start"] <= lo <= t["char_end"]):
                _fail("sigma_offsets_are_sane",
                      f"{t['file']}:{t['function']} node {n['id']} at {lo} is "
                      f"outside [{t['char_start']},{t['char_end']}]")


def sigma_rows_are_well_formed(tables: List[dict]):
    for t in tables:
        ids = [r["id"] for k in ("facts", "nodes", "paths", "calls", "deps",
                                 "reverts", "events")
               for r in t.get(k, [])]
        if len(ids) != len(set(ids)):
            _fail("sigma_rows_are_well_formed",
                  f"{t['file']}:{t['function']} has duplicate row ids")
        node_ids = {n["id"] for n in t["nodes"]}
        for e in t["edges"]:
            if e["from"] not in node_ids or e["to"] not in node_ids:
                _fail("sigma_rows_are_well_formed",
                      f"{t['file']}:{t['function']} edge {e} dangles")
        for r in t.get("reverts", []) + t.get("events", []):
            if r["node"] and r["node"] not in node_ids:
                _fail("sigma_rows_are_well_formed",
                      f"{t['file']}:{t['function']} row {r['id']} dangles")
        for c in t["calls"]:
            if c["node"] and c["node"] not in node_ids:
                _fail("sigma_rows_are_well_formed",
                      f"{t['file']}:{t['function']} call {c['id']} dangles")
            # A canonical name legitimately contains `=>` when a parameter is
            # a mapping, so the marker for raw SlithIR is the operation
            # syntax itself, not a bare `=`.
            if any(m in c["target"] for m in (" = ", "dest:", "arguments:",
                                              "LIBRARY_CALL", "INTERNAL_CALL",
                                              "HIGH_LEVEL_CALL")):
                _fail("sigma_rows_are_well_formed",
                      f"{t['file']}:{t['function']} call {c['id']} target is "
                      f"raw SlithIR: {c['target'][:60]}")
        for p in t["paths"]:
            if len(set(p["nodes"])) != len(p["nodes"]):
                _fail("sigma_rows_are_well_formed",
                      f"{t['file']}:{t['function']} path {p['id']} revisits a node")
            if any(n not in node_ids for n in p["nodes"]):
                _fail("sigma_rows_are_well_formed",
                      f"{t['file']}:{t['function']} path {p['id']} dangles")
        for d in t["deps"]:
            names = [d["variable"], *d["depends_on"]]
            bad = [n for n in names if n.startswith(("TMP_", "REF_", "CONST_"))]
            if bad:
                _fail("sigma_rows_are_well_formed",
                      f"{t['file']}:{t['function']} dep {d['id']} leaks "
                      f"SlithIR temporaries {bad}")


def sigma_tables_are_unique(tables: List[dict]):
    seen = set()
    for t in tables:
        key = (t["file"], t["char_start"])
        if key in seen:
            _fail("sigma_tables_are_unique", f"{key} analysed twice")
        seen.add(key)


def sigma_pair_ids_exist(root: Path, tables: List[dict]):
    known = {}
    for name in ("pairs.jsonl", "pairs_partial.jsonl"):
        p = root / name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    known[r["id"]] = (r["file"], r["code_start"])
    for t in tables:
        if not t["pair_id"]:
            continue
        if t["pair_id"] not in known:
            _fail("sigma_pair_ids_exist",
                  f"{t['file']}:{t['function']} cites unknown pair "
                  f"{t['pair_id']}")
        f, off = known[t["pair_id"]]
        if f != t["file"] or off != t["char_start"]:
            _fail("sigma_pair_ids_exist",
                  f"{t['pair_id']} points at {f}@{off} but the table is "
                  f"{t['file']}@{t['char_start']}")


def run_sigma_checks(root: Path, tables: List[dict]):
    sigma_tables_are_unique(tables)
    sigma_tables_join_to_declarations(root, tables)
    sigma_offsets_are_sane(root, tables)
    sigma_rows_are_well_formed(tables)
    sigma_pair_ids_exist(root, tables)
