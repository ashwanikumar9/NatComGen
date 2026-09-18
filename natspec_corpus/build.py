"""Build the corpus end to end, then verify it before it is allowed to ship.

Stages
  1 discover   staged audit repos -> contracts/<slug>/<repo-relative path>
  2 model      lex + parse every file (a lex failure quarantines that file)
  3 select     scored seeds = non-vendored files with at least one doc'd decl
  4 closure    dependency files = relative-import closure of the seeds
  5 inherit    resolve @inheritdoc across the whole model set
  6 judge      verified / partial pairs, with defect labels
  7 dedup      identical prose on the same signature keeps one copy
  8 split      project-level, group-aware
  9 write      contracts/, pairs.jsonl, pairs_partial.jsonl, manifest.json,
               splits.json, index_allowlist.json, README.md
 10 check      invariants; a failure raises and nothing is declared good
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from .closure import closure, is_vendored, resolve_import
from .errors import CorpusError, LexError, ParseError
from .extract import FileModel, build_file
from .inherit import Index, index, resolve
from .projects import TEST_PROJECTS, VAL_PROJECTS, slug_for
from .score import is_candidate, judge
from . import checks


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _returns(doc, decl):
    """Pair @return tags with the declaration's named return values.

    `@return depositID The ID of the deposit` names a declared return; the name
    must come out of the text, not be duplicated into both fields. `@return the
    sum` names nothing and the whole line is the text.
    """
    declared = [p.name for p in decl.returns if p.name]
    out = []
    for i, t in enumerate(doc.returns):
        name, text = None, t.text
        if t.arg and t.arg in declared:
            name = t.arg
            text = text[len(t.arg):].lstrip() if text.startswith(t.arg) else text
        elif i < len(declared) and not t.arg:
            name = declared[i]
        out.append({"name": name, "text": text})
    return out


def norm_prose(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# -- 1 discover ------------------------------------------------------------

def discover(root: Path) -> Dict[str, Path]:
    """corpus-relative path -> staged source path."""
    out: Dict[str, Path] = {}
    for audit in sorted(p for p in root.iterdir() if p.is_dir()):
        slug = slug_for(audit.name)
        if slug is None:
            print(f"  ! no slug for {audit.name}, skipped", file=sys.stderr)
            continue
        repos = [p for p in audit.iterdir() if p.is_dir()]
        if len(repos) != 1:
            print(f"  ! {audit.name}: expected 1 repo dir, found {len(repos)}",
                  file=sys.stderr)
        for repo in repos:
            for f in sorted(repo.rglob("*.sol")):
                rel = f"{slug}/{f.relative_to(repo).as_posix()}"
                out[rel] = f
    return out


# -- 2 model ---------------------------------------------------------------

def model_all(files: Dict[str, Path]):
    models: Dict[str, FileModel] = {}
    quarantine: Dict[str, str] = {}
    for rel, path in files.items():
        src = path.read_text(encoding="utf-8", errors="replace")
        try:
            models[rel] = build_file(rel, src)
        except (LexError, ParseError) as e:
            quarantine[rel] = f"{type(e).__name__}: {e}"
    return models, quarantine


# -- main ------------------------------------------------------------------

def build(src_root: Path, out_root: Path) -> dict:
    print("1 discover")
    files = discover(src_root)
    print(f"   {len(files)} .sol files across "
          f"{len({r.split('/')[0] for r in files})} projects")

    print("2 model")
    models, quarantine = model_all(files)
    print(f"   {len(models)} modelled, {len(quarantine)} quarantined")
    for rel, why in quarantine.items():
        print(f"     ! {rel}: {why}")

    print("3 select")
    seeds = set()
    for rel, m in models.items():
        if is_vendored(rel):
            continue
        if any(is_candidate(a.decl) for a in m.attachments):
            seeds.add(rel)
    print(f"   {len(seeds)} scored seeds")

    print("4 closure")
    imports_of = {rel: m.imports for rel, m in models.items()}
    present = set(models)
    reach, unresolved = closure(seeds, imports_of, present)
    deps = reach - seeds
    print(f"   {len(deps)} dependency files, "
          f"{sum(len(v) for v in unresolved.values())} unresolved imports")

    print("5 inherit")
    idx = index(list(models.values()))

    print("6 judge")
    pairs: List[dict] = []
    resolved_n = 0
    for rel in sorted(seeds):
        m = models[rel]
        for att in m.attachments:
            if not is_candidate(att.decl):
                continue
            use, source_of = att, None
            if att.doc.inheritdoc:
                tgt = resolve(att, idx)
                if tgt is None:
                    continue                      # unresolvable: not gold
                use, source_of = tgt, tgt.decl.qualified
                resolved_n += 1
            v = judge(att._replace(doc=use.doc))
            d, doc = att.decl, use.doc
            spans = [[s.start, s.end] for s in att.unit.spans]
            end = d.body_end or d.header_end
            pairs.append({
                "id": sha1(f"{rel}:{d.qualified}:{d.header_start}")[:16],
                "project": rel.split("/")[0],
                "file": rel,
                "container": d.container,
                "container_kind": d.container_kind,
                "kind": d.kind,
                "name": d.name,
                "signature": d.sig,
                "arity_key": d.arity_key,
                "visibility": d.visibility,
                "mutability": d.mutability,
                "is_virtual": d.is_virtual,
                "overrides": d.overrides,
                "group_id": sha1(f"{rel.split(chr(47))[0]}|{d.sig}")[:12],
                "doc_start": att.doc.start,
                "doc_end": att.doc.end,
                "doc_spans": spans,
                "doc_raw": att.doc.raw,
                "inheritdoc": att.doc.inheritdoc,
                "resolved_from": source_of,
                "notice": doc.notice,
                "dev": doc.dev,
                "params": doc.params,
                "returns": _returns(doc, d),
                "code_start": d.header_start,
                "code_end": end,
                "code": m.src[d.header_start:end],
                "verified": v.verified,
                "defects": v.defects,
            })
    print(f"   {len(pairs)} pairs, {resolved_n} via @inheritdoc, "
          f"{sum(p['verified'] for p in pairs)} verified")

    print("7 split")
    for p in pairs:
        pr = p["project"]
        p["split"] = ("test" if pr in TEST_PROJECTS
                      else "val" if pr in VAL_PROJECTS else "train")

    print("8 dedup")
    # Identical prose on an identical signature is one item of knowledge. When
    # the twins straddle splits, the evaluation copy is the one kept and the
    # training copy is dropped, so the held-out item cannot have been seen.
    # Order decides which twin of a duplicated comment survives.
    #   1. split: the held-out copy is kept, so a test item cannot have been
    #      seen in training.
    #   2. body: an implementation beats an interface declaration. After
    #      @inheritdoc resolution the two carry identical prose, and until
    #      this was ordered explicitly the survivor was whichever file sorted
    #      first alphabetically. Keeping the declaration throws away the one
    #      with a body — the only one that has a CFG, data dependencies or
    #      anything for Sigma(f) to describe.
    rank = {"test": 0, "val": 1, "train": 2}

    def has_body(p) -> bool:
        return "{" in p["code"] and p["code"].rstrip().endswith("}")

    seen, kept, dropped = set(), [], 0
    for p in sorted(pairs, key=lambda p: (rank[p["split"]], not has_body(p),
                                          p["file"], p["code_start"])):
        key = (p["signature"], norm_prose(p["notice"] + "|" + p["dev"]))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(p)
    pairs = sorted(kept, key=lambda p: (p["file"], p["code_start"]))
    print(f"   {dropped} duplicates dropped, {len(pairs)} remain")

    verified = [p for p in pairs if p["verified"]]
    partial = [p for p in pairs if not p["verified"]]
    counts = Counter(p["split"] for p in verified)
    print(f"   verified train/val/test = "
          f"{counts['train']}/{counts['val']}/{counts['test']}")

    print("9 write")
    out_root.mkdir(parents=True, exist_ok=True)
    cdir = out_root / "contracts"
    if cdir.exists():
        shutil.rmtree(cdir)
    manifest = {}
    for rel in sorted(seeds | deps):
        dst = cdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = models[rel].src
        dst.write_text(text, encoding="utf-8")
        manifest[rel] = {
            "role": "scored" if rel in seeds else "dependency",
            "project": rel.split("/")[0],
            "sha1": sha1(text),
            "bytes": len(text.encode("utf-8")),
            "pragma": models[rel].pragma,
            "unresolved_imports": sorted(set(unresolved.get(rel, []))),
        }

    (out_root / "pairs.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in verified),
        encoding="utf-8")
    (out_root / "pairs_partial.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in partial),
        encoding="utf-8")
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    splits = defaultdict(list)
    for p in verified:
        splits[p["split"]].append(p["id"])
    (out_root / "splits.json").write_text(
        json.dumps({k: sorted(v) for k, v in splits.items()}, indent=1),
        encoding="utf-8")

    # retrieval allowlist: only scored TRAIN files may ever be indexed
    allow = defaultdict(list)
    for rel in sorted(seeds):
        pr = rel.split("/")[0]
        s = ("test" if pr in TEST_PROJECTS
             else "val" if pr in VAL_PROJECTS else "train")
        allow[s].append(rel)
    allowlist = {
        "policy": "default-deny; only files under 'train' may be indexed",
        "train": allow["train"], "val": allow["val"], "test": allow["test"],
        "never_index": sorted(deps),
    }
    (out_root / "index_allowlist.json").write_text(
        json.dumps(allowlist, indent=1), encoding="utf-8")

    report = {
        "files_total": len(files),
        "quarantined": quarantine,
        "scored": len(seeds),
        "dependency": len(deps),
        "pairs_verified": len(verified),
        "pairs_partial": len(partial),
        "inheritdoc_resolved": resolved_n,
        "duplicates_dropped": dropped,
        "unresolved_imports": sum(len(set(v)) for v in unresolved.values()),
        "defect_histogram": dict(Counter(
            d.split(":")[0] for p in partial for d in p["defects"]
        ).most_common()),
        "split_counts": dict(counts),
    }

    tpl = (Path(__file__).parent / "corpus_readme.md").read_text(encoding="utf-8")
    counts_block = (
        "```\n"
        f"source files          {report['files_total']:5d}   "
        f"({len({r.split('/')[0] for r in files})} projects)\n"
        f"quarantined           {len(quarantine):5d}\n"
        f"scored files          {len(seeds):5d}   comments become pairs\n"
        f"dependency files      {len(deps):5d}   present only so the scored "
        f"files compile\n"
        f"pairs verified        {len(verified):5d}   "
        f"train {counts['train']} / val {counts['val']} / test {counts['test']}\n"
        f"pairs partial         {len(partial):5d}   with "
        f"{sum(len(p['defects']) for p in partial)} defect labels\n"
        f"@inheritdoc resolved  {resolved_n:5d}\n"
        f"duplicates dropped    {dropped:5d}\n"
        "```")
    hist = report["defect_histogram"]
    defects_block = "```\n" + "\n".join(
        f"{k:<16}{v}" for k, v in hist.items()) + "\n```"
    (out_root / "README.md").write_text(
        tpl.replace("{{COUNTS}}", counts_block)
           .replace("{{DEFECTS}}", defects_block)
           .replace("{{NFILES}}", str(len(seeds) + len(deps)))
           .replace("{{NVER}}", str(len(verified)))
           .replace("{{NPAR}}", str(len(partial))),
        encoding="utf-8")

    print("10 check")
    checks.run_all(out_root, models, verified, partial, manifest)
    print("   all invariants hold")

    (out_root / "build_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv=None):
    argv = argv or sys.argv[1:]
    src = Path(argv[0]) if argv else Path("/mnt/user-data/uploads/contracts")
    out = Path(argv[1]) if len(argv) > 1 else Path("/home/claude/corpus/out/NatSpecGold")
    rep = build(src, out)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
