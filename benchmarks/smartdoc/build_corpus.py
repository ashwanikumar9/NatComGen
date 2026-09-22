"""Turn the matched functions into a corpus the existing pipeline can build.

The trick here is deliberately blunt: rather than writing `pairs.jsonl` by
hand — which would couple this benchmark to a schema that is free to change,
and would quietly diverge from it — the SmartDoc reference is written *into*
a copy of the recovered contract as a `/// @notice` line above its function.
The ordinary corpus builder then extracts it as a gold pair through exactly
the same code path as every other pair in the project, and Σ(f) runs on a
file that still compiles, because a comment cannot change what a compiler
sees.

One source file becomes one project, named `smartdoc-<split>-NNNN` with a
fixed width. The width is not cosmetic: `projects.slug_for` matches by
substring, so `smartdoc-test-1` would also match `smartdoc-test-12` and two
different contracts would collapse into one project.

Those project names are registered in `natspec_corpus.projects` **in this
process only**, by appending to the module's list and mutating its split
sets. Nothing on disk is edited. Run the ordinary build in a separate
process and it behaves exactly as it always has.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from natspec_corpus import projects                       # noqa: E402
from natspec_corpus.masking import Kind, scan             # noqa: E402

from .index import read_source                            # noqa: E402

DOC = {Kind.DOC_LINE, Kind.DOC_BLOCK, Kind.LINE_COMMENT, Kind.BLOCK_COMMENT}


def _strip_doc_above(src: str, spans, start: int) -> int:
    """Where the function's text begins once any comment above it is gone.

    Returns the offset the replacement should start at. Walks back over runs
    of comment spans separated only by whitespace, so a three-line `///`
    block and a `/** */` both disappear together.

    The span is located by *containment*, not by an exact end offset. A
    `doc_line` span runs to the end of its line terminator, so on a CRLF file
    its `end` sits two characters past the last non-whitespace character and
    an equality test silently finds nothing — which leaves the original
    comment in place, gives the function two doc units, and hands the
    benchmark the contract author's sentence instead of SmartDoc's.
    """
    at = start
    while True:
        cut = len(src[:at].rstrip())
        prev = [s for s in spans if s.kind in DOC and s.start < cut <= s.end]
        if not prev:
            break
        nxt = min(s.start for s in prev)
        if nxt >= at:
            break
        at = nxt
    line_start = src.rfind("\n", 0, at) + 1
    return line_start if not src[line_start:at].strip() else at


def _indent_of(src: str, at: int) -> str:
    line_start = src.rfind("\n", 0, at) + 1
    m = re.match(r"[ \t]*", src[line_start:])
    return m.group(0) if m else ""


def annotate(src: str, items: List[dict]) -> str:
    """Replace whatever sits above each matched function with its reference.

    *Replace*, not insert. Leaving the contract's own doc comment in place
    and adding ours above it gives the function two doc units, and the
    extractor attaches the nearer one — so the corpus would carry the
    original author's sentence while the benchmark believed it carried
    SmartDoc's. That failure is invisible in every count: the pair exists,
    the build passes, and the gold text is simply the wrong text.

    Bottom-up, so every offset recorded against the original file is still
    valid when its turn comes: each edit only rewrites the region between a
    function and the comment above it, which no later (earlier-in-file) item
    can overlap.
    """
    spans = scan(src, strict=False)
    out = src
    for it in sorted(items, key=lambda x: -x["start"]):
        start = it["start"]
        at = _strip_doc_above(out, spans, start)
        indent = _indent_of(out, start)
        ref = " ".join(it["reference"].split())
        block = f"{indent}/// @notice {ref}\n{indent}"
        out = out[:at] + block + out[start:]
        spans = scan(out, strict=False)
    return out


def stage(matched: Path, index_root: Path, out_src: Path, *,
          split: str = "test") -> dict:
    """Write the annotated contracts as a source tree the builder accepts."""
    items = [json.loads(l) for l in matched.read_text().splitlines() if l.strip()]
    by_file: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        by_file[it["file"]].append(it)

    if out_src.exists():
        shutil.rmtree(out_src)
    out_src.mkdir(parents=True)

    mapping: Dict[str, int] = {}
    projects_made, skipped = [], []
    for n, (rel, group) in enumerate(sorted(by_file.items())):
        slug = f"smartdoc-{split}-{n:05d}"
        src = read_source(index_root / rel)
        try:
            annotated = annotate(src, group)
        except Exception as e:                      # noqa: BLE001 - report, skip
            skipped.append({"file": rel, "why": f"{type(e).__name__}: {e}"})
            continue
        dest = out_src / slug / "repo" / Path(rel).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(annotated, encoding="utf-8")
        projects_made.append(slug)
        for it in group:
            key = f"{slug}|{it['contract']}|{it['sig']}"
            # Two matches landing on the same key would make the reverse
            # lookup ambiguous; the first wins and the clash is reported.
            if key in mapping:
                skipped.append({"file": rel, "why": f"duplicate key {key}"})
                continue
            mapping[key] = it["index"]

    meta = {"split": split, "projects": projects_made, "map": mapping,
            "source_files": len(by_file), "functions": len(mapping),
            "skipped": skipped}
    (out_src.parent / f"corpus_map_{split}.json").write_text(
        json.dumps(meta, indent=1))
    return meta


def register(slugs: List[str], *, split: str) -> None:
    """Teach `natspec_corpus.projects` about these projects, in memory.

    `split` is the corpus split these projects land in — "train", "val" or
    "test" — not SmartDoc's own split name.

    `build.py` does `from .projects import TEST_PROJECTS, VAL_PROJECTS`, so it
    holds references to those *objects*. Rebinding the names here would do
    nothing; mutating the sets in place is what reaches it. Membership of
    neither set means "train", which is why that case removes rather than
    adds.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train|val|test, got {split!r}")
    known = {s for _, s in projects.SLUGS}
    for s in slugs:
        if s not in known:
            projects.SLUGS.append((s, s))
    for name, bag in (("test", projects.TEST_PROJECTS),
                      ("val", projects.VAL_PROJECTS)):
        if name == split:
            bag.update(slugs)
        else:
            bag.difference_update(slugs)


def build(out_src: Path, out_corpus: Path, slugs: List[str], *,
          split: str) -> dict:
    """The ordinary corpus build, with one expected invariant tolerated.

    `checks.artifacts_exist` refuses a corpus whose `pairs.jsonl` is empty,
    and on a recovered Etherscan contract that is the *normal* outcome: a
    SmartDoc reference is one sentence, the builder scores a comment as
    verified only when it documents every parameter and return, and crawled
    contracts rarely carry NatSpec of their own to make up the difference.
    So the verified set can legitimately be empty here while the partial set
    holds every function the benchmark cares about.

    The build has already written every artifact by the time that check runs,
    so the failure is caught, named in the result and left for `finalise` to
    resolve by promotion. Any *other* invariant is a real failure and is
    re-raised — this tolerance is for one named check, not for checking.
    """
    from natspec_corpus.build import build as corpus_build
    from natspec_corpus.errors import InvariantError

    register(slugs, split=split)
    try:
        return corpus_build(out_src, out_corpus)
    except InvariantError as e:
        if "pairs.jsonl is empty" not in str(e):
            raise
        report = out_corpus / "build_report.json"
        stats = json.loads(report.read_text()) if report.exists() else {}
        stats["tolerated_invariant"] = str(e)
        print(f"  ! {e} — expected when the recovered contracts carry no "
              f"complete NatSpec of their own; the matched functions are "
              f"promoted from pairs_partial.jsonl next.", file=sys.stderr)
        return stats


def finalise(corpus_root: Path, corpus_map: Path, *, split: str = "val") -> dict:
    """Make the matched functions — and only those — the scored split.

    Two corrections, both forced by what the corpus builder means by its
    labels rather than by anything wrong with it.

    **Promotion.** A SmartDoc reference is one sentence. The builder scores a
    comment as *verified* only when it documents every parameter and return,
    so almost every injected pair lands in `pairs_partial.jsonl` and the
    runner, which reads `pairs.jsonl`, would never see it. "Partial" is the
    right verdict about the comment and the wrong basis for excluding the
    function: the gold here is notice-only by construction, and the notice is
    complete. The matched pairs are therefore moved across, carrying their
    defect list with them so nothing is hidden.

    **Isolation.** The recovered contracts contain their own documentation,
    which would otherwise be run, scored and charged to the benchmark
    although SmartDoc has no reference for any of it. Those pairs are moved
    to a split named `excluded`, which no stage reads — not to `train`, since
    they sit in the same files as the evaluation functions and would be
    retrievable as exemplars for them.
    """
    keys = set(json.loads(corpus_map.read_text())["map"])

    def key_of(p: dict) -> str:
        return f"{p['project']}|{p['container']}|{p['signature']}"

    def rows(name: str) -> List[dict]:
        f = corpus_root / name
        if not f.exists():
            return []
        return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()
                if l.strip()]

    verified, partial = rows("pairs.jsonl"), rows("pairs_partial.jsonl")
    seen = {p["id"] for p in verified}
    promoted = [p for p in partial if key_of(p) in keys and p["id"] not in seen]
    for p in promoted:
        p["promoted_from_partial"] = True

    kept = verified + promoted
    matched_ids, excluded_ids = [], []
    for p in kept:
        if key_of(p) in keys:
            p["split"] = split
            matched_ids.append(p["id"])
        else:
            p["split"] = "excluded"
            excluded_ids.append(p["id"])

    (corpus_root / "pairs.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in kept),
        encoding="utf-8")
    splits = {split: sorted(matched_ids), "excluded": sorted(excluded_ids)}
    for s in ("train", "val", "test"):
        splits.setdefault(s, [])
    (corpus_root / "splits.json").write_text(json.dumps(splits, indent=1))

    # The retrieval index is built from the allowlist; an evaluation function
    # must never be retrievable as an exemplar for itself or its neighbour.
    allow = corpus_root / "index_allowlist.json"
    if allow.exists():
        data = json.loads(allow.read_text())
        if isinstance(data, list):
            data = [i for i in data if i not in set(matched_ids)]
        elif isinstance(data, dict):
            data = {k: v for k, v in data.items() if k not in set(matched_ids)}
        allow.write_text(json.dumps(data, indent=1))

    return {"scored_split": split, "scored": len(matched_ids),
            "promoted_from_partial": len(promoted),
            "excluded": len(excluded_ids)}


def main(argv=None) -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matched", type=Path, default=here / "data/matched.jsonl")
    ap.add_argument("--index-root", type=Path, required=True,
                    help="the root the index was built over")
    ap.add_argument("--out-src", type=Path,
                    default=here / "data/smartdoc_sources")
    ap.add_argument("--out-corpus", type=Path,
                    default=here / "data/smartdoc_corpus")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--stage-only", action="store_true")
    a = ap.parse_args(argv)

    meta = stage(a.matched, a.index_root, a.out_src, split=a.split)
    print(f"staged {meta['functions']} functions from "
          f"{meta['source_files']} contracts", file=sys.stderr)
    if a.stage_only:
        print(json.dumps({k: v for k, v in meta.items() if k != "map"}, indent=1))
        return 0
    # Matched SmartDoc functions are the evaluation set, so they go into the
    # split the ablation matrix actually scores.
    scored_split = "val" if a.split == "test" else "train"
    stats = build(a.out_src, a.out_corpus, meta["projects"], split=scored_split)
    fin = finalise(a.out_corpus,
                   a.out_src.parent / f"corpus_map_{a.split}.json",
                   split=scored_split)
    print(json.dumps({"staged": meta["functions"],
                      "build": {k: stats[k] for k in list(stats)[:8]},
                      "final": fin}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
