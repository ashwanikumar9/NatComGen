"""The Aggregator: per-function comments become documented files.

Everything above this point works one function at a time. The Aggregator is
where the architecture stops being a comment generator and starts being a tool
you could point at a repository: it takes every accepted comment for a file,
places each above its declaration, and writes the file back out with its code
byte-identical.

It is a thin wrapper and deliberately so. `natspec_corpus.assemble` already
solves the hard parts — where a comment goes when the declaration carries
attributes across three lines, which existing doc comment it replaces, what
indentation and comment style the file uses, and the byte-offset arithmetic
that goes wrong the moment a file has CRLF line endings. None of that is
re-derived here. What this module adds is the file-level grouping, and the
decision about which records are eligible to be placed at all.

The eligibility rule is the part worth reading: a comment is placed only if the
Judge did not fail it. A comment that a ComGen run itself marked FAIL has no
business being written into a source file, and writing it anyway is how a
"documented" output ends up less trustworthy than no documentation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from natspec_corpus.assemble import document_file
from natspec_corpus.compile import compile_unit, unit_for
from natspec_corpus.emit import Comment
from natspec_corpus.evaluate import fields
from natspec_corpus.extract import build_file
from natspec_corpus.verify_file import strip_doc_comments


@dataclass
class Assembled:
    """One file, documented, with what it takes to check it."""
    rel: str
    original: str
    emitted: str
    unit_sources: Dict[str, str]
    version: Optional[str]
    placed: int
    skipped: List[str] = field(default_factory=list)

    def write(self, out_dir: Path) -> Path:
        dst = Path(out_dir) / self.rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(self.emitted, encoding="utf-8")
        return dst


def eligible(rec: dict) -> bool:
    """Whether this record's comment may be written into a source file.

    Three ways to be ineligible, and each is a real case in the run records:
    the record is an error, the Judge failed it, or the comment came out empty
    after contradicted claims were stripped from it.
    """
    if rec.get("error"):
        return False
    if rec.get("gate_verdict") == "FAIL":
        return False
    return bool((rec.get("final") or "").strip())


def comment_of(rec: dict) -> Comment:
    """The record's final comment, as the emitter's structure."""
    f = fields(rec["final"])
    return Comment(
        notice=f.get("notice", ""), dev=f.get("dev", ""),
        params={k.split(":", 1)[1]: v for k, v in f.items()
                if k.startswith("param:")},
        returns=[v for k, v in sorted(f.items()) if k.startswith("return:")])


def assemble(corpus_root: Path, records: Iterable[dict],
             pairs: Dict[str, dict], *, mode: str = "fill_gaps",
             cache_dir: Optional[Path] = None) -> Iterator[Assembled]:
    """Group records by file and document each file. Yields one per file.

    Files that do not compile from the corpus alone are skipped rather than
    documented blind: without a compilation unit the Linter cannot check the
    result, and an unchecked emitted file is not a deliverable.
    """
    corpus_root = Path(corpus_root)
    contracts = corpus_root / "contracts"
    cache_dir = cache_dir or (corpus_root / ".compile-cache")

    def read(rel: str) -> Optional[str]:
        p = contracts / rel
        return p.read_text(encoding="utf-8") if p.is_file() else None

    by_file: Dict[str, List[dict]] = {}
    for rec in records:
        if eligible(rec):
            by_file.setdefault(rec["file"], []).append(rec)

    for rel, group in sorted(by_file.items()):
        unit = unit_for(rel, read)
        result = compile_unit(unit, cache_dir=cache_dir)
        if not result.ok:
            continue
        original = unit.sources[rel]

        want: Dict[Tuple[Any, str], Comment] = {}
        skipped: List[str] = []
        for rec in group:
            pair = pairs.get(rec["pair_id"])
            if not pair:
                skipped.append(f"{rec['pair_id']}: not in pairs.jsonl")
                continue
            want[(pair["container"], pair["signature"])] = comment_of(rec)

        # Strip the file's own documentation first. The corpus is masked per
        # function, but a file still carries comments on everything ComGen was
        # not asked about, and leaving them in makes the Linter's
        # code-unchanged check compare a documented file against a differently
        # documented one.
        stripped = strip_doc_comments(original, rel=rel)
        model = build_file(rel, stripped)
        offset_of = {(d.container, d.sig): d.header_start for d in model.decls}

        comments = {offset_of[k]: v for k, v in want.items()
                    if k in offset_of and not v.empty}
        for k in want:
            if k not in offset_of:
                skipped.append(f"{k[1]}: no declaration found after stripping")
        if not comments:
            continue

        yield Assembled(rel=rel, original=original,
                        emitted=document_file(stripped, comments, mode=mode),
                        unit_sources=unit.sources, version=result.version,
                        placed=len(comments), skipped=skipped)
