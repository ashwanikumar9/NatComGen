"""Attach doc comments to the declarations they document.

Attachment is positional and uses lexer spans, never a regex lookbehind: a doc
unit belongs to the declaration whose header begins at the next *code*
character after it. Anything else — a doc unit followed by another doc unit, by
a contract header, or by end of file — is recorded as orphaned rather than
guessed at, because a wrong attachment is the defect that silently poisoned
v1 (a function inherited its neighbour's @param tags).
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

from .masking import Kind, Span, code_view, scan
from .natspec import Doc, parse as parse_doc
from . import solidity as S


class Unit(NamedTuple):
    """A maximal run of doc spans separated only by whitespace or by plain
    (non-doc) comments.

    Real audited code interleaves the two: Element Finance documents some
    parameters with `///` and others with `//`, which solc ignores entirely.
    Treating the `//` line as a separator would split one comment into three
    and attach all three to the same function; treating its text as NatSpec
    would credit the file with documentation solc never sees. So the run is
    joined across the plain comment and the plain comment's text is dropped.
    `raw` is the concatenation of the doc spans only, which is why the pair
    records `doc_spans` as well as `doc_start`/`doc_end`.
    """
    spans: List[Span]
    start: int
    end: int
    raw: str


class Attachment(NamedTuple):
    doc: Doc
    decl: S.Decl
    unit: Unit


class FileModel(NamedTuple):
    path: str
    src: str
    code: str
    spans: List[Span]
    containers: List[S.Container]
    decls: List[S.Decl]
    attachments: List[Attachment]
    orphans: List[Unit]
    contract_docs: Dict[str, Doc]
    imports: List[str]
    pragma: Optional[str]


def _unit(src: str, run: List[Span]) -> Unit:
    return Unit(spans=list(run), start=run[0].start, end=run[-1].end,
                raw="".join(sp.text(src) for sp in run))


def doc_units(src: str, spans: List[Span]) -> List[Unit]:
    """Group doc spans into comments.

    A `/** ... */` block is one whole comment and never joins anything: two
    adjacent blocks are two comments, even with only a newline between them.
    88mph writes a section banner `/** Public action functions */` immediately
    above a function's own `/** @notice ... */`; merging the two put the banner
    text into the notice of a pair that then passed as gold.

    A `///` run is the opposite: one comment spread over many spans, one per
    line. It joins across whitespace and across plain `//` comments, which
    solc ignores.
    """
    units: List[Unit] = []
    run: List[Span] = []

    def close():
        nonlocal run
        if run:
            units.append(_unit(src, run))
            run = []

    for sp in spans:
        if sp.kind is Kind.DOC_BLOCK:
            close()
            units.append(_unit(src, [sp]))
            continue
        if sp.kind is Kind.DOC_LINE:
            run.append(sp)
            continue
        if sp.kind in (Kind.LINE_COMMENT, Kind.BLOCK_COMMENT):
            continue                      # ignored by solc; does not break a run
        if sp.kind is Kind.CODE and not sp.text(src).strip():
            continue                      # whitespace between doc lines
        close()
    close()
    return units


def _next_code_offset(src: str, spans: List[Span], after: int) -> Optional[int]:
    for sp in spans:
        if sp.end <= after or sp.kind is not Kind.CODE:
            continue
        seg = src[max(sp.start, after):sp.end]
        stripped = seg.lstrip()
        if stripped:
            return sp.end - len(stripped)
    return None


def build_file(path: str, src: str) -> FileModel:
    spans = scan(src, strict=True)
    code = code_view(src, spans)
    containers = S.find_containers(code, src)
    decls = S.find_decls(code, src, containers)

    by_start = {d.header_start: d for d in decls}
    ctr_by_start = {c.header_start: c for c in containers}

    attachments: List[Attachment] = []
    orphans: List[Unit] = []
    contract_docs: Dict[str, Doc] = {}

    claimed: Dict[int, Unit] = {}
    for u in doc_units(src, spans):
        off = _next_code_offset(src, spans, u.end)
        if off is None:
            orphans.append(u)
            continue
        if off in by_start or off in ctr_by_start:
            prev = claimed.get(off)
            if prev is not None:            # a banner above the real comment
                orphans.append(prev)
            claimed[off] = u                # the nearest comment wins
        else:
            orphans.append(u)               # state var, using-for, enum, ...

    for off, u in sorted(claimed.items()):
        doc = parse_doc(u.raw, u.start, u.end)
        if off in by_start:
            attachments.append(Attachment(doc=doc, decl=by_start[off], unit=u))
        else:
            contract_docs[ctr_by_start[off].name] = doc

    return FileModel(path=path, src=src, code=code, spans=spans,
                     containers=containers, decls=decls,
                     attachments=attachments, orphans=orphans,
                     contract_docs=contract_docs,
                     imports=S.imports(code, src), pragma=S.pragma(code))
