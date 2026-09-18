"""Putting many comments into one file without corrupting it.

The rule that makes this safe: **insert bottom-up**. Every insertion shifts
every offset below it, so the file is written in descending offset order and
no edit ever invalidates the next one. `gate.attach` does this for a single
comment; here it is a whole file, where an off-by-one is not a bad comment but
a broken contract.

Two modes, because overwriting a person's documentation by default is the
behaviour that gets a tool uninstalled:

  fill_gaps  only declarations that carry no documentation (the default)
  replace    every declaration we have a comment for

Idempotence falls out of the same mechanism: a previously emitted comment is
just a doc span on the declaration, so a second run removes it and writes the
same text back. Emitting twice produces a byte-identical file, and that is a
test rather than a hope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .emit import (Comment, DEFAULT_WIDTH, detect_style, render,
                   render_for)
from .errors import CorpusError
from .extract import FileModel, build_file


class AssemblyError(CorpusError):
    """The file could not be assembled safely."""


class _ContainerDoc:
    """Adapts a contract-level Doc to the (unit.spans) shape a Placement
    expects, so container and member comments take the same code path."""

    def __init__(self, doc) -> None:
        self.doc = doc

    @property
    def unit(self):
        class _U:
            spans = ()
        u = _U()
        u.spans = (_Span(self.doc.start, self.doc.end),)
        return u


@dataclass
class _Span:
    start: int
    end: int


@dataclass
class Placement:
    """One comment destined for one declaration."""
    offset: int                 # the declaration's header_start
    comment: Comment
    replaced: List[Tuple[int, int]]   # doc spans removed, if any


_WORD = __import__("re").compile(r"[A-Za-z]{2,}")


def _is_documentation(attachment) -> bool:
    """Does this attached comment actually document anything?

    Lens Protocol separates its sections with a banner:

        /// ************************
        /// *****VIEW FUNCTIONS*****
        /// ************************

    solc reads untagged prose as `@notice`, so that banner IS the following
    function's documentation as far as the compiler is concerned — and in
    fill-gaps mode it makes an undocumented function look documented. The test
    is therefore: any explicit tag counts, and untagged prose counts only when
    it has three or more words. The threshold is a judgement call; it is set
    where a section banner falls below it and a real one-line notice does not.
    """
    doc = attachment.doc
    if any(t.arg != "implicit" for t in doc.tags):
        return True
    prose = " ".join(t.text for t in doc.tags)
    return len(_WORD.findall(prose)) >= 3


def remove_spans(src: str, spans: Sequence[Tuple[int, int]]) -> str:
    """Delete comment ranges without leaving their indentation behind.

    A `///` span starts at the slashes, not at the start of the line, and ends
    after its newline. Deleting the raw range therefore leaves the four spaces
    that preceded it welded onto the next line — so replacing a two-line
    comment pushed its declaration eight columns to the right, and doing it
    twice pushed it sixteen. Whole comment lines go entirely; a line that also
    holds code keeps the code.
    """
    if not spans:
        return src
    covered = bytearray(len(src))
    for lo, hi in spans:
        for i in range(max(lo, 0), min(hi, len(src))):
            covered[i] = 1
    out, pos = [], 0
    for line in src.splitlines(keepends=True):
        end = pos + len(line)
        if not any(covered[pos:end]):
            out.append(line)          # untouched — never rstrip it, the last
            pos = end                 # line may be the declaration's indent
            continue
        body = line.rstrip("\r\n")
        kept = "".join(ch for i, ch in enumerate(line[:len(body)], start=pos)
                       if not covered[i])
        if body.strip() and not kept.strip():
            pos = end
            continue                  # the whole line was comment
        out.append(kept.rstrip() + line[len(body):])
        pos = end
    return "".join(out)


def _indent_at(src: str, offset: int) -> str:
    line_start = src.rfind("\n", 0, offset) + 1
    lead = src[line_start:offset]
    return lead if not lead.strip() else ""


def plan(src: str, comments: Dict[int, Comment], *, mode: str = "fill_gaps",
         model: Optional[FileModel] = None) -> List[Placement]:
    """Decide what to write where. Pure: reads, never edits.

    `comments` is keyed by declaration offset — the same `header_start` the
    corpus records as `code_start`, so a stage-2 record addresses a
    declaration without any name matching.
    """
    if mode not in ("fill_gaps", "replace"):
        raise AssemblyError(f"unknown mode {mode!r}")
    model = model or build_file("<memory>", src)
    documented = {a.decl.header_start: a for a in model.attachments}
    by_offset = {d.header_start: d for d in model.decls}
    # Contracts, interfaces and libraries are placement targets too: @title
    # and @author are illegal on a function and belong above the `contract`
    # keyword. Their doc units live in model.contract_docs, keyed by name.
    containers = {c.header_start: c for c in model.containers}
    by_offset.update(containers)
    doc_unit_by_name = {name: doc for name, doc in model.contract_docs.items()}
    for c in model.containers:
        doc = doc_unit_by_name.get(c.name)
        if doc is not None:
            documented[c.header_start] = _ContainerDoc(doc)

    out: List[Placement] = []
    for offset, comment in comments.items():
        if offset not in by_offset:
            raise AssemblyError(
                f"no declaration at offset {offset}; the file has "
                f"{len(by_offset)} and the nearest is "
                f"{min(by_offset, key=lambda o: abs(o - offset), default='-')}")
        if comment.empty:
            continue
        existing = documented.get(offset)
        real = existing is not None and _is_documentation(existing)
        if real and mode == "fill_gaps":
            continue
        # A `///` banner sitting directly above a declaration IS that
        # declaration's NatSpec as far as solc is concerned — there is no way
        # to add a comment beside it without the two merging into one doc
        # unit. So it does not count as documentation (it does not block a
        # fill), but it is replaced when we fill, because leaving it would
        # put a row of asterisks in the emitted @notice. A banner written
        # with plain `//` is never touched.
        spans = ([[s.start, s.end] for s in existing.unit.spans]
                 if existing is not None else [])
        out.append(Placement(offset=offset, comment=comment,
                             replaced=[tuple(s) for s in spans]))
    return out


def apply(src: str, placements: Sequence[Placement], *,
          style: Optional[str] = None, width: int = DEFAULT_WIDTH,
          model: Optional[FileModel] = None) -> str:
    """Write the placements into `src`, bottom-up."""
    model = model or build_file("<memory>", src)
    by_offset = {d.header_start: d for d in model.decls}
    style = style or detect_style(src)

    containers = {c.header_start: c for c in model.containers}
    by_offset.update(containers)

    out = src
    # Descending, so that an edit never moves an offset we have not used yet.
    for p in sorted(placements, key=lambda p: p.offset, reverse=True):
        decl = by_offset.get(p.offset)
        if decl is None:
            raise AssemblyError(f"no declaration at offset {p.offset}")
        indent = _indent_at(out, p.offset)
        if p.offset in containers:
            text = render(p.comment, style=style, indent=indent, width=width)
        else:
            text = render_for(p.comment, decl, style=style, indent=indent,
                              width=width)
        if not text:
            continue
        # Remove what is being replaced, then insert at the start of the
        # declaration's own line. The file is split at the declaration first,
        # so the new offset is simply the length of the rewritten prefix —
        # no arithmetic that can drift from what the removal actually did.
        prefix, suffix = out[:p.offset], out[p.offset:]
        prefix = remove_spans(prefix, [(lo, hi) for lo, hi in p.replaced
                                       if hi <= p.offset])
        line_start = prefix.rfind("\n", 0, len(prefix)) + 1
        out = prefix[:line_start] + text + "\n" + prefix[line_start:] + suffix
    return out


def document_file(src: str, comments: Dict[int, Comment], *,
                  mode: str = "fill_gaps", style: Optional[str] = None,
                  width: int = DEFAULT_WIDTH) -> str:
    model = build_file("<memory>", src)
    return apply(src, plan(src, comments, mode=mode, model=model),
                 style=style, width=width, model=model)


# --------------------------------------------------------------------------
# @inheritdoc policy
# --------------------------------------------------------------------------

def inheritdoc_targets(models: Dict[str, FileModel]) -> Dict[str, set]:
    """(contract, signature) pairs that carry real documentation.

    Used to decide whether an override may point at its base instead of
    repeating it. A base that is itself undocumented is not a valid target:
    `@inheritdoc` at an empty declaration documents nothing, and solc will not
    warn about it.
    """
    out: Dict[str, set] = {}
    for rel, m in models.items():
        for a in m.attachments:
            if not a.decl.container:
                continue
            if a.doc.notice or a.doc.dev or a.doc.params:
                out.setdefault(a.decl.container, set()).add(a.decl.sig)
    return out


def choose_inheritdoc(decl, bases: Sequence[str],
                      documented: Dict[str, set]) -> Optional[str]:
    """The base to inherit from, or None to emit full text.

    Only an overriding declaration qualifies, and only when the base actually
    has documentation to inherit. Anything else gets the full text, which is
    the safe direction: a duplicated paragraph is untidy, an `@inheritdoc`
    pointing at silence is a function with no documentation at all.
    """
    if not decl.overrides and not bases:
        return None
    for base in bases:
        if decl.sig in documented.get(base, ()):
            return base
    return None
