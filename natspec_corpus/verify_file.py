"""Whole-file checks after emission.

The per-function gate cannot see a file. Two `@inheritdoc` tags pointing at
different bases, a `@title` that slid inside a contract body, a comment
attached one declaration off after an offset slip — each of those is a whole-
file property, and each produces a file that looks fine until solc or a
reviewer reads it.

Four checks, none of which needs a model or a reference:

  code_unchanged  strip every comment from input and output; the two must be
                  byte-identical. This is what catches an offset slip
                  corrupting code, and it is nearly free.
  compiles        the file still builds, under the same compiler.
  placement       re-extract the output and confirm each comment sits on the
                  declaration it was written for.
  completeness    solc's devdoc has an entry for every exposed member.

`completeness` is the one that measures the point of the whole system, and it
is the compiler's opinion rather than ours.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .compile import CompileError, Unit, _run, installed_versions
from .extract import build_file
from .masking import Kind, scan


@dataclass
class FileReport:
    rel: str
    code_unchanged: bool
    compiles: bool
    placement_ok: bool
    exposed: int
    documented: int
    missing: List[str] = field(default_factory=list)
    misplaced: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.code_unchanged and self.compiles and self.placement_ok

    @property
    def completeness(self) -> Optional[float]:
        return self.documented / self.exposed if self.exposed else None

    def to_dict(self) -> dict:
        return {"file": self.rel, "ok": self.ok,
                "code_unchanged": self.code_unchanged,
                "compiles": self.compiles, "placement_ok": self.placement_ok,
                "exposed": self.exposed, "documented": self.documented,
                "completeness": self.completeness,
                "missing": self.missing, "misplaced": self.misplaced,
                "errors": self.errors}


_COMMENT_KINDS = (Kind.LINE_COMMENT, Kind.DOC_LINE, Kind.BLOCK_COMMENT,
                  Kind.DOC_BLOCK)
_DOC_KINDS = (Kind.DOC_LINE, Kind.DOC_BLOCK)


def _remove_spans(src: str, kinds) -> str:
    """Delete comment spans without disturbing the lines around them.

    A `///` span ends with its newline, so deleting the raw range welds the
    next line onto the previous one's indentation. Working line by line
    instead: a line entirely covered by comment disappears, a line only partly
    covered keeps its code and loses the comment.
    """
    from .assemble import remove_spans
    return remove_spans(src, [(sp.start, sp.end) for sp in scan(src, strict=False)
                              if sp.kind in kinds])


def strip_comments(src: str) -> str:
    """Every comment removed and whitespace normalised.

    The comparison key for "emission changed nothing but comments". Both sides
    get the same treatment, so residual indentation cannot make two equal
    files look different.
    """
    text = _remove_spans(src, _COMMENT_KINDS)
    return "\n".join(line.rstrip() for line in text.splitlines()
                      if line.strip())


def strip_doc_comments(src: str, *, rel: str = "<memory>",
                       only_attached: bool = True) -> str:
    """Remove NatSpec, leaving plain `//` comments and all formatting.

    `only_attached` keeps documentation the extractor cannot place: NatSpec on
    a public state variable, a `using` directive, an enum. solc still emits a
    getter's `@notice` from a state variable's comment, so deleting one loses
    real documentation that emission has no way to write back.

    That is the general rule and not a round-trip convenience: a tool that
    removes comments it does not understand destroys a person's work. The
    honest behaviour is to leave them exactly where they are.
    """
    if not only_attached:
        return _remove_spans(src, _DOC_KINDS)

    from .extract import build_file
    model = build_file(rel, src)
    placeable = set()
    for a in model.attachments:
        placeable.update((sp.start, sp.end) for sp in a.unit.spans)
    for doc in model.contract_docs.values():
        placeable.add((doc.start, doc.end))

    from .assemble import remove_spans
    return remove_spans(src, [(sp.start, sp.end)
                              for sp in scan(src, strict=False)
                              if sp.kind in _DOC_KINDS
                              and (sp.start, sp.end) in placeable])


def exposed_members(rel: str, src: str) -> List[Tuple[str, str]]:
    """(kind, name) for members solc will document: public and external
    functions, plus errors and events. Everything else has no devdoc entry no
    matter how well commented, so counting it would make completeness
    unreachable by construction."""
    model = build_file(rel, src)
    out = []
    for d in model.decls:
        if d.kind == "function" and d.visibility in ("public", "external"):
            out.append(("function", d.sig))
        elif d.kind == "constructor":
            out.append(("constructor", "constructor"))
        elif d.kind in ("event", "error"):
            out.append((d.kind, d.sig))
    return out


def _doc_names(out: dict, rel: str) -> set:
    names = set()
    for entry in ((out.get("contracts") or {}).get(rel) or {}).values():
        for doc in ("userdoc", "devdoc"):
            d = entry.get(doc) or {}
            for section in ("methods", "errors", "events"):
                for key, val in (d.get(section) or {}).items():
                    body = val[0] if isinstance(val, list) and val else val
                    if isinstance(body, dict) and body:
                        names.add(key)
    return names


def verify(rel: str, original: str, emitted: str, unit_sources: Dict[str, str],
           version: Optional[str] = None) -> FileReport:
    """Check an emitted file against the original it came from."""
    rep = FileReport(rel=rel, code_unchanged=False, compiles=False,
                     placement_ok=False, exposed=0, documented=0)

    rep.code_unchanged = strip_comments(original) == strip_comments(emitted)

    sources = dict(unit_sources)
    sources[rel] = emitted
    unit = Unit(entry=rel, sources=sources, unresolved=[])
    payload = unit.standard_json()
    payload["settings"]["outputSelection"] = {
        "*": {"*": ["devdoc", "userdoc"], "": ["ast"]}}

    for v in ([version] if version else installed_versions()):
        try:
            out = _run(v, payload)
        except CompileError as e:
            rep.errors.append(str(e)[:160])
            continue
        errs = [e for e in out.get("errors", []) if e.get("severity") == "error"]
        if errs:
            rep.errors = [e.get("message", "")[:160] for e in errs[:3]]
            continue
        rep.compiles = True
        documented = _doc_names(out, rel)
        members = exposed_members(rel, emitted)
        rep.exposed = len(members)
        for kind, name in members:
            key = "constructor" if kind == "constructor" else name
            hit = key in documented or any(
                d.split("(", 1)[0] == name.split("(", 1)[0] for d in documented)
            if hit:
                rep.documented += 1
            else:
                rep.missing.append(f"{kind} {name}")
        break

    # Placement asks whether WE misplaced something, not whether the file has
    # any unattached comment at all. NatSpec on a public state variable is an
    # orphan to the extractor and is deliberately left where it was, so an
    # orphan that already existed in the original is not a failure — only a
    # new one is.
    model = build_file(rel, emitted)
    attached = {a.decl.header_start for a in model.attachments}
    decls = {d.header_start for d in model.decls}
    before_model = build_file(rel, original)
    was = {" ".join(original[o.start:o.end].split())
           for o in before_model.orphans}
    now = [o for o in model.orphans
           if " ".join(emitted[o.start:o.end].split()) not in was]
    rep.placement_ok = attached <= decls and not now
    if now:
        rep.misplaced = [emitted[o.start:o.start + 48].strip() for o in now[:5]]
    return rep


def devdoc_of(rel: str, sources: Dict[str, str],
              version: Optional[str] = None) -> Optional[dict]:
    """solc's userdoc+devdoc for one file, or None if it does not build.

    This is the round trip's comparison key: strip a file's comments, emit the
    same documentation back, and require solc to read exactly what it read
    from the original. If that holds, formatting, tag order, offsets,
    indentation and idempotence are all correct at once.
    """
    unit = Unit(entry=rel, sources=dict(sources), unresolved=[])
    payload = unit.standard_json()
    payload["settings"]["outputSelection"] = {
        "*": {"*": ["devdoc", "userdoc"], "": ["ast"]}}
    for v in ([version] if version else installed_versions()):
        try:
            out = _run(v, payload)
        except CompileError:
            continue
        if [e for e in out.get("errors", []) if e.get("severity") == "error"]:
            continue
        got = {}
        for cname, entry in (((out.get("contracts") or {}).get(rel)) or {}).items():
            got[cname] = {"userdoc": entry.get("userdoc"),
                          "devdoc": entry.get("devdoc")}
        return got
    return None


def _norm(value):
    """Collapse whitespace inside every string, recursively.

    solc preserves the interior spacing of a continuation line verbatim, so a
    comment wrapped as `which      improves` reaches devdoc with those spaces
    in it. The round trip asks whether solc read the same *documentation*, not
    whether it read the same whitespace, so both sides are normalised before
    comparison. Nothing else is touched: a changed word, a dropped tag or a
    reordered return still differs.
    """
    import re as _re
    if isinstance(value, str):
        return _re.sub(r"\s+", " ", value).strip()
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_norm(v) for v in value]
    return value


def devdoc_equal(a: Optional[dict], b: Optional[dict]) -> bool:
    """Did solc read the same documentation from both files?"""
    if a is None or b is None:
        return False
    return _norm(a) == _norm(b)


def devdoc_diff(a: Optional[dict], b: Optional[dict]) -> List[str]:
    """Where they disagree, one line per member."""
    out: List[str] = []
    a, b = _norm(a or {}), _norm(b or {})
    for cname in sorted(set(a) | set(b)):
        for doc in ("userdoc", "devdoc"):
            ad = ((a.get(cname) or {}).get(doc) or {})
            bd = ((b.get(cname) or {}).get(doc) or {})
            # Contract-level fields first: @title, @author and the
            # contract's own @notice/@dev live here, not under methods, and
            # a diff that ignores them reports "no differences" on a file
            # that lost its title.
            for field_name in ("title", "author", "notice", "details"):
                if ad.get(field_name) != bd.get(field_name):
                    out.append(f"{cname}.{doc}.{field_name}")
            for section in ("methods", "errors", "events"):
                am, bm = (ad.get(section) or {}), (bd.get(section) or {})
                for key in sorted(set(am) | set(bm)):
                    if am.get(key) != bm.get(key):
                        out.append(f"{cname}.{doc}.{section}.{key}")
    return out
