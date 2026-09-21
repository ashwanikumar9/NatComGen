"""Rendering one NatSpec comment as text solc will read back correctly.

Two things here are correctness, not formatting.

**Tag order.** `@param` and `@return` are matched by solc in **declaration
order**, not by name for returns. Emitting the returns of `(uint a, uint b)`
in the other order silently swaps what the two tags mean, and nothing
complains. So the renderer never takes the model's ordering: it takes the
declaration's, and fills each slot.

**`@inheritdoc` is exclusive.** solc inherits every tag from the base, and a
`@notice` emitted beside `@inheritdoc` is a documentation conflict. Only
`@dev` may accompany it, because implementation notes are genuinely local to
the override.

Style is matched to the file rather than fixed: a file written entirely in
`/** */` that gains a block of `///` reads as machine-generated, which is
exactly the impression a documentation tool should avoid.
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .masking import Kind, scan

DEFAULT_WIDTH = 100

#: The order solc reads and a reviewer expects. `@param` and `@return` are
#: expanded in declaration order at render time.
TAG_ORDER = ("inheritdoc", "title", "author", "notice", "dev", "param",
             "return", "custom")


@dataclass
class Comment:
    """One member's documentation, as fields rather than text."""
    # str, or a list when the source had several @notice / @dev paragraphs.
    # solc concatenates repeated tags with NO separator, so merging them into
    # one tag and re-emitting inserts a space solc never had — which is a
    # different document as far as the compiler is concerned.
    notice: object = ""
    dev: object = ""
    params: Dict[str, str] = field(default_factory=dict)
    returns: List[str] = field(default_factory=list)
    inheritdoc: Optional[str] = None
    title: str = ""
    author: str = ""
    preamble: str = ""      # untagged prose that preceded the first tag
    custom: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not any((self.notice, self.dev, self.params, self.returns,
                        self.inheritdoc, self.title, self.author, self.custom,
                        self.preamble))


def from_doc(doc) -> Comment:
    """A parsed `natspec.Doc` as fields. Used for contract-level comments,
    which have no pair record of their own.

    Untagged prose is kept separate from an explicit `@notice`. solc folds
    both into the notice but concatenates them with no separator, so merging
    them here and re-emitting one `@notice` changes what solc reads back.
    """
    implicit = [t.text for t in doc.of("notice") if t.arg == "implicit"]
    explicit = [t.text for t in doc.of("notice") if t.arg != "implicit"]
    devs = [t.text for t in doc.of("dev") if t.text]
    # `@return depositID The ID` parses with the name still at the head of the
    # text, so joining arg and text would duplicate it.
    returns = [(t.text if (t.arg and (t.text or "").startswith(t.arg))
                else " ".join(x for x in (t.arg or "", t.text or "") if x)).strip()
               for t in doc.returns]
    if implicit and explicit:
        return Comment(
            preamble=" ".join(implicit).strip(),
            notice=explicit, dev=devs, returns=returns,
            title=" ".join(t.text for t in doc.of("title")).strip(),
            author=" ".join(t.text for t in doc.of("author")).strip(),
            params=dict(doc.params),
            custom=[(t.name, t.text) for t in doc.tags
                    if t.name.startswith("custom:")])
    return Comment(
        notice=[t.text for t in doc.of("notice") if t.text], dev=devs,
        returns=returns,
        title=" ".join(t.text for t in doc.of("title")).strip(),
        author=" ".join(t.text for t in doc.of("author")).strip(),
        params=dict(doc.params),
        custom=[(t.name, t.text) for t in doc.tags
                if t.name.startswith("custom:")])


def from_pair(pair: dict) -> Comment:
    """The corpus's own record of a comment, as fields."""
    return Comment(
        notice=pair.get("notice", "") or "",
        dev=pair.get("dev", "") or "",
        params=dict(pair.get("params") or {}),
        returns=[" ".join(x for x in (r.get("name") or "", r.get("text") or "")
                          if x).strip()
                 for r in (pair.get("returns") or [])],
        inheritdoc=pair.get("inheritdoc"))


def detect_style(src: str) -> str:
    """`line` or `block`, whichever the file already uses more.

    Ties go to `line`: `///` is the more common convention and the safer
    default for a file with no existing documentation at all.
    """
    spans = scan(src, strict=False)
    line = sum(1 for s in spans if s.kind is Kind.DOC_LINE)
    block = sum(1 for s in spans if s.kind is Kind.DOC_BLOCK)
    # A block comment is one span covering many lines; a /// run is one span
    # per line. Count a block as the number of lines it holds, so the
    # comparison is like for like.
    if block:
        block = sum(len(s.text(src).splitlines())
                    for s in spans if s.kind is Kind.DOC_BLOCK)
    return "block" if block > line else "line"


def _each(value) -> List[str]:
    """One tag per paragraph, in order."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [v for v in value if v]


def _tag_lines(c: Comment, param_order: Sequence[str],
               return_count: int, return_names: Sequence[Optional[str]]
               ) -> List[str]:
    out: List[str] = []

    if c.inheritdoc:  # noqa: PLR0911
        # Exclusive by design: solc inherits every tag from the base, so a
        # @notice here would be a second, conflicting source. @dev is the one
        # exception — an override's implementation notes are its own.
        out.append(f"@inheritdoc {c.inheritdoc}")
        for text in _each(c.dev):
            out.append(f"@dev {text}")
        return out

    if c.preamble:
        out.append(c.preamble)        # untagged, exactly as it was written
    if c.title:
        out.append(f"@title {c.title}")
    if c.author:
        out.append(f"@author {c.author}")
    for text in _each(c.notice):
        out.append(f"@notice {text}")
    for text in _each(c.dev):
        out.append(f"@dev {text}")

    # Declaration order, never the model's order. A parameter the declaration
    # does not have is dropped: solc rejects the file outright for it.
    for name in param_order:
        text = c.params.get(name)
        if text:
            out.append(f"@param {name} {text}")

    for i in range(return_count):
        if i >= len(c.returns) or not c.returns[i]:
            continue
        text = c.returns[i]
        name = return_names[i] if i < len(return_names) else None
        # A named return must lead with its name, or solc reads the first word
        # as the name and the rest as the description.
        if name and not text.split(" ", 1)[0] == name:
            text = f"{name} {text}"
        out.append(f"@return {text}")

    for key, text in c.custom:
        k = key if key.startswith("custom:") else f"custom:{key}"
        out.append(f"@{k} {text}")
    return out


def render(c: Comment, *, param_order: Sequence[str] = (),
           return_count: int = 0,
           return_names: Sequence[Optional[str]] = (),
           style: str = "line", indent: str = "",
           width: int = DEFAULT_WIDTH) -> str:
    """The comment as it should appear above the declaration."""
    lines = _tag_lines(c, param_order, return_count, return_names)
    if not lines:
        return ""

    if style == "block":
        prefix, opener, closer = f"{indent} * ", f"{indent}/**", f"{indent} */"
    else:
        prefix, opener, closer = f"{indent}/// ", None, None

    body: List[str] = []
    for tag in lines:
        wrapped = textwrap.wrap(tag, width=max(width - len(prefix), 20),
                                break_long_words=False,
                                break_on_hyphens=False) or [tag]
        body.extend(prefix + w for w in wrapped)

    if style == "block":
        return "\n".join([opener, *body, closer])
    return "\n".join(body)


def render_for(c: Comment, decl, *, style: str = "line", indent: str = "",
               width: int = DEFAULT_WIDTH) -> str:
    """Render against a parsed declaration, which supplies the orders."""
    return render(c,
                  param_order=[p.name for p in decl.params if p.name],
                  return_count=len(decl.returns),
                  return_names=[p.name for p in decl.returns],
                  style=style, indent=indent, width=width)


#: Tags models reach for that solc does not accept. `@returns` is a plain
#: typo for `@return`; the rest have no NatSpec equivalent and solc rejects
#: the file outright, so they are dropped rather than guessed at.
_ALIASES = {"returns": "return", "params": "param", "parameter": "param"}
_ALIAS_RE = re.compile(r"(^|\n)(\s*(?:///?|\*)?\s*)@(" +
                       "|".join(_ALIASES) + r")\b")


def normalise(natspec: str, *, style: str = "line", indent: str = "",
              width: int = DEFAULT_WIDTH) -> str:
    """Whatever the model returned, as a comment solc will actually read.

    Models return NatSpec *content* — `@notice ...` with no comment markers at
    all, or with `//`, which solc ignores completely. Splicing that into a
    source file does not produce an undocumented function, it produces a
    syntax error: in one run 71 of 110 refined comments failed to compile and
    77 emitted no tags, entirely because nobody turned the text into a
    comment. Every downstream measure — the gate, tag emission, parameter
    coverage, claim support — was measuring formatting rather than content.

    This reformats and nothing else. It never adds a tag, never invents a
    description, and drops only tags solc would reject anyway.
    """
    from .natspec import parse as parse_doc
    text = _ALIAS_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}@{_ALIASES[m.group(3)]}",
                         natspec or "")
    c = from_doc(parse_doc(text))
    # `render` emits only the parameters named in `param_order` and only
    # `return_count` returns, so the orders have to come from the comment
    # itself. Taking them from the declaration instead would quietly inject
    # the gold parameter list into a generated comment — which would make
    # parameter coverage measure the corpus rather than the model.
    return render(c,
                  param_order=list(c.params),
                  return_count=len(c.returns),
                  return_names=[None] * len(c.returns),
                  style=style, indent=indent, width=width)


_TAG = re.compile(r"^\s*@([A-Za-z][A-Za-z0-9:_-]*)")


def tags_in(natspec: str) -> List[str]:
    """Tag names present, in order. Used to check what a render produced."""
    from .natspec import strip_markers
    out = []
    for line in strip_markers(natspec):
        m = _TAG.match(line)
        if m:
            out.append(m.group(1))
    return out
