"""Parse a NatSpec comment into tags.

Two rules decide what counts, and both were got wrong in v1:

1. Untagged prose is `@notice`. solc's own documentation-generator treats the
   text before the first tag as the notice, so a comment reading
   `/// Transfers `amount` to `to`` is a documented notice, not an undocumented
   function. v1 required a literal @notice or @dev and threw away real docs.
2. Only the first `@`-word on a line opens a tag. An `@` inside prose (an
   email, a `@dev`-looking word mid-sentence) continues the current tag.
"""
from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional

KNOWN_TAGS = {"title", "author", "notice", "dev", "param", "return",
              "inheritdoc"}

_TAG_RE = re.compile(r"^\s*@([A-Za-z][A-Za-z0-9:_-]*)\s*(.*)$")
_IDENT = re.compile(r"([A-Za-z_$][A-Za-z0-9_$]*)")


class Tag(NamedTuple):
    name: str            # notice | dev | param | return | inheritdoc | custom:x
    arg: Optional[str]   # param name, return name, inheritdoc target
    text: str


class Doc(NamedTuple):
    raw: str
    start: int
    end: int
    tags: List[Tag]

    def of(self, name: str) -> List[Tag]:
        return [t for t in self.tags if t.name == name]

    @property
    def notice(self) -> str:
        return " ".join(t.text for t in self.of("notice")).strip()

    @property
    def dev(self) -> str:
        return " ".join(t.text for t in self.of("dev")).strip()

    @property
    def inheritdoc(self) -> Optional[str]:
        t = self.of("inheritdoc")
        return t[0].arg if t else None

    @property
    def params(self) -> Dict[str, str]:
        return {t.arg: t.text for t in self.of("param") if t.arg}

    @property
    def returns(self) -> List[Tag]:
        return self.of("return")


def strip_markers(raw: str) -> List[str]:
    """Comment text with `///`, `/**`, leading `*` and `*/` removed, one entry
    per line. Interior indentation after the marker is preserved so that a
    continuation line still reads as a continuation."""
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if s.endswith("*/"):              # strip the terminator FIRST, so the
            s = s[:-2].rstrip()           # ` */` line does not leave a stray /
        if s.startswith("/**"):
            s = s[3:]
        elif s.startswith("///"):
            s = s[3:]
        elif s.startswith("//"):
            s = s[2:]
        elif s.startswith("*"):
            s = s[1:]
        lines.append(s.rstrip())
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def parse(raw: str, start: int = 0, end: int = 0) -> Doc:
    lines = strip_markers(raw)
    tags: List[Tag] = []
    cur_name, cur_arg, buf = None, None, []

    def flush() -> None:
        if cur_name is None and not any(b.strip() for b in buf):
            return
        text = " ".join(b.strip() for b in buf if b.strip()).strip()
        name = cur_name or "notice"          # untagged prose is the notice
        if cur_name is None and not text:
            return
        # Mark prose that carried no tag. solc folds it into the notice, but
        # it concatenates the two with no separator at all, so re-emitting it
        # as an explicit @notice changes what solc reads. Recording which one
        # it was lets the emitter put it back exactly where it started.
        arg = "implicit" if cur_name is None else cur_arg
        tags.append(Tag(name=name, arg=arg, text=text))

    for line in lines:
        m = _TAG_RE.match(line)
        if m:
            flush()
            cur_name, rest = m.group(1), m.group(2)
            cur_arg = None
            if cur_name in ("param", "inheritdoc"):
                im = _IDENT.match(rest.strip())
                if im:
                    cur_arg = im.group(1)
                    rest = rest.strip()[im.end():].strip()
            elif cur_name == "return":
                # The first word of an @return is the return variable's name
                # only when the function actually declares that name. natspec
                # cannot know that, so it records the candidate and leaves the
                # text intact; build.py resolves it against the declaration.
                im = _IDENT.match(rest.strip())
                if im:
                    cur_arg = im.group(1)
            buf = [rest]
        else:
            buf.append(line)
    flush()
    return Doc(raw=raw, start=start, end=end, tags=tags)
