"""Structure over the *code view*, never over raw source.

Every function here assumes its input has had comments and string interiors
blanked by masking.code_view(), so a brace, paren or semicolon it sees is
really code. Offsets are preserved by the masker, so every offset returned
here indexes the original source.
"""
from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional, Tuple

from .errors import ParseError

# --------------------------------------------------------------------------
# balanced-delimiter primitives
# --------------------------------------------------------------------------

_PAIRS = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {v: k for k, v in _PAIRS.items()}


def match_delim(s: str, open_idx: int) -> int:
    """Index of the delimiter closing the one at `open_idx` (exclusive end).

    v1 used a non-greedy regex here and truncated
    `mapping(int16 => uint256) storage self` at the first `)`.
    """
    if open_idx >= len(s) or s[open_idx] not in _PAIRS:
        raise ParseError(f"no opening delimiter at offset {open_idx}")
    depth = 0
    for i in range(open_idx, len(s)):
        c = s[i]
        if c in _PAIRS:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
            if depth == 0:
                return i + 1
    raise ParseError(f"unbalanced delimiter opened at offset {open_idx}")


def paren_group(s: str, open_idx: int) -> Tuple[str, int]:
    """(inner text, index just past the closing delimiter)."""
    end = match_delim(s, open_idx)
    return s[open_idx + 1:end - 1], end


def split_top(s: str, sep: str = ",") -> List[str]:
    """Split on `sep` only at nesting depth zero."""
    out, depth, buf = [], 0, []
    for c in s:
        if c in _PAIRS:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        if c == sep and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    out.append("".join(buf))
    return [p.strip() for p in out if p.strip()]


def scan_top(s: str, start: int, stops: str) -> int:
    """First index >= start where a char from `stops` appears at depth zero.
    Returns len(s) if none. Used to find the `{` or `;` ending a header."""
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if depth == 0 and c in stops:      # stop chars win over nesting, so a
            return i                       # `{` may itself be the stop char
        if c in _PAIRS:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
    return len(s)


# --------------------------------------------------------------------------
# parameter lists
# --------------------------------------------------------------------------

_LOCATIONS = {"memory", "storage", "calldata"}
_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*\Z")


class Param(NamedTuple):
    type: str
    name: Optional[str]      # None for an unnamed parameter
    raw: str


def parse_param(raw: str) -> Param:
    """Split one declaration into type and name.

    The name is the trailing identifier, but only when it is *not* part of the
    type: `uint256` alone is a type with no name, `mapping(a => b) storage self`
    has type `mapping(a => b) storage`, and a function-type parameter
    `function (uint) external returns (uint) cb` ends in `cb` after a returns
    group. Working right-to-left from the end of the string handles all three.
    """
    raw = raw.strip()
    if not raw:
        raise ParseError("empty parameter")
    # a trailing returns(...) group belongs to a function type, never a name
    tail = raw
    m = _IDENT.search(tail)
    if not m:
        return Param(type=raw, name=None, raw=raw)
    name = m.group(0)
    head = tail[:m.start()].rstrip()
    if not head:                                  # only a type, e.g. `uint256`
        return Param(type=raw, name=None, raw=raw)
    if name in _LOCATIONS or name in ("payable", "external", "internal",
                                      "public", "private", "pure", "view",
                                      "memory", "indexed", "anonymous"):
        return Param(type=raw, name=None, raw=raw)
    if head.endswith("."):
        # A qualified type: `DataTypes.PubType`, `Tick.Info`. The trailing
        # identifier is the second half of the type name, not a parameter
        # name — and taking it as one invents a @param that does not exist
        # and truncates the type to `DataTypes.`
        return Param(type=raw, name=None, raw=raw)
    if head.endswith(")") and not head.endswith("returns)"):
        # `mapping(K => V)` with no location and no name  -> unnamed
        # but `Foo.Bar storage x` reaches here only via head not ending in )
        if re.search(r"\bmapping\s*\(", head) and not re.search(
                r"\b(memory|storage|calldata)\b", head):
            return Param(type=raw, name=None, raw=raw)
    return Param(type=head, name=name, raw=raw)


def parse_params(inner: str) -> List[Param]:
    return [parse_param(p) for p in split_top(inner)]


def type_key(p: Param) -> str:
    """Canonical type for signature comparison: whitespace squeezed, data
    location and `payable` dropped (they do not participate in overloading)."""
    t = re.sub(r"\s+", " ", p.type).strip()
    t = re.sub(r"\b(memory|storage|calldata|payable)\b", " ", t)
    return re.sub(r"\s+", "", t)


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------

CONTAINER_KEYWORDS = ("contract", "interface", "library", "abstract contract")
_CONTAINER_RE = re.compile(
    r"\b(?:(abstract)\s+)?(contract|interface|library)\s+"
    r"([A-Za-z_$][A-Za-z0-9_$]*)")
_CALLABLE_RE = re.compile(
    r"\b(function|constructor|fallback|receive|modifier|event|error)\b")
_TYPEDECL_RE = re.compile(r"\b(struct|enum)\s+([A-Za-z_$][A-Za-z0-9_$]*)")

_VISIBILITY = ("external", "public", "internal", "private")
_MUTABILITY = ("pure", "view", "payable")


class Container(NamedTuple):
    kind: str            # contract | interface | library
    name: str
    abstract: bool
    header_start: int
    body_start: int      # index of `{`
    body_end: int        # index just past `}`
    bases: List[str]


class Decl(NamedTuple):
    kind: str            # function | constructor | fallback | receive |
                         # modifier | event | error
    name: str            # "" for constructor/fallback/receive
    container: Optional[str]
    container_kind: Optional[str]
    header_start: int    # start of the keyword
    header_end: int      # index of `{` or `;`
    body_end: Optional[int]
    header: str          # raw header text from the ORIGINAL source
    params: List[Param]
    returns: List[Param]
    visibility: Optional[str]
    mutability: Optional[str]
    is_virtual: bool
    overrides: bool

    @property
    def sig(self) -> str:
        n = self.name or self.kind
        return f"{n}({','.join(type_key(p) for p in self.params)})"

    @property
    def qualified(self) -> str:
        return f"{self.container or '<file>'}.{self.sig}"

    @property
    def arity_key(self) -> str:
        return f"{self.name or self.kind}/{len(self.params)}"


def find_containers(code: str, src: str) -> List[Container]:
    out = []
    for m in _CONTAINER_RE.finditer(code):
        brace = scan_top(code, m.end(), "{;")
        if brace >= len(code) or code[brace] != "{":
            continue                                    # forward declaration
        header = code[m.end():brace]
        bases = []
        bm = re.search(r"\bis\b(.*)", header, re.S)
        if bm:
            for b in split_top(bm.group(1)):
                bn = re.match(r"([A-Za-z_$][A-Za-z0-9_$]*)", b.strip())
                if bn:
                    bases.append(bn.group(1))
        try:
            end = match_delim(code, brace)
        except ParseError:
            end = len(code)
        out.append(Container(kind=m.group(2), name=m.group(3),
                             abstract=bool(m.group(1)),
                             header_start=m.start(), body_start=brace,
                             body_end=end, bases=bases))
    return out


def _container_at(containers: List[Container], off: int) -> Optional[Container]:
    best = None
    for c in containers:
        if c.body_start < off < c.body_end:
            if best is None or c.body_start > best.body_start:
                best = c
    return best


def find_decls(code: str, src: str,
               containers: Optional[List[Container]] = None) -> List[Decl]:
    """All callable declarations, in source order."""
    if containers is None:
        containers = find_containers(code, src)
    out: List[Decl] = []
    for m in _CALLABLE_RE.finditer(code):
        kw = m.group(1)
        # `function` also appears as a *type* (`function(uint) external`);
        # a type is never followed by `(` after an identifier at header level
        # and never terminates in `{`/`;` at depth 0 of its own.
        pos = m.end()
        name = ""
        if kw in ("function", "modifier", "event", "error"):
            nm = re.match(r"\s*([A-Za-z_$][A-Za-z0-9_$]*)", code[pos:])
            if nm:
                name = nm.group(1)
                pos += nm.end()
            elif kw == "function":
                continue                        # function type, not a decl
        op = code.find("(", pos)
        if op == -1:
            continue
        between = code[pos:op]
        if between.strip():                     # something odd between; skip
            continue
        try:
            inner, after = paren_group(code, op)
        except ParseError:
            continue
        stop = scan_top(code, after, "{;")
        header_end = min(stop, len(code))
        tail = code[after:header_end]
        rets: List[Param] = []
        rm = re.search(r"\breturns\b", tail)
        if rm:
            rop = code.find("(", after + rm.end() - 1)
            if rop != -1 and rop < header_end:
                rinner, _ = paren_group(code, rop)
                try:
                    rets = parse_params(rinner)
                except ParseError:
                    rets = []
        vis = next((v for v in _VISIBILITY if re.search(rf"\b{v}\b", tail)), None)
        mut = next((v for v in _MUTABILITY if re.search(rf"\b{v}\b", tail)), None)
        body_end = None
        if header_end < len(code) and code[header_end] == "{":
            try:
                body_end = match_delim(code, header_end)
            except ParseError:
                body_end = None
        ctr = _container_at(containers, m.start())
        try:
            params = parse_params(inner)
        except ParseError as e:
            raise ParseError(f"{name or kw} at {m.start()}: {e}") from e
        out.append(Decl(
            kind=kw, name=name,
            container=ctr.name if ctr else None,
            container_kind=ctr.kind if ctr else None,
            header_start=m.start(), header_end=header_end, body_end=body_end,
            header=src[m.start():header_end].strip(),
            params=params, returns=rets,
            visibility=vis, mutability=mut,
            is_virtual=bool(re.search(r"\bvirtual\b", tail)),
            overrides=bool(re.search(r"\boverride\b", tail)),
        ))
    return out


_IMPORT_RE = re.compile(r"""import\s+(?:[^;'"]*?\bfrom\s*)?["']([^"']+)["']""")


def imports(code: str, src: str) -> List[str]:
    """Import paths. Run on the *original* source with code-view offsets so
    the quoted path survives (code_view blanks string interiors)."""
    return [m.group(1) for m in _IMPORT_RE.finditer(src)]


_PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);")


def pragma(code: str) -> Optional[str]:
    m = _PRAGMA_RE.search(code)
    return m.group(1).strip() if m else None
