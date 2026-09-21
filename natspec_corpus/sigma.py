"""Σ(f): the deterministic fact table a function is described from.

Every row carries a stable id — `F*` facts, `N*` CFG nodes, `P*` paths, `C*`
call edges, `D*` data-dependency rows — because the prompts require the model
to cite the evidence for each claim it makes, and a citation is only checkable
if the thing cited has a name.

Slither does the analysis. Hand-rolling a CFG from the solc AST means
reimplementing modifier splicing, `require` as a branch, loop back-edges,
low-level calls and inline assembly, each with its own long tail of bugs.
Slither already has all of it, plus variable-level data dependency, and it is
the tool reviewers will expect.

Scope of the data dependency, decided deliberately: intra-procedural def-use
inside one function, plus the storage read and write sets. Full
inter-procedural, alias-aware dependency analysis is a research contribution
of its own and is not what this project is about.

One trap worth naming. Slither reports source positions as **byte** offsets;
the corpus records **character** offsets, because it is built in Python. The
two agree for ASCII and diverge the moment a file contains a non-ASCII
comment or a `unicode"…"` literal — silently, by a few positions, which is
the worst kind of wrong. Every offset crossing this boundary goes through
`OffsetMap`.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .errors import CorpusError


class SigmaError(CorpusError):
    """Σ(f) could not be produced for a unit."""


_IR_TEMP = re.compile(r"^(TMP|REF|CONST)_\d+$")

MAX_PATHS = 16          # enumerated acyclic paths per function
MAX_PATH_NODES = 64     # guard against pathological CFGs


# --------------------------------------------------------------------------
# byte <-> character offsets
# --------------------------------------------------------------------------

class OffsetMap:
    """Converts between byte offsets (what solc and Slither report) and
    character offsets (what the corpus records).

    Built once per source file. For a pure-ASCII file the two are identical
    and the conversion is a no-op; the class exists so that the one file with
    a unicode comment does not silently shift every offset after it.
    """

    def __init__(self, src: str) -> None:
        self.src = src
        self._ascii = src.isascii()
        if self._ascii:
            self._byte_at = None
            return
        # byte offset of each character index, plus the end
        acc, out = 0, []
        for ch in src:
            out.append(acc)
            acc += len(ch.encode("utf-8"))
        out.append(acc)
        self._byte_at = out

    def to_char(self, byte_off: int) -> int:
        if self._ascii:
            return byte_off
        i = bisect.bisect_right(self._byte_at, byte_off) - 1
        return max(0, min(i, len(self.src)))

    def to_byte(self, char_off: int) -> int:
        if self._ascii:
            return char_off
        return self._byte_at[max(0, min(char_off, len(self.src)))]


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------

@dataclass
class FactTable:
    file: str
    contract: Optional[str]
    function: str
    canonical_name: str
    char_start: int
    char_end: int
    pair_id: Optional[str] = None
    facts: List[dict] = field(default_factory=list)
    nodes: List[dict] = field(default_factory=list)
    edges: List[dict] = field(default_factory=list)
    paths: List[dict] = field(default_factory=list)
    calls: List[dict] = field(default_factory=list)
    deps: List[dict] = field(default_factory=list)
    reverts: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    ast_types: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file": self.file, "contract": self.contract,
            "function": self.function, "canonical_name": self.canonical_name,
            "char_start": self.char_start, "char_end": self.char_end,
            "pair_id": self.pair_id,
            "facts": self.facts, "nodes": self.nodes, "edges": self.edges,
            "paths": self.paths, "calls": self.calls, "deps": self.deps,
            "reverts": self.reverts, "events": self.events,
            "ast_types": self.ast_types,
        }

    @property
    def row_ids(self) -> List[str]:
        return ([r["id"] for r in self.facts] + [r["id"] for r in self.nodes]
                + [r["id"] for r in self.paths] + [r["id"] for r in self.calls]
                + [r["id"] for r in self.deps] + [r["id"] for r in self.reverts]
                + [r["id"] for r in self.events])


def _name(v: Any) -> str:
    return getattr(v, "name", None) or str(v)


def _names(vs: Iterable[Any]) -> List[str]:
    """Deduplicated and **sorted**.

    Slither hands several of these back as sets, whose iteration order varies
    with PYTHONHASHSEED between processes. Two runs of the same build then
    differ only in the order of a node's `reads` list — which is invisible in
    review, breaks byte-level reproducibility, and would make any cache keyed
    on the fact table miss at random. These lists are set-like in meaning, so
    sorting costs nothing.
    """
    return sorted({_name(v) for v in vs if _name(v)})


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def _facts_for(fn: Any) -> List[dict]:
    """F* rows: the properties a comment may assert without reading the body."""
    rows: List[dict] = []

    def add(kind: str, text: str, **extra) -> None:
        rows.append({"id": f"F{len(rows) + 1}", "kind": kind, "text": text,
                     **extra})

    add("kind", getattr(fn, "function_type", None).name.lower()
        if getattr(fn, "function_type", None) is not None else "function")
    add("visibility", str(getattr(fn, "visibility", "") or "internal"))
    if getattr(fn, "payable", False):
        add("mutability", "payable")
    elif getattr(fn, "pure", False):
        add("mutability", "pure")
    elif getattr(fn, "view", False):
        add("mutability", "view")
    for p in getattr(fn, "parameters", []) or []:
        add("parameter", f"{p.type} {p.name}".strip(), name=p.name,
            type=str(p.type))
    for r in getattr(fn, "returns", []) or []:
        add("return", f"{r.type} {r.name}".strip() if r.name else str(r.type),
            name=r.name or None, type=str(r.type))
    for m in getattr(fn, "modifiers", []) or []:
        add("modifier", _name(m))
    reads = _names(getattr(fn, "state_variables_read", []) or [])
    writes = _names(getattr(fn, "state_variables_written", []) or [])
    if reads:
        add("state_read", ", ".join(reads), variables=reads)
    if writes:
        add("state_written", ", ".join(writes), variables=writes)
    if getattr(fn, "is_constructor", False):
        add("role", "constructor")
    if getattr(fn, "view", False) is False and not writes and not reads:
        add("purity", "touches no state")
    return rows


def _nodes_for(fn: Any, omap: OffsetMap) -> Tuple[List[dict], Dict[int, str],
                                                  List[dict]]:
    """N* rows and the CFG edges between them."""
    rows: List[dict] = []
    ident: Dict[int, str] = {}
    for i, n in enumerate(getattr(fn, "nodes", []) or [], start=1):
        nid = f"N{i}"
        ident[id(n)] = nid
        sm = getattr(n, "source_mapping", None)
        start = omap.to_char(sm.start) if sm else -1
        end = omap.to_char(sm.start + sm.length) if sm else -1
        rows.append({
            "id": nid,
            "type": getattr(getattr(n, "type", None), "name", "UNKNOWN"),
            "src": [start, end],
            "text": (str(n.expression) if getattr(n, "expression", None)
                     else ""),
            "reads": _names(getattr(n, "variables_read", []) or []),
            "writes": _names(getattr(n, "variables_written", []) or []),
            "state_reads": _names(getattr(n, "state_variables_read", []) or []),
            "state_writes": _names(getattr(n, "state_variables_written", [])
                                   or []),
        })

    edges: List[dict] = []
    for n in getattr(fn, "nodes", []) or []:
        src = ident.get(id(n))
        if src is None:
            continue
        t, f = getattr(n, "son_true", None), getattr(n, "son_false", None)
        labelled = {id(t): "true", id(f): "false"}
        for s in getattr(n, "sons", []) or []:
            dst = ident.get(id(s))
            if dst is None:
                continue
            edges.append({"from": src, "to": dst,
                          "label": labelled.get(id(s), "")})
    return rows, ident, edges


def _paths_for(fn: Any, ident: Dict[int, str]) -> List[dict]:
    """P* rows: acyclic paths from entry to a terminal node.

    Capped, because a function with a dozen branches has thousands of paths
    and the prompt can only cite a handful. The cap is recorded on the row so
    a downstream consumer knows the enumeration was truncated rather than
    complete."""
    entry = getattr(fn, "entry_point", None)
    if entry is None:
        return []
    out: List[dict] = []
    truncated = False

    def walk(node: Any, acc: List[str], seen: set) -> None:
        nonlocal truncated
        if len(out) >= MAX_PATHS or len(acc) > MAX_PATH_NODES:
            truncated = True
            return
        nid = ident.get(id(node))
        if nid is None or id(node) in seen:
            return
        acc = acc + [nid]
        seen = seen | {id(node)}
        sons = [s for s in (getattr(node, "sons", []) or [])
                if id(s) not in seen]
        if not sons:
            out.append({"id": f"P{len(out) + 1}", "nodes": acc,
                        "length": len(acc)})
            return
        for s in sons:
            walk(s, acc, seen)

    walk(entry, [], set())
    for r in out:
        r["enumeration_truncated"] = truncated
    return out


def _ir_target(op: Any) -> str:
    """Name a call site's target.

    Slither's shapes differ per collection and per version: `library_calls`
    and `internal_calls` hand back IR operations, while `high_level_calls`
    hands back a `(Contract, Operation)` tuple. Stringifying an operation
    yields the whole SlithIR line — `TMP_63(uint128) = LIBRARY_CALL, dest:…` —
    which is useless in a prompt, so the callee is resolved properly and the
    IR string is never used as a name.
    """
    if isinstance(op, tuple):
        parts = [_ir_target(x) for x in op if x is not None]
        named = [p for p in parts if p and "=" not in p and " " not in p]
        return named[-1] if named else (parts[-1] if parts else "?")
    fn = getattr(op, "function", None)
    if fn is not None:
        return (getattr(fn, "canonical_name", None)
                or getattr(fn, "name", None) or str(fn))
    for attr in ("function_name", "destination", "name"):
        v = getattr(op, attr, None)
        if v is not None:
            return _name(v)
    return _name(op)


_CALL_PRIORITY = {"library": 0, "builtin": 1, "external": 2,
                  "internal": 3, "low_level": 4}


def _calls_for(fn: Any, ident: Dict[int, str]) -> List[dict]:
    """C* rows: one per call site, tagged by kind.

    Two corrections over the naive version. Slither reports the same call
    under more than one heading — a library call is also a high-level call, a
    builtin is also an internal call — so the more specific kind wins and the
    duplicate is dropped; without this every `SafeMath.add` appeared twice,
    once readable and once as raw IR. And these collections are *sets*, whose
    iteration order varies with PYTHONHASHSEED, so rows are sorted within each
    node before ids are assigned: otherwise two runs of the same build produce
    the same facts under different C-numbers, and a prompt that cites C2 means
    something different depending on the run.
    """
    rows: List[dict] = []
    for idx, n in enumerate(getattr(fn, "nodes", []) or []):
        nid = ident.get(id(n))
        lib = {_ir_target(c) for c in (getattr(n, "library_calls", []) or [])}
        builtin = {_ir_target(c)
                   for c in (getattr(n, "solidity_calls", []) or [])}
        here: List[Tuple[str, str]] = []
        for c in getattr(n, "library_calls", []) or []:
            here.append(("library", _ir_target(c)))
        for c in getattr(n, "solidity_calls", []) or []:
            here.append(("builtin", _ir_target(c)))
        for c in getattr(n, "high_level_calls", []) or []:
            t = _ir_target(c)
            if t not in lib:
                here.append(("external", t))
        for c in getattr(n, "internal_calls", []) or []:
            t = _ir_target(c)
            if t not in builtin and t not in lib:
                here.append(("internal", t))
        for c in getattr(n, "low_level_calls", []) or []:
            here.append(("low_level", _ir_target(c)))
        for kind, target in sorted(
                here, key=lambda kt: (_CALL_PRIORITY.get(kt[0], 9), kt[1])):
            rows.append({"id": f"C{len(rows) + 1}", "kind": kind,
                         "target": target, "node": nid})
    return rows


def _transitive(direct: Dict[str, set]) -> Dict[str, set]:
    """Least fixpoint of `direct`, computed with a worklist.

    Slither's own dependency table is not a fixpoint: it propagates over
    collections whose iteration order varies with PYTHONHASHSEED, so two runs
    of the same analysis disagree about whether `amountOut` depends on
    `feePips`. Closing the relation here makes the answer both reproducible
    and more complete — a closure is order-independent by construction.
    """
    closed = {k: set(v) for k, v in direct.items()}
    changed = True
    while changed:
        changed = False
        for k in list(closed):
            grown = set(closed[k])
            for s2 in list(closed[k]):
                grown |= closed.get(s2, set())
            grown.discard(k)
            if grown != closed[k]:
                closed[k] = grown
                changed = True
    return closed


def _deps_for(fn: Any) -> List[dict]:
    """D* rows: which values each variable in this function derives from.

    Restricted to variables in the function's own scope — parameters, locals,
    returns and the state it touches — so a row is always something a comment
    could plausibly mention. The restriction is applied *after* the closure,
    not before: a chain `total <- TMP_5 <- amount` only reaches `amount`
    through a SlithIR temporary, and filtering first would cut it.
    """
    try:
        from slither.analyses.data_dependency.data_dependency import (
            get_all_dependencies)
    except Exception:
        return []
    try:
        table = get_all_dependencies(fn)
    except Exception:
        return []

    scope = set()
    for group in ("parameters", "returns", "variables",
                  "state_variables_read", "state_variables_written"):
        for v in getattr(fn, group, []) or []:
            n = _name(v)
            if n:
                scope.add(n)

    direct = {}
    for var, sources in table.items():
        name = _name(var)
        if name:
            direct.setdefault(name, set()).update(
                n for n in (_name(s) for s in sources) if n)
    closed = _transitive(direct)

    rows: List[dict] = []
    for name in sorted(closed):
        if name not in scope or _IR_TEMP.match(name):
            continue
        srcs = sorted((closed[name] & scope) - {name})
        if not srcs:
            continue
        rows.append({"id": f"D{len(rows) + 1}", "variable": name,
                     "depends_on": srcs})
    return rows



_GUARDS = ("require", "revert", "assert")


def _reverts_for(fn: Any, ident: Dict[int, str]) -> List[dict]:
    """R* rows: the conditions under which the function reverts.

    A NatSpec `@dev` is expected to state caller obligations, and the only
    honest source for "only the owner may call this" is a guard, never a
    modifier's name. These rows are what the writing rules point at when they
    say a name is not a fact.

    Read off the node expression rather than the IR, because the expression
    keeps the condition as written — `amount > 0` — while the IR has already
    lowered it to a temporary.
    """
    rows: List[dict] = []
    for n in getattr(fn, "nodes", []) or []:
        expr = str(getattr(n, "expression", "") or "")
        ntype = getattr(getattr(n, "type", None), "name", "")
        head = expr.strip()
        kind = next((g for g in _GUARDS if head.startswith(g)), None)
        if kind is None and ntype != "THROW":
            continue
        # Slither prints the lowered form `require(bool,string)(amount > 0,zero)`
        # — the FIRST paren group is solc's type signature, not the arguments.
        # Taking it would record every condition as the word "bool".
        from .solidity import split_top
        args = ""
        depth, open_at = 0, None
        for i, ch in enumerate(head):
            if ch == "(":
                if depth == 0:
                    open_at = i
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and open_at is not None:
                    args = head[open_at + 1:i]          # keep the LAST group
        parts = split_top(args) if args else []
        condition = parts[0] if parts else head
        message = parts[-1].strip('"') if len(parts) > 1 else ""
        rows.append({"id": f"R{len(rows) + 1}", "kind": kind or "throw",
                     "condition": condition, "message": message,
                     "node": ident.get(id(n)), "text": head})
    return rows


def _events_for(fn: Any, ident: Dict[int, str]) -> List[dict]:
    """E* rows: events emitted, with their arguments."""
    try:
        from slither.slithir.operations import EventCall
    except Exception:
        return []
    rows: List[dict] = []
    for n in getattr(fn, "nodes", []) or []:
        for ir in getattr(n, "irs", []) or []:
            if isinstance(ir, EventCall):
                rows.append({"id": f"E{len(rows) + 1}",
                             "event": str(getattr(ir, "name", "") or ""),
                             "args": [str(a) for a in
                                      (getattr(ir, "arguments", []) or [])],
                             "node": ident.get(id(n))})
    return rows


def _interaction_order(nodes: List[dict], calls: List[dict],
                       paths: List[dict]) -> Optional[dict]:
    """Does a state write happen after an external call on any path?

    The checks-effects-interactions question, answered from the CFG rather
    than asserted. It is the single most useful thing a `@dev` can say about a
    state-changing function, and it is exactly the kind of claim a model will
    invent if the evidence does not contain it.
    """
    external = {c["node"] for c in calls
                if c["kind"] in ("external", "low_level") and c["node"]}
    if not external:
        return None
    writes = {n["id"] for n in nodes if n["state_writes"]}
    if not writes:
        return {"id": "F_CEI", "kind": "interaction_order",
                "text": "makes an external call and writes no state"}
    for p in paths:
        seen_ext = False
        for nid in p["nodes"]:
            if nid in external:
                seen_ext = True
            elif seen_ext and nid in writes:
                return {"id": "F_CEI", "kind": "interaction_order",
                        "text": (f"writes state at {nid} after an external "
                                 f"call on path {p['id']} — the callee can "
                                 f"re-enter before that write")}
    return {"id": "F_CEI", "kind": "interaction_order",
            "text": "every state write precedes the external calls on all "
                    "enumerated paths"}


AST_TYPE_CAP = 400


def _src_start(node: dict) -> Optional[int]:
    """solc writes `src` as "<byteOffset>:<length>:<fileIndex>"."""
    src = node.get("src")
    if not isinstance(src, str):
        return None
    head = src.split(":", 1)[0]
    return int(head) if head.isdigit() else None


def ast_functions(ast_root: dict) -> Dict[int, dict]:
    """Byte offset -> declaration node, for every callable in one file's AST.

    The key is the same byte offset Slither reports, so a function's Slither
    object and its AST subtree are matched by integer equality rather than by
    name — which matters for overloads, where the name is not unique.
    """
    out: Dict[int, dict] = {}
    stack = [ast_root]
    wanted = {"FunctionDefinition", "ModifierDefinition", "EventDefinition",
              "ErrorDefinition"}
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if node.get("nodeType") in wanted:
            off = _src_start(node)
            if off is not None:
                out.setdefault(off, node)
        for v in node.values():
            if isinstance(v, dict):
                stack.append(v)
            elif isinstance(v, list):
                stack.extend(x for x in v if isinstance(x, dict))
    return out


def flatten_ast(node: dict, cap: int = AST_TYPE_CAP) -> List[str]:
    """Pre-order sequence of AST node types under `node`.

    This is the φ_AST view. Node *types* only, never identifiers: the point of
    a syntactic view is to match functions that share a shape even when they
    share no names, and leaving names in collapses it into a worse copy of the
    code view. Children are visited in source order so the sequence is
    deterministic and reflects the program's actual structure.
    """
    out: List[str] = []

    def walk(n: Any) -> None:
        if len(out) >= cap or not isinstance(n, dict):
            return
        t = n.get("nodeType")
        if t == "StructuredDocumentation":
            return          # the NatSpec node itself. A documented training
                            # function would carry it and a query function
                            # never would, so leaving it in lets the view
                            # separate the two sets on the wrong signal.
        if isinstance(t, str):
            out.append(t)
        kids: List[dict] = []
        for v in n.values():
            if isinstance(v, dict) and "nodeType" in v:
                kids.append(v)
            elif isinstance(v, list):
                kids.extend(x for x in v
                            if isinstance(x, dict) and "nodeType" in x)
        for k in sorted(kids, key=lambda c: _src_start(c) or 0):
            walk(k)

    walk(node)
    return out


def fact_table(fn: Any, rel: str, omap: OffsetMap,
               ast_by_offset: Optional[Dict[int, dict]] = None) -> FactTable:
    """Σ(f) for one Slither function object."""
    sm = getattr(fn, "source_mapping", None)
    start = omap.to_char(sm.start) if sm else -1
    end = omap.to_char(sm.start + sm.length) if sm else -1
    contract = getattr(getattr(fn, "contract_declarer", None), "name", None) \
        or getattr(getattr(fn, "contract", None), "name", None)
    nodes, ident, edges = _nodes_for(fn, omap)
    paths = _paths_for(fn, ident)
    calls = _calls_for(fn, ident)
    facts = _facts_for(fn)
    cei = _interaction_order(nodes, calls, paths)
    if cei is not None:
        facts.append({"id": f"F{len(facts) + 1}", "kind": cei["kind"],
                      "text": cei["text"]})
    ast_types: List[str] = []
    if ast_by_offset and sm is not None:
        node = ast_by_offset.get(sm.start)
        if node is not None:
            ast_types = flatten_ast(node)
    return FactTable(
        file=rel, contract=contract,
        function=getattr(fn, "name", "") or "",
        canonical_name=getattr(fn, "canonical_name", "") or "",
        char_start=start, char_end=end,
        facts=facts, nodes=nodes, edges=edges,
        paths=paths, calls=calls, deps=_deps_for(fn),
        reverts=_reverts_for(fn, ident), events=_events_for(fn, ident),
        ast_types=ast_types)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def analyze(unit, version: str, *, only: Optional[str] = None,
            ast_root: Optional[dict] = None) -> List[FactTable]:
    """Σ(f) for every function in `unit`, or only those declared in `only`.

    The unit's sources are materialised into a temporary directory and Slither
    is run with that as the working directory. crytic-compile insists the
    source paths it sees in solc's output exist on disk, so an in-memory unit
    alone is not enough — and materialising also makes the batch path and the
    single-contract inference path identical, which is the point.
    """
    import os
    import tempfile
    from crytic_compile import CryticCompile
    from crytic_compile.platform.solc_standard_json import SolcStandardJson
    from slither import Slither

    from .compile import solc_path

    cwd = os.getcwd()
    saved_path = os.environ.get("PATH", "")
    saved_version = os.environ.get("SOLC_VERSION")
    tmp = tempfile.TemporaryDirectory()
    try:
        import pathlib
        for rel, content in unit.sources.items():
            p = pathlib.Path(tmp.name) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        os.chdir(tmp.name)
        # crytic-compile resolves `solc` from PATH whatever the `solc=`
        # argument says. On a machine with solc-select installed — an anaconda
        # base, typically — that PATH entry is a shim which consults its OWN
        # store, and fails with "Version '0.7.6' not installed" while thirteen
        # perfectly good compilers sit in this project's .solc directory.
        #
        # Our layout puts exactly one binary, named `solc`, in a directory per
        # version, so prepending that directory makes every lookup — ours or
        # crytic-compile's or the shim's — land on the right compiler.
        # SOLC_VERSION is cleared because the shim reads it and second-guesses
        # the binary it was handed. Both are restored in the finally block:
        # this process may go on to analyse a file pinned to another version.
        binary = solc_path(version)
        os.environ.pop("SOLC_VERSION", None)
        os.environ["PATH"] = os.path.dirname(binary) + os.pathsep + saved_path
        try:
            sl = Slither(CryticCompile(
                SolcStandardJson(target=unit.standard_json()),
                solc=binary))
        except Exception as e:                      # noqa: BLE001
            raise SigmaError(f"{unit.entry}: slither failed: "
                             f"{type(e).__name__}: {e}") from e
    finally:
        os.chdir(cwd)
        os.environ["PATH"] = saved_path
        if saved_version is None:
            os.environ.pop("SOLC_VERSION", None)
        else:
            os.environ["SOLC_VERSION"] = saved_version
        tmp.cleanup()

    target = only or unit.entry
    omap = OffsetMap(unit.sources[target])
    by_offset = ast_functions(ast_root) if ast_root else None
    out: List[FactTable] = []
    seen_offsets = set()
    for contract in sl.contracts:
        # `functions` includes everything inherited, so a contract that
        # implements an interface declared in the same file yields the
        # interface's declaration a second time — same source offset, same
        # everything. `functions_declared` is the set actually written in this
        # contract. The offset guard covers the diamond case, where one
        # declaration is inherited by two contracts.
        declared = (list(getattr(contract, "functions_declared", None)
                         or contract.functions)
                    + list(getattr(contract, "modifiers_declared", None)
                           or getattr(contract, "modifiers", [])))
        for fn in declared:
            sm = getattr(fn, "source_mapping", None)
            if sm is None:
                continue
            fname = str(getattr(sm, "filename", "") or "")
            if target not in fname:
                continue
            if getattr(fn, "name", "").startswith("slitherConstructor"):
                continue                    # synthetic, has no source comment
            if sm.start in seen_offsets:
                continue
            seen_offsets.add(sm.start)
            out.append(fact_table(fn, target, omap, by_offset))
    return out
