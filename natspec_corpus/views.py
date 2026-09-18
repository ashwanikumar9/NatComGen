"""The three retrieval views of a function.

SAGE retrieves over code, AST and CFG separately and fuses the results. The
three views must be built the same way for an indexed training function and
for a query function, or the similarity measures the difference between the
two pipelines rather than between the two functions. That is why the AST view
drops the NatSpec node: a documented training function carries one and a
function awaiting documentation never does.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

_WS = re.compile(r"\s+")


def code_view(pair: Dict) -> str:
    """φ_code — the function as written, with every comment removed.

    The pair's `code` field starts at the declaration, so the doc comment
    above it is already excluded. Comments *inside* the body are stripped here
    as well, for two reasons: the generator is shown comment-free source, so
    the query view must be comment-free to match; and otherwise one body
    comment can match another body comment instead of the code, which is a
    similarity between two authors' habits rather than between two functions.
    """
    src = pair.get("code", "")
    if not src:
        return ""
    try:
        from .masking import Kind, scan
        keep = []
        cursor = 0
        for sp in scan(src, strict=False):
            if sp.kind in (Kind.LINE_COMMENT, Kind.DOC_LINE,
                           Kind.BLOCK_COMMENT, Kind.DOC_BLOCK):
                keep.append(src[cursor:sp.start])
                cursor = sp.end
        keep.append(src[cursor:])
        src = " ".join(keep)
    except Exception:                       # noqa: BLE001 - never fail a view
        pass
    return _WS.sub(" ", src).strip()


def ast_view(table: Optional[Dict]) -> str:
    """φ_AST — the pre-order node-type sequence, node types only.

    Identifiers are deliberately absent. A syntactic view exists to match
    functions that share a shape while sharing no names; leaving the names in
    makes it a worse copy of the code view, which is the usual reason a
    multi-view retriever's views turn out to be correlated.
    """
    if not table:
        return ""
    return " ".join(table.get("ast_types", []))


def cfg_view(table: Optional[Dict]) -> str:
    """φ_CFG — execution shape: node types in path order, branch labels, and
    the kinds of call made. Again no identifiers, for the same reason."""
    if not table:
        return ""
    ntype = {n["id"]: n["type"] for n in table.get("nodes", [])}
    label = {(e["from"], e["to"]): e.get("label", "")
             for e in table.get("edges", [])}
    out: List[str] = []
    for p in table.get("paths", []):
        nodes = p["nodes"]
        for a, b in zip(nodes, nodes[1:]):
            out.append(ntype.get(a, "?"))
            lab = label.get((a, b), "")
            if lab:
                out.append(f"[{lab}]")
        if nodes:
            out.append(ntype.get(nodes[-1], "?"))
        out.append("|")
    if not out:                       # no enumerated path: fall back to nodes
        out = [n["type"] for n in table.get("nodes", [])]
    out += [f"call:{c['kind']}" for c in table.get("calls", [])]
    out += [f"revert:{r['kind']}" for r in table.get("reverts", [])]
    return " ".join(out)


VIEW_NAMES = ("code", "ast", "cfg")


def views_for(pair: Dict, table: Optional[Dict]) -> Dict[str, str]:
    return {"code": code_view(pair), "ast": ast_view(table),
            "cfg": cfg_view(table)}
