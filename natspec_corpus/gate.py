"""The deterministic gate between L3 and L8. No model, no cost.

Two questions, both answerable without asking anything:

  1. Does solc still compile the file with this comment attached, and does it
     actually **emit** the tags? A malformed `@param` is not a compile error —
     solc drops the tag and says nothing. So the check is not "did it compile"
     but "does solc's own devdoc/userdoc contain what the comment claims to
     say". That is the compiler's opinion, not ours.
  2. Is the refined comment's defect score no worse than the draft's? Scored
     by `score.judge`, the same function that labelled the corpus, so the gate
     and the ground truth cannot disagree about what a defect is.

Fail either and the draft is kept. This replaces v2's two ranking judges,
which spent two model calls deciding the same question and were subject to
position bias.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .compile import CompileError, Unit, _run, compile_unit, installed_versions
from .extract import build_file
from .score import judge


@dataclass
class GateResult:
    passed: bool
    compiles: bool
    tags_emitted: bool
    score_kept: bool
    draft_defects: List[str]
    refined_defects: List[str]
    reasons: List[str]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "compiles": self.compiles,
                "tags_emitted": self.tags_emitted,
                "score_kept": self.score_kept,
                "draft_defects": self.draft_defects,
                "refined_defects": self.refined_defects,
                "reasons": self.reasons}


# --------------------------------------------------------------------------
# attaching a comment
# --------------------------------------------------------------------------

def attach(src: str, pair: dict, natspec: str) -> str:
    """Return `src` with the function's doc comment replaced by `natspec`.

    The existing comment is removed span by span using `doc_spans`, not by
    cutting from `doc_start` to `doc_end`: those two can enclose a plain `//`
    line that belongs to nobody, and removing it would change the file in a
    way the generator never asked for.
    """
    spans = pair.get("doc_spans") or []
    out = src
    for lo, hi in sorted(spans, reverse=True):
        out = out[:lo] + out[hi:]
    removed = sum(hi - lo for lo, hi in spans
                  if lo < pair["code_start"])
    at = pair["code_start"] - removed
    line_start = out.rfind("\n", 0, at) + 1
    indent = out[line_start:at]
    indent = indent if not indent.strip() else ""
    body = "\n".join(indent + line.strip()
                     for line in natspec.strip().splitlines() if line.strip())
    return out[:line_start] + body + "\n" + out[line_start:]


def _solc_key(pair: dict) -> str:
    """How solc names this member in devdoc/userdoc.

    Not the same as our signature. solc writes a constructor as the bare word
    `constructor`, a fallback as `fallback()`, and resolves user-defined types
    to their underlying ABI types — `IERC20` becomes `address`. Matching our
    own signature string against solc's key therefore fails for every
    constructor and for any function taking a contract or enum parameter, and
    the gate then reports that solc dropped the tags when it did no such
    thing: a false negative that discards every refinement.
    """
    kind = pair.get("kind", "function")
    if kind == "constructor":
        return "constructor"
    if kind in ("fallback", "receive"):
        return f"{kind}()"
    return pair["signature"]


def _doc_for(out: dict, rel: str, contract: Optional[str],
             sig: str) -> Tuple[dict, dict]:
    contracts = (out.get("contracts") or {}).get(rel, {})
    if contract and contract in contracts:
        entries = [contracts[contract]]
    else:
        entries = list(contracts.values())
    name = sig.split("(", 1)[0]
    arity = sig.count(",") + 1 if "(" in sig and sig[sig.index("(") + 1] != ")" \
        else 0
    for e in entries:
        ud, dd = (e.get("userdoc") or {}), (e.get("devdoc") or {})
        # Methods, errors and events live under three different keys. Looking
        # only in `methods` reports every documented custom error and event as
        # a dropped tag.
        user, dev = {}, {}
        for section in ("methods", "errors", "events"):
            user.update(_flatten(ud.get(section) or {}))
            dev.update(_flatten(dd.get(section) or {}))
        keys = set(user) | set(dev)
        if sig in keys:
            return user.get(sig, {}), dev.get(sig, {})
        # Fall back to name and arity. solc resolves user-defined types to
        # their ABI form, so the exact string often differs while the member
        # is unambiguous.
        cands = [k for k in keys if k.split("(", 1)[0] == name
                 and _arity(k) == arity]
        if len(cands) == 1:
            k = cands[0]
            return user.get(k, {}), dev.get(k, {})
    return {}, {}


def _flatten(section: dict) -> dict:
    """devdoc lists errors and events as arrays (a name can be overloaded);
    methods map straight to an object. Normalise both to name -> object."""
    out = {}
    for k, v in section.items():
        out[k] = v[0] if isinstance(v, list) and v else (
            v if isinstance(v, dict) else {})
    return out


def _arity(key: str) -> int:
    """Top-level parameter count.

    solc expands a struct parameter into its ABI tuple, so `fulfillOrder(Order,
    address)` becomes `fulfillOrder((address,address,(uint8,…)[],…),bytes32)`.
    Counting every comma then reports twenty parameters instead of two, the
    name-and-arity fallback misses, and the gate reports that solc dropped
    every tag on the function.
    """
    if "(" not in key:
        return 0
    from .solidity import split_top
    inner = key[key.index("(") + 1:key.rindex(")")]
    return len(split_top(inner))


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

def solc_emits(unit: Unit, rel: str, pair: dict, natspec: str,
               version: Optional[str] = None
               ) -> Tuple[bool, bool, Dict[str, object]]:
    """(compiles, tags_emitted, what solc produced).

    `tags_emitted` compares solc's own devdoc against the declaration: every
    named parameter documented in the comment must appear in devdoc's params,
    and a notice in the comment must appear in userdoc. If solc dropped it,
    the comment does not say what its author thinks it says.
    """
    patched = dict(unit.sources)
    patched[rel] = attach(unit.sources[rel], pair, natspec)
    probe = Unit(entry=rel, sources=patched, unresolved=list(unit.unresolved))
    payload = probe.standard_json()
    payload["settings"]["outputSelection"] = {
        "*": {"*": ["devdoc", "userdoc"], "": ["ast"]}}

    versions = [version] if version else installed_versions()
    last = {}
    for v in versions:
        try:
            out = _run(v, payload)
        except CompileError:
            continue
        errs = [e for e in out.get("errors", [])
                if e.get("severity") == "error"]
        if errs:
            last = {"errors": [e.get("message", "")[:160] for e in errs[:3]]}
            continue
        # solc emits no devdoc for a modifier at all, so there is nothing to
        # compare against. The compile check still applies; the tag check is
        # reported as not applicable rather than as a failure.
        # solc only emits devdoc/userdoc for members it exposes: public and
        # external functions, plus errors and events. A modifier, or an
        # internal or private function, has no devdoc entry no matter how
        # well documented it is, so there is nothing to compare against. The
        # compile check still applies; the tag check is reported as not
        # applicable rather than as a dropped tag.
        if pair.get("kind") == "modifier":
            return True, True, {"note": "modifiers carry no devdoc"}
        if (pair.get("kind") == "function"
                and pair.get("visibility") in ("internal", "private")):
            return True, True, {"note": f"{pair['visibility']} functions "
                                        f"carry no devdoc"}
        user, dev = _doc_for(out, rel, pair.get("container"), _solc_key(pair))
        want_notice = "@notice" in natspec or natspec.strip().startswith("///")
        got = {"userdoc": user, "devdoc": dev}
        ok = True
        if want_notice and "@notice" in natspec and not user.get("notice"):
            ok = False
        claimed = set()
        for line in natspec.splitlines():
            line = line.strip().lstrip("/* ").strip()
            if line.startswith("@param"):
                parts = line.split()
                if len(parts) > 1:
                    claimed.add(parts[1])
        emitted = set((dev.get("params") or {}).keys())
        if claimed and not claimed <= emitted:
            ok = False
            got["dropped_params"] = sorted(claimed - emitted)
        return True, ok, got
    return False, False, last


def run_gate(unit: Unit, rel: str, pair: dict, draft: str, refined: str,
             version: Optional[str] = None) -> GateResult:
    reasons: List[str] = []

    d_before = judge_text(pair, draft)
    d_after = judge_text(pair, refined)
    score_kept = len(d_after) <= len(d_before)
    if not score_kept:
        reasons.append(f"defects rose from {len(d_before)} to {len(d_after)}")

    compiles, tags, detail = solc_emits(unit, rel, pair, refined, version)
    if not compiles:
        reasons.append(f"does not compile: {detail}")
    elif not tags:
        reasons.append(f"solc dropped tags: {detail.get('dropped_params')}")

    return GateResult(passed=compiles and tags and score_kept,
                      compiles=compiles, tags_emitted=tags,
                      score_kept=score_kept,
                      draft_defects=d_before, refined_defects=d_after,
                      reasons=reasons)


def judge_text(pair: dict, natspec: str) -> List[str]:
    """Defect labels for a candidate comment against the real declaration.

    Reuses `score.judge` by rebuilding the (doc, decl) attachment from the
    candidate text, so the gate's notion of a defect is the corpus's notion of
    a defect, down to the same code.
    """
    from .natspec import parse as parse_doc
    from .extract import Attachment, Unit as _U   # noqa: F401
    from . import solidity as S

    doc = parse_doc(natspec)
    params = [S.Param(type="", name=n, raw=n)
              for n in _declared_params(pair)]
    rets = [S.Param(type=r.get("name") or "", name=r.get("name"), raw="")
            for r in pair.get("returns", [])]
    decl = S.Decl(kind=pair.get("kind", "function"), name=pair.get("name", ""),
                  container=pair.get("container"),
                  container_kind=pair.get("container_kind"),
                  header_start=0, header_end=0, body_end=None,
                  header="", params=params, returns=rets,
                  visibility=pair.get("visibility"),
                  mutability=pair.get("mutability"),
                  is_virtual=False, overrides=False)
    from .extract import Attachment as A, Unit as UU
    unit = UU(spans=[], start=0, end=0, raw=natspec)
    return judge(A(doc=doc, decl=decl, unit=unit)).defects


def _declared_params(pair: dict) -> List[str]:
    """Parameter names as declared, recovered from the pair's own code."""
    from . import solidity as S
    from .masking import code_view, scan
    code = pair.get("code", "")
    try:
        spans = scan(code, strict=False)
        decls = S.find_decls(code_view(code, spans), code)
        if decls:
            return [p.name for p in decls[0].params if p.name]
    except Exception:                       # noqa: BLE001
        pass
    return list(pair.get("params", {}).keys())
