"""Which declarations a function-level evaluation is allowed to score.

NatSpecGold holds 860 pairs, and 175 of them are events, errors, modifiers,
constructors and `receive`. SmartDoc, CCGIR, SCCLLM and SmartBT all document
*functions*. Scoring a mixed population against their numbers compares two
different things, and it drags in precisely the declarations Sigma(f) has no
coverage for -- so a weakness of the fact table shows up as a weakness of the
generator.

The scope is therefore declared once, here, and every module in this package
takes it from this file rather than filtering inline.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

#: The default. What every baseline in the related work documents.
FUNCTION_ONLY: Tuple[str, ...] = ("function",)

#: Everything with a parameter list and a body. Constructors and `receive`
#: are functions in Solidity's grammar but not in the baselines' datasets, so
#: this is offered and not defaulted to.
CALLABLE: Tuple[str, ...] = ("function", "constructor", "receive", "fallback")

#: Every kind in the corpus. Reproduces the existing whole-corpus tables.
ALL: Tuple[str, ...] = ("function", "constructor", "receive", "fallback",
                        "modifier", "event", "error")

PRESETS = {"function": FUNCTION_ONLY, "callable": CALLABLE, "all": ALL}


def parse_kinds(spec: str) -> Tuple[str, ...]:
    """`function` | `callable` | `all` | a comma-separated list of kinds."""
    spec = (spec or "function").strip()
    if spec in PRESETS:
        return PRESETS[spec]
    kinds = tuple(k.strip() for k in spec.split(",") if k.strip())
    unknown = [k for k in kinds if k not in ALL]
    if unknown:
        raise SystemExit(f"unknown declaration kind(s): {', '.join(unknown)}. "
                         f"Known: {', '.join(ALL)}, or a preset "
                         f"({', '.join(PRESETS)}).")
    return kinds


def load_pairs(corpus_root: Path) -> Dict[str, dict]:
    path = Path(corpus_root) / "pairs.jsonl"
    if not path.exists():
        raise SystemExit(f"no corpus at {path}")
    out: Dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def scoped(pairs: Dict[str, dict], *, split: str = None,
           kinds: Sequence[str] = FUNCTION_ONLY) -> Dict[str, dict]:
    """The pairs this evaluation may see."""
    keep = set(kinds)
    return {i: p for i, p in pairs.items()
            if p.get("kind") in keep
            and (split is None or p.get("split") == split)}


# --------------------------------------------------------------------------
# what the scope costs, stated rather than assumed
# --------------------------------------------------------------------------

def census(pairs: Dict[str, dict], kinds: Sequence[str]) -> Tuple[list, list]:
    """Pairs per split per kind, with the in-scope total marked.

    Printed before any metric, because the honest reading of every number
    below depends on how many pairs survived the filter -- and on val the
    answer is under 60% of them.
    """
    splits = ("train", "val", "test")
    present: List[str] = []
    for k in ALL:
        if any(p.get("kind") == k for p in pairs.values()):
            present.append(k)
    headers = ["split"] + [f"{k}{'*' if k in kinds else ''}" for k in present] \
        + ["in scope", "total", "kept"]
    rows = []
    for s in splits:
        sub = [p for p in pairs.values() if p.get("split") == s]
        if not sub:
            continue
        counts = [sum(1 for p in sub if p.get("kind") == k) for k in present]
        in_scope = sum(1 for p in sub if p.get("kind") in set(kinds))
        share = f"{in_scope / len(sub) * 100:.0f}%" if sub else "-"
        rows.append([s] + [str(c) for c in counts]
                    + [str(in_scope), str(len(sub)), share])
    return headers, rows


def reference_lengths(pairs: Dict[str, dict]) -> Tuple[list, list]:
    """Word counts of the gold text, per field.

    This is here because it is the input to length calibration: BLEU and
    METEOR both reward matching the reference length distribution, and a
    generator writing three sentences where the corpus writes one loses
    precision on every one of them for a reason that has nothing to do with
    being wrong.
    """
    buckets: Dict[str, List[int]] = {"notice": [], "dev": [], "param": [],
                                     "return": [], "whole": []}
    for p in pairs.values():
        whole = 0
        for name, text in (("notice", p.get("notice")), ("dev", p.get("dev"))):
            if text and text.strip():
                n = len(text.split())
                buckets[name].append(n)
                whole += n
        for text in (p.get("params") or {}).values():
            if text and text.strip():
                n = len(text.split())
                buckets["param"].append(n)
                whole += n
        for r in (p.get("returns") or []):
            text = " ".join(x for x in (r.get("name") or "",
                                        r.get("text") or "") if x)
            if text.strip():
                n = len(text.split())
                buckets["return"].append(n)
                whole += n
        if whole:
            buckets["whole"].append(whole)
    headers = ["field", "present in", "mean words", "median", "p10", "p90"]
    rows = []
    for name in ("notice", "dev", "param", "return", "whole"):
        vals = sorted(buckets[name])
        if not vals:
            rows.append([name, "0", "-", "-", "-", "-"])
            continue
        p10 = vals[max(0, int(len(vals) * 0.10) - 1)]
        p90 = vals[min(len(vals) - 1, int(len(vals) * 0.90))]
        rows.append([name, str(len(vals)), f"{sum(vals) / len(vals):.1f}",
                     f"{median(vals):.0f}", str(p10), str(p90)])
    return headers, rows
