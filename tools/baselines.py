#!/usr/bin/env python3
"""Model-free baselines on NatSpecGold, so the agentic system has a floor.

A generation system reported without baselines is a number with nothing to
compare it to. These three need no GPU and no model, which also makes them the
only part of the evaluation that can be run anywhere:

  name-split      The function's own name, split into words, as the @notice.
                  The floor. Anything that cannot beat this is not working.

  retrieve-1nn    The nearest training function's NatSpec, copied verbatim.
                  This is the family CCGIR and NNGen belong to, and on corpora
                  full of cloned contracts it is a genuinely strong baseline —
                  which is exactly why it belongs here rather than in a
                  footnote.

  retrieve+struct The same retrieved prose, but with the tag STRUCTURE taken
                  from the function being documented rather than from its
                  neighbour: one @param per declared parameter, under the real
                  names, and @return only if the function returns something.
                  No model, no new prose — only structural grounding.

The third exists to separate two things that get conflated. Some of what
grounding buys is factual correctness the model could not have known; some is
merely getting the tag skeleton right, which needs no intelligence at all.
Reporting the second as though it were the first would overstate what the
agentic pipeline contributes.

    python tools/baselines.py [--split val] [--corpus data/NatSpecGold]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from natspec_corpus.evaluate import (aggregate, fields, reference_fields,  # noqa: E402
                                     score_record)
from natspec_corpus.retrieve import build_index, load_units  # noqa: E402

_SPLIT = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def words(identifier: str) -> str:
    """`getTotalSupply` -> `get total supply`. camelCase, snake_case and
    SCREAMING_CASE all decompose; leading underscores are dropped."""
    parts = _SPLIT.findall(identifier.lstrip("_").replace("_", " "))
    return " ".join(p.lower() for p in parts)


# --------------------------------------------------------------------------
# the baselines
# --------------------------------------------------------------------------

def name_split(pair: dict, hit=None) -> str:
    w = words(pair.get("name") or "")
    return f"/// @notice {w.capitalize()}." if w else ""


def retrieve_1nn(pair: dict, hit=None) -> str:
    """The neighbour's comment, untouched. Copying it verbatim is the point:
    any cleanup here would quietly make this a different system."""
    return (hit.unit.natspec or "") if hit is not None else ""


def retrieve_struct(pair: dict, hit=None) -> str:
    """Neighbour's prose, this function's structure.

    Parameter descriptions carry over only when the neighbour documented a
    parameter of the same name — a name match is the one piece of evidence
    available without a model that the description is about the same thing.
    Everything else falls back to the parameter's own name in words, which is
    weak but honest, and never invents behaviour.
    """
    donor = fields(hit.unit.natspec or "") if hit is not None else {}
    out: List[str] = []
    if donor.get("notice"):
        out.append(f"/// @notice {donor['notice']}")
    if donor.get("dev"):
        out.append(f"/// @dev {donor['dev']}")

    for name in (pair.get("params") or {}):
        carried = donor.get(f"param:{name}")
        text = carried or (words(name) or name)
        out.append(f"/// @param {name} {text}")

    returns = pair.get("returns") or []
    donor_returns = [v for k, v in sorted(donor.items())
                     if k.startswith("return:")]
    for i, r in enumerate(returns):
        text = donor_returns[i] if i < len(donor_returns) else ""
        label = (r.get("name") or "").strip()
        body = text or (words(label) if label else "the result")
        out.append(f"/// @return {(label + ' ' + body).strip()}")
    return "\n".join(out)


BASELINES = {
    "name-split": name_split,
    "retrieve-1nn": retrieve_1nn,
    "retrieve+struct": retrieve_struct,
}
NEEDS_RETRIEVAL = {"retrieve-1nn", "retrieve+struct"}


# --------------------------------------------------------------------------
# structural correctness — the part that is comparable across datasets
# --------------------------------------------------------------------------

def structural(pair: dict, natspec: str) -> dict:
    """Does the tag skeleton match the declaration?

    None of this compares against a human reference, so unlike BLEU it means
    the same thing on anyone's corpus: a @param set that matches the declared
    parameters is correct in a way that does not depend on how some author
    happened to word their comment.
    """
    got = fields(natspec or "")
    declared = set((pair.get("params") or {}))
    documented = {k.split(":", 1)[1] for k in got if k.startswith("param:")}
    n_returns = len(pair.get("returns") or [])
    n_documented_returns = sum(1 for k in got if k.startswith("return:"))
    return {
        "param_exact": declared == documented,
        "param_invented": bool(documented - declared),
        "param_missing": bool(declared - documented),
        "return_arity_ok": n_returns == n_documented_returns,
        "has_notice": bool(got.get("notice", "").strip()),
    }


#: Declarations that can actually take parameters and return values. The rest
#: — errors, events — make the structural metrics trivially satisfiable.
FUNCTION_KINDS = {"function", "constructor", "receive", "fallback", "modifier"}


def _share(rows: List[dict]) -> dict:
    return {k: sum(1 for r in rows if r[k]) / len(rows)
            for k in rows[0]} if rows else {}


def run(corpus_root: Path, split: str = "val") -> dict:
    pairs = {}
    for line in (corpus_root / "pairs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("split") == split:
                pairs[rec["id"]] = rec
    if not pairs:
        raise SystemExit(f"no pairs in split {split!r}")

    # Which functions have a fact table, so the two conditions the pipeline
    # reports separately can be reported separately here too.
    with_sigma = set()
    sigma = corpus_root / "sigma" / "sigma.jsonl"
    if sigma.exists():
        for line in sigma.read_text(encoding="utf-8").splitlines():
            if line.strip():
                pid = json.loads(line).get("pair_id")
                if pid:
                    with_sigma.add(pid)

    index = build_index(corpus_root, split="train")
    units = {u.pair_id: u for u in load_units(corpus_root, split)}

    # One retrieval per function, shared by both retrieval baselines, so they
    # differ only in what they do with the same neighbour.
    neighbours = {}
    for pid, unit in units.items():
        hits = index.search(unit.views, top_k=1, tau=0.0)
        neighbours[pid] = hits[0] if hits else None

    out = {"split": split, "n": len(pairs), "baselines": {}}
    for name, fn in BASELINES.items():
        scored, struct, kinds = [], [], []
        for pid, pair in sorted(pairs.items()):
            hit = neighbours.get(pid) if name in NEEDS_RETRIEVAL else None
            if name in NEEDS_RETRIEVAL and hit is None:
                continue
            text = fn(pair, hit)
            kinds.append(pair.get("kind"))
            scored.append(score_record(
                {"pair_id": pid, "final": text,
                 "has_sigma": pid in with_sigma}, pair))
            struct.append(structural(pair, text))
        agg = aggregate(scored)
        agg["structural"] = _share(struct)
        # Reported for functions alone as well, because 45% of this split is
        # errors, events and constructors, and most of those declare no
        # parameters and return nothing. A baseline that emits no @param at
        # all is therefore "structurally correct" on them for free, which
        # makes the pooled figure a fact about the corpus rather than about
        # the baseline.
        fn_struct = [s for s, k in zip(struct, kinds) if k in FUNCTION_KINDS]
        agg["structural_functions_only"] = _share(fn_struct)
        agg["n_functions_only"] = len(fn_struct)
        agg["scored"] = len(scored)
        out["baselines"][name] = agg
    out["kinds"] = {k: sum(1 for x in pairs.values() if x.get("kind") == k)
                    for k in sorted({x.get("kind") for x in pairs.values()})}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, default=HERE / "data" / "NatSpecGold")
    ap.add_argument("--split", default="val")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    report = run(args.corpus, args.split)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        # Never clobber an earlier answer: the second run writes
        # `<name>_2.json`. See natspec_corpus/versioning.py.
        from natspec_corpus.versioning import next_path
        dest = next_path(args.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {dest}", file=sys.stderr)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
