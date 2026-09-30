"""SmartDoc's test set, reshaped as a corpus ComGen can run on.

The point is one fully controlled comparison. Scoring each system against its
own corpus's reference cancels the corpus difference but still leaves two
different function populations; running ComGen over SmartDoc's own 1,000 test
functions removes that too. Same functions, same references, same scorer, both
systems.

    python3 -m functionWise.smartdoc_corpus --pkg ~/Ashwani/MTP/smartdoc-pkg \\
        --out data/SmartDocEval --sample 10          # pilot
    python3 -m functionWise.smartdoc_corpus --pkg ~/Ashwani/MTP/smartdoc-pkg \\
        --out data/SmartDocEval                      # all 1000

WHAT COMGEN LOSES HERE, AND WHY IT IS STILL WORTH RUNNING

Their release is tokenised function bodies: no contract, no imports, no
pragma. Nothing compiles, so **there is no fact table** and ComGen runs in
what its own ablation calls G2 — the configuration that costs -0.435 claim
support. The compiler half of the deterministic critic also sits out; the tag
half still works, because parameter names parse from the declaration.

That handicap cuts different ways for different metrics, and the report has to
say which:

  gate recall   unaffected. Caller gates are read off the declaration, which
                is present. A win here is won WITH the handicap, so it is
                conservative.
  BLEU          affected, and against us. Report it as "zero-shot, ungrounded,
                on the corpus the other system was trained on" or not at all.

REFERENCES. Only `notice` is filled, from their `.nl` file. `params` and
`returns` stay empty because their corpus has none — and `reference_fields`
drops empty fields, so nothing is scored against a reference that does not
exist. Do not be tempted to score @param here; there is nothing to score
against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from natspec_corpus import access as A                        # noqa: E402
from natspec_corpus import solidity as S                      # noqa: E402
from natspec_corpus.masking import code_view, scan            # noqa: E402


def declaration(code: str):
    """The function declaration, parsed from the tokenised source.

    Their tokenisation spaces out every punctuation mark
    (`function f ( uint a ) public { ... }`), which `find_decls` handles — but
    a line that does not parse is dropped rather than guessed at, because a
    wrong parameter list would corrupt the deterministic critic downstream.
    """
    try:
        spans = scan(code, strict=False)
        decls = S.find_decls(code_view(code, spans), code)
        return decls[0] if decls else None
    except Exception:                                    # noqa: BLE001
        return None


def build(pkg: Path) -> List[dict]:
    ds = pkg / "dataset" / "test"
    code_lines = (ds / "test.token.code").read_text(
        encoding="utf-8", errors="ignore").splitlines()
    nl_lines = (ds / "test.token.nl").read_text(
        encoding="utf-8", errors="ignore").splitlines()
    n = min(len(code_lines), len(nl_lines))
    out: List[dict] = []
    for i in range(n):
        code, notice = code_lines[i].strip(), nl_lines[i].strip()
        d = declaration(code)
        if d is None or not d.name:
            continue
        out.append({
            "id": f"sd{i:04d}",
            "line": i,
            "project": "smartdoc",
            "file": f"smartdoc/f{i:04d}.sol",
            "container": d.container or "Contract",
            "container_kind": d.container_kind or "contract",
            "kind": "function",
            "name": d.name,
            "signature": d.sig,
            "arity_key": d.arity_key,
            "visibility": d.visibility or "public",
            "mutability": d.mutability,
            "is_virtual": False,
            "overrides": False,
            "doc_raw": "",
            "inheritdoc": None,
            "notice": notice,
            "dev": "",
            "params": {},          # their corpus has no @param references
            "returns": [],         # nor @return
            "code": code,
            "verified": True,
            "defects": [],
            "split": "test",
        })
    return out


def sample(pairs: List[dict], k: int) -> List[dict]:
    """`k` pairs, every caller-gated one first.

    Gated functions are the scarce resource — 297 of 1,000 — and the metric
    that matters is computed only on them. Taking a plain random sample would
    throw away the experiment to save time on the filler.
    """
    gated = [p for p in pairs if A.caller_gates(p, None)]
    rest = [p for p in pairs if not A.caller_gates(p, None)]
    if k >= len(pairs):
        return pairs
    take = gated[:k]
    if len(take) < k and rest:
        step = max(1, len(rest) // max(k - len(take), 1))
        take += rest[::step][:k - len(take)]
    return sorted(take, key=lambda p: p["line"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "data/SmartDocEval")
    ap.add_argument("--sample", type=int, default=0,
                    help="keep only this many, gated functions first")
    args = ap.parse_args(argv)

    pkg = args.pkg.expanduser()
    if not (pkg / "dataset" / "test" / "test.token.code").exists():
        raise SystemExit(f"run `unzip -o dataset.zip -d .` inside {pkg} first")

    pairs = build(pkg)
    total = len(pairs)
    if args.sample:
        pairs = sample(pairs, args.sample)

    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "contracts").mkdir(exist_ok=True)     # empty: nothing compiles
    (out / "pairs.jsonl").write_text(
        "\n".join(json.dumps(p) for p in pairs), encoding="utf-8")
    (out / "index.jsonl").write_text("\n".join(
        json.dumps({"id": p["id"], "line": p["line"]}) for p in pairs),
        encoding="utf-8")

    gated = sum(1 for p in pairs if A.caller_gates(p, None))
    print(f"wrote {len(pairs)} pairs to {out}/pairs.jsonl "
          f"({total} parsed from 1000 lines)")
    print(f"  caller-gated: {gated}")
    print(f"  no sigma/ and an empty contracts/ — ComGen will run UNGROUNDED "
          f"here, which is the point to state in the write-up")
    print(f"\nnext:\n  python3 -m comgen run --corpus {out} "
          f"--index-corpus data/NatSpecGold --split test "
          f"--configs G1 --seeds 0 --out comgen/results/smartdoc_runs\n"
          f"\n  --index-corpus matters: retrieval exemplars come from "
          f"ComGen's own training pool, and this corpus has no train split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
