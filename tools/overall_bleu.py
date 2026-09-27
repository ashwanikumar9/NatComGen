#!/usr/bin/env python3
"""Overall corpus BLEU-4 for a NatComGen run.

The BLEU columns in `results/tables/main_*.md` are **sentence** BLEU-4 with
add-one smoothing, computed per field and averaged over functions. They are
the right thing for comparing `@notice` against `@param`, and they are not the
number a paper reports.

This computes **corpus BLEU-4**: n-gram counts pooled across the whole corpus
first, then the ratio. That is what SmartDoc, CCGIR, SCCLLM and the rest
report, and the two are not interchangeable — on this project's own data the
same predictions once read 0.036 as mean sentence BLEU and 0.64 as corpus
BLEU, a factor of eighteen.

Rendering follows `tools/corpus_bleu.py` exactly: parse the comment, join its
field values in a fixed order, drop nothing. An empty hypothesis scores zero
and stays in the corpus, because excluding the cases where a configuration
produced nothing scores the configurations over different segment sets.

The BLEU implementation is the one in `benchmarks/smartdoc/bleu.py`, reused
rather than copied: it reproduces SmartDoc's published 47.39 from their
released output and agrees with nltk to 1e-9, both as tests.

    python tools/overall_bleu.py
    python tools/overall_bleu.py --split val --json results/overall_bleu.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from benchmarks.smartdoc.bleu import corpus_bleu          # noqa: E402
from natspec_corpus.natspec import parse as parse_doc     # noqa: E402
from natspec_corpus.versioning import next_path           # noqa: E402


def render(natspec: str) -> str:
    """A comment as one scoreable string: its field values, fixed order."""
    d = parse_doc(natspec or "")
    bits = [d.notice, d.dev]
    bits += list(d.params.values())
    bits += [t.text for t in d.returns]
    return " ".join(b.strip() for b in bits if b and b.strip()).strip()


def score(refs: List[str], hyps: List[str]) -> dict:
    r = [[x.split()] for x in refs]
    h = [x.split() for x in hyps]
    w = {"BLEU": (0.25,) * 4, "B1": (1, 0, 0, 0), "B2": (0, 1, 0, 0),
         "B3": (0, 0, 1, 0), "B4": (0, 0, 0, 1)}
    out = {k: round(corpus_bleu(r, h, v) * 100, 2) for k, v in w.items()}
    out["n"] = len(h)
    out["empty_hypotheses"] = sum(1 for x in hyps if not x.strip())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=HERE / "data/NatSpecGold")
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    runs = a.runs or (a.corpus / "runs")
    gold = {}
    for line in (a.corpus / "pairs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            gold[p["id"]] = render(p.get("doc_raw", ""))

    by: Dict[tuple, Dict[str, str]] = defaultdict(dict)
    for path in sorted(runs.rglob(f"{a.split}.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("pair_id") not in gold:
                continue
            by[(rec.get("config", "?"), rec.get("seed", 0))][rec["pair_id"]] = \
                "" if rec.get("error") else render(rec.get("final") or "")
    if not by:
        raise SystemExit(f"no run records for split {a.split!r} under {runs}")

    # Score every configuration on the functions ALL of them produced, so the
    # rows are comparable. The per-configuration n is reported beside it.
    common = sorted(set.intersection(*(set(v) for v in by.values())))
    report = {"split": a.split, "scored_on": len(common), "rows": {}}
    print(f"corpus BLEU-4, {a.split} split, {len(common)} functions "
          f"common to every configuration\n")
    print(f"{'config':10} {'seed':>4} {'n':>5} {'BLEU':>7} {'B1':>7} "
          f"{'B2':>7} {'B3':>7} {'B4':>7} {'empty':>6}")
    for (cfg, seed), preds in sorted(by.items()):
        s = score([gold[i] for i in common], [preds[i] for i in common])
        report["rows"][f"{cfg}/seed{seed}"] = s
        print(f"{cfg:10} {seed:>4} {s['n']:>5} {s['BLEU']:>7} {s['B1']:>7} "
              f"{s['B2']:>7} {s['B3']:>7} {s['B4']:>7} "
              f"{s['empty_hypotheses']:>6}")

    if a.json:
        dest = next_path(a.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
