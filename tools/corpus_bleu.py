#!/usr/bin/env python3
"""Corpus BLEU-4 for the 1-NN baseline, under two split regimes.

This exists so the split-methodology number in the write-up is reproducible.
Three decisions are pinned here because each one moves the answer, in one case
by a factor of fifteen:

**Rendering.** A NatSpec comment is turned into one string by parsing it and
joining the field values in a fixed order. Scoring the raw `/** ... */` text
instead inflates BLEU by rewarding the comment markers themselves.

**Nothing is dropped.** An empty hypothesis scores zero and stays in the
corpus. Excluding the cases where the baseline produced nothing was worth
+3 BLEU in the random-split arm and — because the number of such cases differs
by condition — meant the two arms were scored over different segment sets,
which is precisely the comparison the experiment is supposed to make fair.

**Per-kind breakdown.** All 52 `error` declarations in this corpus sit in the
validation split and none in training, so part of any project-split penalty is
an unseen declaration kind rather than project disjointness. Reporting the
split by kind keeps those two explanations apart.

    python tools/corpus_bleu.py --seeds 5
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))

import sacrebleu  # noqa: E402

from natspec_corpus.natspec import parse as parse_doc  # noqa: E402
from natspec_corpus.retrieve import build_index, load_units  # noqa: E402
from split_effect import _corpus_with_splits  # noqa: E402


def render(natspec: str) -> str:
    """A comment as one scoreable string: its field values, fixed order."""
    d = parse_doc(natspec or "")
    bits = [d.notice, d.dev]
    bits += list(d.params.values())
    bits += [t.text for t in d.returns]
    return " ".join(b.strip() for b in bits if b and b.strip()).strip()


def one_nn(root: Path, pairs: Dict[str, dict], split: str = "val") -> dict:
    index = build_index(root, split="train")
    hyps, refs, kinds = [], [], []
    for unit in load_units(root, split):
        hits = index.search(unit.views, top_k=1, tau=0.0)
        # An empty hypothesis is a result, not a reason to drop the segment.
        hyps.append(render(hits[0].unit.natspec) if hits else "")
        refs.append(render(pairs[unit.pair_id]["doc_raw"]))
        kinds.append(pairs[unit.pair_id].get("kind", "?"))

    out = {"n": len(hyps),
           "empty_hypotheses": sum(1 for h in hyps if not h),
           "empty_references": sum(1 for r in refs if not r),
           "bleu": sacrebleu.corpus_bleu(hyps, [refs]).score}
    by_kind = {}
    for kind in sorted(set(kinds)):
        sel = [(h, r) for h, r, k in zip(hyps, refs, kinds) if k == kind]
        by_kind[kind] = {"n": len(sel),
                         "bleu": sacrebleu.corpus_bleu(
                             [h for h, _ in sel], [[r for _, r in sel]]).score}
    out["by_kind"] = by_kind
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, default=HERE / "data" / "NatSpecGold")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    pairs, sizes = {}, collections.Counter()
    for line in (args.corpus / "pairs.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            pairs[p["id"]] = p
            sizes[p["split"]] += 1

    report = {"rendering": "parsed fields joined; nothing dropped",
              "encoder": "HashingEncoder (offline); a floor, not a ceiling",
              "project_split": one_nn(args.corpus, pairs),
              "random_split": []}
    print(f"project split: BLEU {report['project_split']['bleu']:.2f} "
          f"(n={report['project_split']['n']}, "
          f"{report['project_split']['empty_hypotheses']} empty hyps)",
          file=sys.stderr, flush=True)

    ids = sorted(pairs)
    for seed in range(args.seeds):
        rng = random.Random(seed)
        shuffled = ids[:]
        rng.shuffle(shuffled)
        assignment, i = {}, 0
        for split in ("train", "val", "test"):
            for pid in shuffled[i:i + sizes[split]]:
                assignment[pid] = split
            i += sizes[split]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpus"
            _corpus_with_splits(args.corpus, root, assignment)
            shuffled_pairs = {}
            for line in (root / "pairs.jsonl").read_text(
                    encoding="utf-8").splitlines():
                if line.strip():
                    p = json.loads(line)
                    shuffled_pairs[p["id"]] = p
            r = one_nn(root, shuffled_pairs)
            r["seed"] = seed
            report["random_split"].append(r)
        print(f"  seed {seed}: BLEU {r['bleu']:.2f} (n={r['n']}, "
              f"{r['empty_hypotheses']} empty hyps)", file=sys.stderr, flush=True)

    vals = [r["bleu"] for r in report["random_split"]]
    report["random_split_summary"] = {
        "mean": statistics.mean(vals), "sd": statistics.pstdev(vals),
        "min": min(vals), "max": max(vals), "seeds": len(vals)}
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
