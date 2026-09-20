#!/usr/bin/env python3
"""How much of a reported BLEU is the split rather than the method?

Every smart contract comment-generation paper I could find splits its corpus
randomly over contracts mined from Etherscan, where the same ERC-20 is
deployed thousands of times. A random split puts near-duplicates of a test
function into the training set, and a retrieval baseline then scores very
well for reasons that have nothing to do with understanding code.

NatSpecGold splits by PROJECT, so nothing from a validation project appears in
training at all. That is the harder and more honest setting, but it makes the
numbers look worse, and a reader comparing them to a published 33-47 BLEU
without knowing this would draw the wrong conclusion.

This measures the difference directly: the same retrieval baseline, the same
functions, the same metric — only the split changes. Everything else is held
fixed, so whatever gap appears is attributable to the split alone.

    python tools/split_effect.py [--seeds 5]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from natspec_corpus.evaluate import aggregate, score_record  # noqa: E402
from natspec_corpus.retrieve import build_index, load_units  # noqa: E402


def _corpus_with_splits(src: Path, dst: Path, assignment: Dict[str, str]) -> None:
    """A corpus identical to `src` except for which split each pair is in.

    The allowlist is rebuilt to match, because `load_units` enforces it and a
    stale one would raise rather than quietly disagree — which is the correct
    behaviour and the reason this has to be rebuilt rather than ignored.
    """
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "sigma").mkdir(exist_ok=True)
    sigma = src / "sigma" / "sigma.jsonl"
    if sigma.exists():
        shutil.copy2(sigma, dst / "sigma" / "sigma.jsonl")

    allow: Dict[str, List[str]] = {"train": set(), "val": set(), "test": set(),
                                   "never_index": []}
    out = []
    for line in (src / "pairs.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        p["split"] = assignment[p["id"]]
        allow[p["split"]].add(p["file"])
        out.append(json.dumps(p, ensure_ascii=False))
    (dst / "pairs.jsonl").write_text("\n".join(out) + "\n", encoding="utf-8")
    (dst / "index_allowlist.json").write_text(
        json.dumps({k: sorted(v) if isinstance(v, set) else v
                    for k, v in allow.items()}), encoding="utf-8")


def _one_nn_bleu(root: Path, split: str = "val") -> dict:
    pairs = {}
    for line in (root / "pairs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            if p["split"] == split:
                pairs[p["id"]] = p
    index = build_index(root, split="train")
    scored = []
    for unit in load_units(root, split):
        hits = index.search(unit.views, top_k=1, tau=0.0)
        if not hits:
            continue
        scored.append(score_record(
            {"pair_id": unit.pair_id, "final": hits[0].unit.natspec or "",
             "has_sigma": True}, pairs[unit.pair_id]))
    return aggregate(scored)["all"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, default=HERE / "data" / "NatSpecGold")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    original = {}
    sizes = {"train": 0, "val": 0, "test": 0}
    for line in (args.corpus / "pairs.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            original[p["id"]] = p["split"]
            sizes[p["split"]] += 1

    report = {"sizes": sizes,
              "project_split": _one_nn_bleu(args.corpus),
              "random_split": []}

    ids = sorted(original)
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
            report["random_split"].append(_one_nn_bleu(root))
        print(f"  seed {seed}: notice BLEU "
              f"{report['random_split'][-1]['notice']['bleu']:.3f}",
              file=sys.stderr, flush=True)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
