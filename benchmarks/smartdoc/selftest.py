"""Prove the benchmark works before spending a day downloading a corpus.

The real run needs half a million verified contracts. That download is the
expensive step and the last place you want to discover that an offset is off
by two, so this builds a *fixture* with the same shape out of NatComGen's own
352 audited sources: take functions from those files, tokenise them exactly
the way SmartDoc's release is tokenised, and present them as a test set.

Every stage then runs for real — index, match, annotate, build, Σ(f)-less
generation against the offline stub, score — and two things must hold at the
end:

* **coverage is 1.0.** Every fixture function came out of the indexed corpus,
  so anything less than a complete match is a bug in the matcher, not a
  property of the data.
* **every scored pair's gold notice is byte-equal to its reference.** This is
  the check that matters most and the one that is easiest to skip: the
  reference travels from `ref.txt` into a `///` line in a copy of a contract,
  through the corpus builder's lexer, extractor and scorer, and back out of
  `pairs.jsonl`. A single failure anywhere in that chain leaves the benchmark
  scoring against the *contract author's* sentence instead of SmartDoc's, and
  every count still looks right.

The BLEU numbers this produces are meaningless by construction — the fixture
pairs NatComGen's function bodies with SmartDoc's unrelated references, and
the stub model writes filler. Mechanism is what is being tested.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from . import reground as RG, tokens as T                 # noqa: E402
from .fetch import SYSTEMS                                 # noqa: E402

N_DEFAULT = 200


def make_fixture(db: Path, real_data: Path, dest: Path, *, n: int = N_DEFAULT,
                 seed: int = 1) -> dict:
    """A SmartDoc-shaped directory whose code comes from the indexed corpus."""
    con = sqlite3.connect(str(db))
    root = RG._root(con)
    rows = con.execute(
        "SELECT f.start, f.end, fl.path FROM funcs f "
        "JOIN files fl ON fl.id = f.file_id").fetchall()
    con.close()
    if len(rows) < n:
        n = len(rows)

    rnd = random.Random(seed)
    codes = []
    for start, end, path in rnd.sample(rows, len(rows)):
        # `\n` inside a preserved string literal would split one function
        # across two lines and misalign everything after it.
        line = " ".join(T.tokenise(RG._slice(root, path, start, end)))
        line = line.replace("\n", " ").replace("\r", " ").strip()
        if line:
            codes.append(line)
        if len(codes) >= n:
            break

    (dest / "dataset/test").mkdir(parents=True, exist_ok=True)
    (dest / "dataset/test/test.token.code").write_text(
        "\n".join(codes) + "\n", encoding="utf-8")
    # The training comments come along so the self-test also exercises the
    # seen/unseen split, which is otherwise silently skipped.
    train = real_data / "dataset/train/train.token.nl"
    if train.exists():
        (dest / "dataset/train").mkdir(parents=True, exist_ok=True)
        (dest / "dataset/train/train.token.nl").write_text(
            train.read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8")

    for name in ["ref.txt", *(f"{s}.out" for s in SYSTEMS)]:
        src = (real_data / name)
        if not src.exists():
            raise SystemExit(f"{src} missing — run the fetch stage first")
        lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
        (dest / name).write_text("\n".join(lines[:len(codes)]),
                                 encoding="utf-8")
    return {"functions": len(codes), "dest": str(dest)}


def check(corpus_root: Path, corpus_map: Path, fixture: Path) -> dict:
    from .score import load_map
    refs = (fixture / "ref.txt").read_text(
        encoding="utf-8", errors="replace").splitlines()
    pair_to_index = load_map(corpus_root, corpus_map)
    checked = mismatched = 0
    examples = []
    for line in (corpus_root / "pairs.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        i = pair_to_index.get(p["id"])
        if i is None:
            continue
        checked += 1
        got = " ".join((p.get("notice") or "").split())
        want = " ".join(refs[i].split())
        if got != want:
            mismatched += 1
            if len(examples) < 3:
                examples.append({"file": p["file"], "sig": p["signature"],
                                 "want": want[:90], "got": got[:90]})
    return {"joined": len(pair_to_index), "checked": checked,
            "gold_mismatches": mismatched, "examples": examples,
            "ok": checked > 0 and mismatched == 0}


def main(argv=None) -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["reground", "check"],
                    default="reground")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    a = ap.parse_args(argv)

    fixture = a.data / "selftest"
    if a.stage == "reground":
        info = make_fixture(a.data / "corpus.db", a.data / "smartdoc",
                            fixture, n=a.n)
        print(json.dumps(info, indent=1))
        stats = RG.reground(a.data / "corpus.db", fixture,
                            a.data / "matched.jsonl", split="test")
        print(json.dumps(stats, indent=1))
        if stats["coverage"] < 1.0:
            print("!! the fixture came out of the indexed corpus, so coverage "
                  "below 1.0 is a matcher bug", file=sys.stderr)
            return 1
        return 0

    r = check(a.data / "smartdoc_corpus", a.data / "corpus_map_test.json",
              fixture)
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
