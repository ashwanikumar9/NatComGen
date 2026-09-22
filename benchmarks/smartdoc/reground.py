"""Match each SmartDoc test function back to the contract it came from.

Two passes, in this order and never the other way round.

**Exact.** The whitespace-stripped body hashes to the same key as some
function in the index. This is not a similarity score that happens to be 1.0
— the two token streams are character-for-character identical once
whitespace is removed — so it needs no threshold and admits no false
positives beyond genuine duplicate code.

**Fuzzy**, only for what exact missed, and only inside the `name/arity`
bucket. Jaccard over token trigrams, with both a floor and a margin over the
runner-up. Deployed Solidity is full of near-identical functions — a bucket
for `transfer/2` holds thousands of copies of the same ERC-20 body — so a
high score alone proves nothing. The margin is what makes the pass safe, and
a bucket that cannot separate its candidates yields no match rather than a
guess.

When several files hold the same function, the one with the most functions
wins. That is a proxy for "the most complete flattened contract", which is
the copy most likely to compile on its own and therefore the copy most likely
to produce a fact table. The choice is recorded per match so it can be
audited, and `candidates` says how many there were.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from . import fetch, tokens as T
from .index import NO_COMMENTS, read_source

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from natspec_corpus.masking import mask, scan            # noqa: E402


def _root(con: sqlite3.Connection) -> Path:
    row = con.execute("SELECT v FROM meta WHERE k='root'").fetchone()
    if not row:
        raise SystemExit("index has no root recorded; rebuild it")
    return Path(row[0])


def _slice(root: Path, path: str, start: int, end: int) -> str:
    src = read_source(root / path)
    return mask(src, scan(src, strict=False), NO_COMMENTS)[start:end]


def _rank(rows):
    """Most functions in the file first, then shortest path, then path."""
    return sorted(rows, key=lambda r: (-r["nfuncs"], len(r["path"]), r["path"]))


def _rows_for(con, where: str, arg) -> list[dict]:
    q = ("SELECT f.id, f.contract, f.name, f.nparams, f.sig, f.start, f.end,"
         "       f.ntok, fl.path, fl.nfuncs "
         "FROM funcs f JOIN files fl ON fl.id = f.file_id "
         f"WHERE {where}")
    cols = ("fid contract name nparams sig start end ntok path nfuncs").split()
    return [dict(zip(cols, r)) for r in con.execute(q, (arg,))]


def reground(db: Path, data: Path, out: Path, *, split: str = "test",
             threshold: float = 0.85, margin: float = 0.05,
             fuzzy_cap: int = 400, limit: Optional[int] = None) -> dict:
    con = sqlite3.connect(str(db))
    root = _root(con)
    codes, refs = (fetch.load(data) if split == "test"
                   else fetch.load_train(data))
    if limit:
        codes, refs = codes[:limit], refs[:limit]

    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total": len(codes), "exact": 0, "fuzzy": 0, "none": 0,
             "ambiguous_bucket": 0, "no_header": 0, "bucket_too_big": 0}
    matched = []

    for i, (code, ref) in enumerate(zip(codes, refs)):
        code = code.strip()
        ek, bk = T.exact_key(code), T.bucket_key(code)
        rows = _rows_for(con, "f.ekey = ?", ek)
        if rows:
            best = _rank(rows)[0]
            matched.append({"index": i, "how": "exact", "score": 1.0,
                            "candidates": len(rows), "reference": ref,
                            "file": best["path"], "contract": best["contract"],
                            "name": best["name"], "nparams": best["nparams"],
                            "sig": best["sig"],
                            "start": best["start"], "end": best["end"]})
            stats["exact"] += 1
            continue

        if bk is None:
            stats["no_header"] += 1
            stats["none"] += 1
            continue

        bucket = _rows_for(con, "f.bkey = ?", bk)
        # A near-miss on length is a near-miss on content; skipping those
        # first keeps the expensive disk reads off obviously wrong candidates.
        n = len(T.tokenise(code))
        bucket = [r for r in bucket if 0.6 * n <= r["ntok"] <= 1.6 * n]
        if not bucket:
            stats["none"] += 1
            continue
        if len(bucket) > fuzzy_cap:
            # A bucket this crowded is a boilerplate function that exists in
            # thousands of copies. No margin rule can pick the right one, and
            # reading 400+ files to prove it is wasted work.
            stats["bucket_too_big"] += 1
            stats["none"] += 1
            continue

        by_id = {r["fid"]: r for r in bucket}
        cands = []
        for r in bucket:
            try:
                cands.append((r["fid"], _slice(root, r["path"], r["start"], r["end"])))
            except OSError:
                continue
        hit = T.best_match(code, cands, threshold=threshold, margin=margin)
        if hit is None:
            stats["ambiguous_bucket"] += int(bool(cands))
            stats["none"] += 1
            continue
        fid, score = hit
        r = by_id[fid]
        matched.append({"index": i, "how": "fuzzy", "score": round(score, 4),
                        "candidates": len(bucket), "reference": ref,
                        "file": r["path"], "contract": r["contract"],
                        "name": r["name"], "nparams": r["nparams"],
                        "sig": r["sig"],
                        "start": r["start"], "end": r["end"]})
        stats["fuzzy"] += 1

    con.close()
    with out.open("w", encoding="utf-8") as fh:
        for m in matched:
            fh.write(json.dumps(m) + "\n")

    stats["matched"] = len(matched)
    stats["coverage"] = round(len(matched) / max(stats["total"], 1), 4)
    stats["distinct_files"] = len({m["file"] for m in matched})
    stats["index_root"] = str(root)
    stats["split"] = split
    (out.parent / f"coverage_{split}.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv=None) -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "data/smartdoc")
    ap.add_argument("--out", type=Path, default=here / "data/matched.jsonl")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--fuzzy-cap", type=int, default=400)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    s = reground(a.db, a.data, a.out, split=a.split, threshold=a.threshold,
                 margin=a.margin, fuzzy_cap=a.fuzzy_cap, limit=a.limit)
    print(json.dumps(s, indent=1))
    print(f"\ncoverage: {s['matched']} of {s['total']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
