"""Fetch the SmartDoc release and check it is the one the paper reports on.

Only `raw.githubusercontent.com` is used. The repository landing page and
`codeload` are both blocked on some university networks while raw is not, and
a git clone of a repository this small buys nothing anyway.

Everything downloaded is verified by size and line count before it is used.
A silently truncated `ref.txt` would shift every reference by one line and
produce a BLEU score that looks plausible and means nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = "https://raw.githubusercontent.com/xing-hu/SmartDoc/master"

# The four published systems whose outputs ship with the repository. These
# are what we re-score on the matched subset.
SYSTEMS = ["attendgru", "ast-attendgru", "re2com", "smartdoc"]

FILES = {
    "dataset.zip": f"{RAW}/dataset.zip",
    "ref.txt": f"{RAW}/final_results/RQ1/ref.txt",
    "evaluate.py": f"{RAW}/final_results/evaluate.py",
    **{f"{s}.out": f"{RAW}/final_results/RQ1/{s}.out" for s in SYSTEMS},
}

EXPECTED_LINES = 1000


def _get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "natcomgen-bench"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _lines(b: bytes) -> list[str]:
    return b.decode("utf-8", "replace").splitlines()


def fetch(dest: Path, *, force: bool = False) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    got = {}
    for name, url in FILES.items():
        p = dest / name
        if p.exists() and not force:
            got[name] = p.stat().st_size
            continue
        print(f"  fetching {name}", file=sys.stderr)
        data = _get(url)
        p.write_bytes(data)
        got[name] = len(data)

    with zipfile.ZipFile(dest / "dataset.zip") as z:
        for member in z.namelist():
            if member.startswith("__MACOSX") or member.endswith("/"):
                continue
            if not member.startswith("dataset/"):
                continue
            target = dest / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(member))

    report = verify(dest)
    (dest / "fetch_report.json").write_text(json.dumps(report, indent=1))
    return report


def verify(dest: Path) -> dict:
    """Line counts and hashes for everything the benchmark reads.

    `ref.txt` ships without a trailing newline, so `wc -l` reports 999 for a
    1000-line file. Counting with `splitlines` rather than trusting `wc` is
    the difference between a correct benchmark and an off-by-one.
    """
    code = dest / "dataset/test/test.token.code"
    nl = dest / "dataset/test/test.token.nl"
    ref = dest / "ref.txt"

    report = {"ok": True, "files": {}, "problems": []}
    expect = {
        str(code.relative_to(dest)): EXPECTED_LINES,
        str(nl.relative_to(dest)): EXPECTED_LINES,
        "ref.txt": EXPECTED_LINES,
        **{f"{s}.out": EXPECTED_LINES for s in SYSTEMS},
    }
    for rel, want in expect.items():
        p = dest / rel
        if not p.exists():
            report["ok"] = False
            report["problems"].append(f"{rel} missing")
            continue
        b = p.read_bytes()
        n = len(_lines(b))
        report["files"][rel] = {
            "bytes": len(b), "lines": n,
            "sha1": hashlib.sha1(b).hexdigest(),
        }
        if n != want:
            report["ok"] = False
            report["problems"].append(f"{rel}: {n} lines, expected {want}")

    if code.exists() and ref.exists():
        c, r = _lines(code.read_bytes()), _lines(ref.read_bytes())
        if len(c) == len(r):
            # The paper's own `test.token.nl` and `final_results/RQ1/ref.txt`
            # differ on exactly one line (645, whitespace). ref.txt is what
            # their evaluate.py scores against, so ref.txt is authoritative
            # here too; the disagreement is recorded, not repaired.
            nls = _lines(nl.read_bytes()) if nl.exists() else []
            if nls and nls != r:
                diff = [i for i, (a, b2) in enumerate(zip(nls, r)) if a != b2]
                report["nl_vs_ref_differ_at"] = diff
    return report


def load(dest: Path):
    """`(codes, refs)` — 1000 tokenised bodies and 1000 reference comments."""
    codes = _lines((dest / "dataset/test/test.token.code").read_bytes())
    refs = _lines((dest / "ref.txt").read_bytes())
    if len(codes) != len(refs):
        raise SystemExit(f"misaligned: {len(codes)} codes, {len(refs)} refs")
    return codes, refs


def load_train(dest: Path):
    codes = _lines((dest / "dataset/train/train.token.code").read_bytes())
    refs = _lines((dest / "dataset/train/train.token.nl").read_bytes())
    n = min(len(codes), len(refs))
    return codes[:n], refs[:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data/smartdoc")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args(argv)
    rep = verify(a.dest) if a.verify_only else fetch(a.dest, force=a.force)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
