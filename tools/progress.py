#!/usr/bin/env python3
"""How is the run going, and is it going *well*?

Two questions, because the first one on its own has repeatedly been
misleading: a run can be advancing steadily while producing output that is
useless for reasons visible in the first five minutes. The health block below
is there so a broken run is caught at minute five rather than hour eight.

    python tools/progress.py
    watch -n 300 'python tools/progress.py'
"""
from __future__ import annotations

import collections
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("CORPUS", HERE / "data" / "NatSpecGold"))


def _rows():
    out = []
    for f in sorted(glob.glob(str(ROOT / "runs" / "*" / "*" / "*.jsonl"))):
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            out.append(json.loads(line))
                        except ValueError:
                            pass          # a line still being written
        except OSError:
            pass
    return out


def _alive() -> str:
    try:
        r = subprocess.run(["pgrep", "-af", "run_pipeline"], capture_output=True,
                           text=True, timeout=10)
        n = len([l for l in r.stdout.splitlines() if l.strip()])
        return f"running ({n} processes)" if n else "NOT RUNNING"
    except Exception:                                       # noqa: BLE001
        return "unknown"


def _started() -> float | None:
    """When the current run began, from the oldest run file's mtime."""
    files = glob.glob(str(ROOT / "runs" / "*" / "*" / "*.jsonl"))
    return min((os.path.getmtime(f) for f in files), default=None)


def main() -> None:
    target = int(os.environ.get("TARGET", "960"))
    rows = _rows()
    done = len(rows)
    print(f"{time.strftime('%H:%M:%S')}   {_alive()}")
    print(f"records   {done}/{target}  ({done / target:.0%})" if target else
          f"records   {done}")

    start = _started()
    if start and done:
        elapsed = time.time() - start
        rate = done / elapsed
        left = (target - done) / rate if rate and target > done else 0
        print(f"elapsed   {elapsed/3600:.1f}h   rate {rate*3600:.0f}/h"
              + (f"   eta {left/3600:.1f}h" if left else "   (complete)"))

    log = HERE / "logs" / "experiments.log"
    if log.exists():
        tail = [l.rstrip() for l in log.read_text(errors="replace").splitlines()][-2:]
        for l in tail:
            print(f"now       {l.strip()}")

    if not rows:
        print("\nno records yet — give it a few minutes")
        return

    err = [r for r in rows if r.get("error")]
    print(f"errors    {len(err)}/{done}"
          + (f"   e.g. {err[-1].get('error','')[:70]}" if err else ""))

    # ---- health: is the output usable at all? ----------------------------
    ok = [r for r in rows if not r.get("error")]

    def marker(t):
        t = (t or "").lstrip()
        for m in ("/**", "///"):
            if t.startswith(m):
                return m
        return "//" if t.startswith("//") else "bare/empty"

    print("\nHEALTH  (these decide whether the run is worth finishing)")
    m = collections.Counter(marker(r.get("final")) for r in ok)
    good = m.get("///", 0) + m.get("/**", 0)
    print(f"  real NatSpec syntax   {good}/{len(ok)}"
          f"   {dict(m)}"
          + ("   <-- BAD: solc cannot see these" if good < len(ok) * 0.8 else ""))

    g = [r["gate"] for r in ok if isinstance(r.get("gate"), dict)]
    if g:
        c = sum(1 for x in g if x.get("compiles"))
        t = sum(1 for x in g if x.get("tags_emitted"))
        print(f"  gate: compiles        {c}/{len(g)}")
        print(f"  gate: tags emitted    {t}/{len(g)}")
        kept = sum(1 for r in ok if r.get("gate_kept_draft"))
        print(f"  gate kept the draft   {kept}/{len(ok)}"
              "   (high = refinements being rejected)")

    sup = [r["support_rate"] for r in ok if r.get("support_rate") is not None]
    if sup:
        print(f"  mean claim support    {sum(sup)/len(sup):.3f}  over {len(sup)}")
    v = collections.Counter(r.get("gate_verdict") for r in ok
                            if r.get("gate_verdict"))
    if v:
        print(f"  verifier              {dict(v)}")


if __name__ == "__main__":
    sys.exit(main())
