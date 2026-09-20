#!/usr/bin/env python3
"""Install solc compilers WITHOUT sudo, entirely inside this repository.

`compile.py` looks for `$SOLC_SHIM_ROOT/<anything>/solc` and asks each binary
its own version, so the layout it needs is one directory per compiler with an
executable called `solc` inside. This builds that under `<repo>/.solc`, which
keeps every byte written inside the project directory — nothing lands in
`~/.solc-select`, nothing needs root, and nothing touches a shared machine
outside the one folder you were given.

Two routes, tried in order:

1. The official static linux-amd64 builds from binaries.soliditylang.org.
   These are self-contained executables; downloading and chmod +x is the
   whole installation.
2. `solc-select`, pointed at `<repo>/.solc-select` via SOLC_SELECT_HOME so it
   also stays inside the project, with its artifacts linked into the layout
   above.

    python tools/install_solc.py            the versions the corpus pins
    python tools/install_solc.py --check    what is already installed
    python tools/install_solc.py 0.8.13     just one
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("SOLC_SHIM_ROOT", HERE / ".solc"))
INDEX = "https://binaries.soliditylang.org/linux-amd64/list.json"
BASE = "https://binaries.soliditylang.org/linux-amd64/"
#: Route 0. The same static linux-amd64 builds, published as GitHub release
#: assets. Kept first because binaries.soliditylang.org is blocked outright on
#: some university and corporate networks — it answered 403 to every version
#: on the network this was written for — while github.com is reachable
#: wherever git is.
GH = "https://github.com/ethereum/solidity/releases/download/v{v}/solc-static-linux"

#: What the 13 corpus projects actually pin.
VERSIONS = ["0.8.13", "0.8.10", "0.8.7", "0.8.6", "0.8.3", "0.8.0",
            "0.7.6", "0.7.5", "0.7.3", "0.6.12", "0.6.10", "0.5.17", "0.5.16"]


def installed() -> Dict[str, Path]:
    """Version -> binary, by asking each one rather than reading its path."""
    out: Dict[str, Path] = {}
    if not ROOT.is_dir():
        return out
    for d in sorted(ROOT.iterdir()):
        exe = d / "solc"
        if not exe.is_file():
            continue
        try:
            r = subprocess.run([str(exe), "--version"], capture_output=True,
                               text=True, timeout=60)
        except Exception:                                   # noqa: BLE001
            continue
        for line in (r.stdout or "").splitlines():
            if "Version:" in line:
                out[line.split("Version:")[1].strip().split("+")[0]] = exe
    return out


def _place(version: str, blob: bytes) -> Path:
    """Write the binary into the layout compile.py probes, atomically."""
    d = ROOT / version
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "solc.part"
    tmp.write_bytes(blob)
    tmp.chmod(tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    os.replace(tmp, d / "solc")
    return d / "solc"


def from_github(versions: List[str]) -> Dict[str, str]:
    """Route 0: GitHub release assets, one static binary per tag."""
    results: Dict[str, str] = {}
    for v in versions:
        try:
            req = urllib.request.Request(
                GH.format(v=v), headers={"User-Agent": "natcomgen-installer"})
            with urllib.request.urlopen(req, timeout=180) as r:
                blob = r.read()
            if len(blob) < 1_000_000:
                results[v] = f"suspiciously small download ({len(blob)} bytes)"
                continue
            _place(v, blob)
            results[v] = "ok"
        except Exception as e:                              # noqa: BLE001
            results[v] = str(e)[:120]
    return results


def from_binaries(versions: List[str]) -> Dict[str, str]:
    """Route 1: the official static builds."""
    try:
        with urllib.request.urlopen(INDEX, timeout=30) as r:
            releases = json.loads(r.read())["releases"]
    except Exception as e:                                  # noqa: BLE001
        return {v: f"index unreachable: {e}" for v in versions}

    results: Dict[str, str] = {}
    for v in versions:
        name = releases.get(v)
        if not name:
            results[v] = "not in the release index"
            continue
        try:
            with urllib.request.urlopen(BASE + name, timeout=180) as r:
                blob = r.read()
            if len(blob) < 1_000_000:       # a real solc is tens of megabytes
                results[v] = f"suspiciously small download ({len(blob)} bytes)"
                continue
            _place(v, blob)
            results[v] = "ok"
        except Exception as e:                              # noqa: BLE001
            results[v] = str(e)[:120]
    return results


def from_solc_select(versions: List[str]) -> Dict[str, str]:
    """Route 2: solc-select, confined to the project by SOLC_SELECT_HOME."""
    if not shutil.which("solc-select"):
        return {v: "solc-select not installed" for v in versions}
    home = HERE / ".solc-select"
    home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SOLC_SELECT_HOME=str(home))
    results: Dict[str, str] = {}
    for v in versions:
        r = subprocess.run(["solc-select", "install", v], env=env,
                           capture_output=True, text=True)
        art = home / "artifacts" / f"solc-{v}" / f"solc-{v}"
        if art.is_file():
            d = ROOT / v
            d.mkdir(parents=True, exist_ok=True)
            link = d / "solc"
            if link.exists() or link.is_symlink():
                link.unlink()
            try:
                link.symlink_to(art)
            except OSError:
                shutil.copy2(art, link)
            link_target = link if link.is_file() else art
            link_target.chmod(link_target.stat().st_mode | stat.S_IXUSR)
            results[v] = "ok"
        else:
            tail = (r.stderr or r.stdout or "").strip().splitlines()
            results[v] = tail[-1][:120] if tail else "failed"
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("versions", nargs="*", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    have = installed()
    if args.check:
        print(f"SOLC_SHIM_ROOT = {ROOT}")
        for v, p in sorted(have.items()):
            print(f"  ok   {v}  {p}")
        missing = [v for v in VERSIONS if v not in have]
        print(f"  {len(have)} installed, {len(missing)} missing"
              + (f": {', '.join(missing)}" if missing else ""))
        return 0 if have else 1

    wanted = [v for v in (args.versions or VERSIONS) if v not in have]
    if not wanted:
        print(f"all {len(have)} compilers already present under {ROOT}")
        return 0

    print(f"installing {len(wanted)} compilers into {ROOT}\n", flush=True)
    results: Dict[str, str] = {}
    for label, route in (("github releases", from_github),
                         ("binaries.soliditylang.org", from_binaries),
                         ("solc-select", from_solc_select)):
        todo = [v for v in wanted if results.get(v) != "ok"]
        if not todo:
            break
        print(f"  trying {label} for {len(todo)}…", flush=True)
        results.update(route(todo))
        done = sum(1 for v in wanted if results.get(v) == "ok")
        print(f"  {done}/{len(wanted)} installed", flush=True)

    for v in sorted(results):
        print(f"  {'ok  ' if results[v] == 'ok' else 'FAIL'} {v}"
              + ("" if results[v] == "ok" else f"  — {results[v]}"))

    have = installed()
    print(f"\n{len(have)} compilers now usable under {ROOT}")
    print("run_pipeline.sh picks this up automatically; for a bare shell:")
    print(f"  export SOLC_SHIM_ROOT={ROOT}")
    return 0 if have else 1


if __name__ == "__main__":
    raise SystemExit(main())
