#!/usr/bin/env python3
"""Environment setup and verification.

You asked for a `requirements.py`; pip needs a `requirements.txt`, so that
file holds the dependency list and this one installs it and then *checks* the
things a list cannot express: that solc binaries are present, that Slither can
actually drive one, and whether a model is reachable.

    python setup_env.py            install, then verify
    python setup_env.py --check    verify only, install nothing
    python setup_env.py --solc     install the solc versions the corpus needs

Exit code 0 means the pipeline can run. Exit code 1 names what is missing.
Stages that need a model are reported separately: their absence is not a
failure, it just means `run_pipeline.sh` will stop after stage 3.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: The compilers the 13 corpus projects actually pin. Newest first, which is
#: also the order the compiler search tries them in.
SOLC_VERSIONS = ["0.8.13", "0.8.10", "0.8.7", "0.8.6", "0.8.3", "0.8.0",
                 "0.7.6", "0.7.5", "0.7.3", "0.6.12", "0.6.10", "0.5.17",
                 "0.5.16"]

OK, BAD, WARN = "  ok   ", "  FAIL ", "  warn "


def say(mark: str, what: str, detail: str = "") -> None:
    print(f"{mark} {what}" + (f"  — {detail}" if detail else ""), flush=True)


# --------------------------------------------------------------------------

def install() -> bool:
    req = HERE / "requirements.txt"
    print(f"installing from {req}\n", flush=True)
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req)]
    if _is_externally_managed():
        cmd.append("--break-system-packages")
    return subprocess.run(cmd).returncode == 0


def _is_externally_managed() -> bool:
    """Debian and Ubuntu mark the system Python as externally managed; pip
    then refuses to install without an explicit flag. Detected rather than
    assumed, so a virtualenv is not given the flag it does not need."""
    if sys.prefix != sys.base_prefix:
        return False                                   # inside a virtualenv
    marker = Path(getattr(sys, "base_prefix", sys.prefix))
    return any(p.name == "EXTERNALLY-MANAGED"
               for p in marker.glob("lib/python*/EXTERNALLY-MANAGED"))


def install_solc(versions=None) -> bool:
    versions = versions or SOLC_VERSIONS
    if not shutil.which("solc-select"):
        say(BAD, "solc-select", "not installed; run without --check first")
        return False
    ok = True
    for v in versions:
        r = subprocess.run(["solc-select", "install", v],
                           capture_output=True, text=True)
        if r.returncode != 0 and "already installed" not in r.stdout:
            say(WARN, f"solc {v}", r.stderr.strip().splitlines()[-1][:90]
                if r.stderr.strip() else "could not install")
            ok = False
        else:
            say(OK, f"solc {v}")
    return ok


# --------------------------------------------------------------------------

def check() -> dict:
    """Everything the pipeline needs, and what each stage needs it for."""
    report = {"python": sys.version.split()[0], "problems": [],
              "warnings": [], "stages": {}}
    say(OK, f"python {report['python']}")

    for mod, why in (("numpy", "retrieval"), ("faiss", "retrieval"),
                     ("scipy", "statistics"), ("matplotlib", "the figure"),
                     ("pytest", "tests")):
        try:
            m = __import__(mod)
            say(OK, mod, getattr(m, "__version__", ""))
        except ImportError:
            say(BAD, mod, f"needed for {why}")
            report["problems"].append(mod)

    try:
        import slither
        say(OK, "slither", getattr(slither, "__version__", "?"))
    except ImportError:
        say(BAD, "slither", "needed for Σ(f) — stage 1b cannot run")
        report["problems"].append("slither")

    # --- compilers --------------------------------------------------------
    sys.path.insert(0, str(HERE))
    try:
        from natspec_corpus.compile import installed_versions, probe
        found = installed_versions()
        report["solc_versions"] = found
        if found:
            say(OK, f"solc ({len(found)})", ", ".join(found[:6])
                + (" …" if len(found) > 6 else ""))
        else:
            say(BAD, "solc", "no compiler found; run with --solc")
            report["problems"].append("solc")
        missing = [v for v in ("0.8.13", "0.7.6", "0.6.12") if v not in found]
        if missing and found:
            say(WARN, "solc coverage",
                f"{', '.join(missing)} absent; some files will not compile")
            report["warnings"].append(f"solc missing {missing}")
    except Exception as e:                              # noqa: BLE001
        say(BAD, "natspec_corpus", f"import failed: {e}")
        report["problems"].append("natspec_corpus")

    # --- the model, which is optional here --------------------------------
    report["stages"]["1 corpus"] = "ready" if not report["problems"] else "blocked"
    report["stages"]["1b sigma"] = ("ready" if "slither" not in report["problems"]
                                    and report.get("solc_versions") else "blocked")
    report["stages"]["2 retrieval index"] = (
        "ready" if "faiss" not in report["problems"] else "blocked")

    # A zip that has been through Windows loses the executable bit, and then
    # `./run_pipeline.sh` fails with Permission denied for a reason that looks
    # nothing like a file mode. Cheap to check, annoying to diagnose.
    not_executable = [f for f in ("run_pipeline.sh", "setup_env.py",
                                  "tools/stub_ollama.py")
                      if (HERE / f).exists() and not os.access(HERE / f, os.X_OK)]
    if not_executable:
        say(WARN, "executable bit",
            f"missing on {', '.join(not_executable)} — "
            f"run: chmod +x {' '.join(not_executable)}")
        report["warnings"].append(f"not executable: {not_executable}")
    else:
        say(OK, "executable bit")

    host = _ollama()
    if host:
        say(OK, "ollama", f"{host['url']} — {len(host['models'])} models")
        report["ollama"] = host
        report["stages"]["3 generation"] = "ready"
        report["stages"]["4 experiments"] = "ready"
    else:
        say(WARN, "ollama", "not reachable; stages 3 and 4 will be skipped")
        report["stages"]["3 generation"] = "no model"
        report["stages"]["4 experiments"] = "no model"

    print()
    for stage, state in report["stages"].items():
        say(OK if state == "ready" else WARN, stage, state)
    return report


def _ollama(url: str = "http://localhost:11434") -> dict | None:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=4) as r:
            data = json.loads(r.read())
        return {"url": url,
                "models": [m.get("name") for m in data.get("models", [])]}
    except Exception:                                   # noqa: BLE001
        return None


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify only")
    ap.add_argument("--solc", action="store_true",
                    help="install the solc versions the corpus needs")
    ap.add_argument("--json", type=Path, help="write the report here")
    args = ap.parse_args()

    if not args.check:
        if not install():
            say(BAD, "pip install", "failed")
            return 1
        print()
    if args.solc:
        install_solc()
        print()

    report = check()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")

    if report["problems"]:
        print(f"\nmissing: {', '.join(report['problems'])}")
        return 1
    print("\nready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
