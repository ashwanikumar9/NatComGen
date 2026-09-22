"""Get a bulk corpus of **full, compilable** Solidity sources to match against.

The whole benchmark stands on this step. SmartDoc's functions came from
verified Etherscan contracts, so the recall of the re-grounding pass is
bounded by how much of verified Etherscan the local corpus contains. A few
hundred audited files will match almost nothing; half a million crawled
contracts will match a large fraction.

Four sources, in rough order of yield per gigabyte:

  `disl`       ASSERT-KTH/DISL on Hugging Face — 514k deduplicated verified
               sources, MIT. The best ratio of coverage to disk.
  `sanctuary`  tintinweb/smart-contract-sanctuary — ~1.1M flattened mainnet
               contracts as a git repository. Archived upstream, still the
               broadest crawl. Large: budget tens of gigabytes.
  `local`      a directory you already have. Use this on a machine where the
               downloads are blocked and the files arrived some other way.
  `self`       NatComGen's own `data/sources`. 352 audited files; it will
               match essentially none of SmartDoc, and it is here so the
               wiring can be tested end to end without a download.

Network reality, measured rather than assumed: from the container this was
written in, `huggingface.co`, `zenodo.org` and `api.etherscan.io` are all
unreachable and `github.com` returns 400, while `raw.githubusercontent.com`
works. A university network will differ. Each route therefore fails with the
URL it tried and what to do instead, rather than a stack trace.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HF_API = "https://huggingface.co/api/datasets"
SANCTUARY = "https://github.com/tintinweb/smart-contract-sanctuary.git"

DATASETS = {
    "disl": ("ASSERT-KTH/DISL", "raw", "train", "source_code", "contract_name"),
    "fiesta": ("Zellic/smart-contract-fiesta", None, "train", None, None),
    "andstor": ("andstor/smart_contracts", "raw", "train", "source_code",
                "contract_name"),
}


def _reachable(url: str, timeout: int = 20) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "natcomgen"})
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except (urllib.error.URLError, OSError):
        return False


def from_hf(name: str, dest: Path, *, limit: int | None = None) -> dict:
    """Stream a Hugging Face dataset's source column into one .sol per row.

    Uses the `datasets` library when it is importable, because it handles
    the parquet shard listing, resumption and the cache. `pip install --user
    datasets pyarrow` is enough on a box with no sudo.
    """
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name}; try {sorted(DATASETS)}")
    repo, config, split, src_col, name_col = DATASETS[name]
    if not _reachable("https://huggingface.co"):
        raise SystemExit(
            "huggingface.co is not reachable from this machine.\n"
            "  Options: run this step somewhere with access and copy the "
            "directory over, then use --source local --from DIR;\n"
            "  or use --source sanctuary, which goes through github.com.")
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "the `datasets` package is required for this source:\n"
            "  pip install --user datasets pyarrow")

    dest.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(repo, config, split=split, streaming=True)
    n = 0
    for row in ds:
        src = row.get(src_col) or row.get("source_code") or row.get("source")
        if not src:
            continue
        # Shard by a prefix so no directory holds half a million entries;
        # ext4 copes, but `ls` and rsync do not.
        stem = (row.get(name_col) or row.get("contract_name")
                or f"c{n:08d}").replace("/", "_")[:80]
        d = dest / f"{n % 1000:03d}"
        d.mkdir(exist_ok=True)
        (d / f"{stem}_{n:08d}.sol").write_text(str(src), encoding="utf-8")
        n += 1
        if n % 5000 == 0:
            print(f"  {n} files", file=sys.stderr)
        if limit and n >= limit:
            break
    return {"source": f"hf:{repo}", "files": n, "dest": str(dest)}


def from_sanctuary(dest: Path, *, chains=("mainnet",)) -> dict:
    """Shallow-clone the sanctuary. Large, and archived upstream."""
    if shutil.which("git") is None:
        raise SystemExit("git not found on PATH")
    dest.mkdir(parents=True, exist_ok=True)
    repo = dest / "smart-contract-sanctuary"
    if not repo.exists():
        cmd = ["git", "clone", "--depth", "1", "--filter=blob:none",
               "--sparse", SANCTUARY, str(repo)]
        print("  " + " ".join(cmd), file=sys.stderr)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(
                "clone failed. github.com may be blocked here; if so there is "
                "no raw-only route to a repository this size.")
    subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set",
                    *[f"contracts/{c}" for c in chains]], check=False)
    n = sum(1 for _ in repo.rglob("*.sol"))
    return {"source": "sanctuary", "files": n, "dest": str(repo)}


def from_local(src: Path, dest: Path) -> dict:
    if not src.is_dir():
        raise SystemExit(f"{src} is not a directory")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.is_symlink():
            dest.unlink()
        else:
            raise SystemExit(f"{dest} exists and is not a symlink; move it")
    os.symlink(src.resolve(), dest)
    return {"source": f"local:{src}", "files": sum(1 for _ in src.rglob('*.sol')),
            "dest": str(dest)}


def main(argv=None) -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True,
                    choices=["disl", "fiesta", "andstor", "sanctuary",
                             "local", "self"])
    ap.add_argument("--dest", type=Path, default=here / "data/corpus")
    ap.add_argument("--from", dest="src", type=Path,
                    help="with --source local")
    ap.add_argument("--limit", type=int, help="stop after N contracts")
    a = ap.parse_args(argv)

    if a.source == "self":
        info = from_local(here.parent / "data/sources", a.dest)
    elif a.source == "local":
        if not a.src:
            raise SystemExit("--source local needs --from DIR")
        info = from_local(a.src, a.dest)
    elif a.source == "sanctuary":
        info = from_sanctuary(a.dest)
    else:
        info = from_hf(a.source, a.dest, limit=a.limit)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
