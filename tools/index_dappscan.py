"""Build a dependency-only vendor tree from DAppSCAN, for Σ(f) coverage.

64 of the 135 files carrying gold pairs cannot compile because their npm
packages are not in the corpus. DAppSCAN ships whole versioned package trees
(`openzeppelin-contracts-4.2.0/`, `openzeppelin-contracts-upgradeable-master/`),
so the missing dependencies can be supplied from a dataset this project already
uses, with no network and with the version visible in the directory name.

    python3 tools/index_dappscan.py --dappscan /path/to/DAppSCAN-source/contracts
    python3 tools/index_dappscan.py --dry-run          # report, copy nothing

It writes `data/vendor/` — the package trees, pruned to the transitive closure
the corpus actually needs — plus `index.json` holding the remapping and the
provenance of every tree. That directory is small enough to commit, so the GPU
box gets it with a pull and never needs DAppSCAN itself.

WHAT THIS DOES NOT TOUCH. `pairs.jsonl`, `splits.json`, the references, the
retrieval index. Vendored files are dependencies: never scored, never pairs,
never exemplars, never @inheritdoc donors. The evaluation set is bit-identical
before and after; the only thing that changes is how many pairs have a fact
table. `natspec_corpus/vendor.py` explains why that is safe in more detail.

CHOOSING A TREE. For each package prefix, every candidate root in DAppSCAN is
scored by how many of the corpus's missing files it actually contains, and the
best one wins. Picking per-file across projects would mix incompatible versions;
picking one tree keeps a package internally consistent.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from natspec_corpus.compile import _imports, unit_for                # noqa: E402
from natspec_corpus.vendor import INDEX                              # noqa: E402


def split_spec(spec: str) -> Tuple[str, str]:
    """('@openzeppelin/contracts/token/ERC20/IERC20.sol')
        -> ('@openzeppelin/contracts/', 'token/ERC20/IERC20.sol')"""
    parts = spec.split("/")
    n = 2 if spec.startswith("@") and len(parts) > 2 else 1
    return "/".join(parts[:n]) + "/", "/".join(parts[n:])


def wanted(corpus: Path) -> Tuple[Dict[str, Set[str]], Set[str], Dict[str, int]]:
    """(file -> its missing specs, all corpus paths, spec -> how many files)."""
    contracts = corpus / "contracts"
    pairs = [json.loads(l) for l in (corpus / "pairs.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    files = sorted({p["file"] for p in pairs})
    on_disk = {str(p.relative_to(contracts)).replace(os.sep, "/")
               for p in contracts.rglob("*.sol")}

    def read(rel: str) -> Optional[str]:
        p = contracts / rel
        return p.read_bytes().decode("utf-8", "replace") if p.is_file() else None

    need: Dict[str, Set[str]] = {}
    demand: Dict[str, int] = collections.Counter()
    for rel in files:
        try:
            u = unit_for(rel, read)
        except Exception:                                    # noqa: BLE001
            continue
        # A non-relative import naming a corpus file needs no vendoring — the
        # resolver handles it directly. Only genuine packages are wanted here.
        missing = {s for s in u.unresolved if s not in on_disk}
        if missing:
            need[rel] = missing
            for s in missing:
                demand[s] += 1
    return need, on_disk, demand


def find_roots(dappscan: Path, specs: Set[str],
               keep: int = 4) -> Dict[str, List[Tuple[Path, int]]]:
    """Donor roots per package prefix, best first.

    Several, not one. The corpus spans OpenZeppelin v3 and v4, which moved
    files between releases — `math/SafeMath.sol` became `utils/math/
    SafeMath.sol` — so one tree leaves every project on the other major version
    uncompilable. Roots are ordered by how many wanted files each contains and
    tried in order at resolution time.
    """
    by_prefix: Dict[str, Set[str]] = collections.defaultdict(set)
    for spec in specs:
        prefix, rest = split_spec(spec)
        by_prefix[prefix].add(rest)
    basenames = {Path(r).name for rests in by_prefix.values() for r in rests}

    score: Dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for dirpath, _, filenames in os.walk(dappscan):
        for fn in filenames:
            if fn not in basenames:
                continue
            full = os.path.join(dirpath, fn).replace(os.sep, "/")
            for prefix, rests in by_prefix.items():
                for rest in rests:
                    if Path(rest).name == fn and full.endswith("/" + rest):
                        score[prefix][full[: -(len(rest) + 1)]] += 1
    out: Dict[str, List[Tuple[Path, int]]] = {}
    for prefix, counter in score.items():
        chosen: List[Tuple[Path, int]] = []
        covered: Set[str] = set()
        # Greedy by marginal contribution: a second tree earns its place only
        # by holding files the first one does not.
        for root, _ in counter.most_common():
            rests = {r for r in by_prefix[prefix]
                     if (Path(root) / r).is_file()} - covered
            if not rests:
                continue
            chosen.append((Path(root), len(rests)))
            covered |= rests
            if len(chosen) >= keep or covered >= by_prefix[prefix]:
                break
        if chosen:
            out[prefix] = chosen
    return out


def copy_closure(root: Path, rests: Set[str], dest: Path) -> Set[str]:
    """Copy the wanted files and everything they relatively import."""
    import posixpath
    copied: Set[str] = set()
    stack = list(rests)
    while stack:
        rel = stack.pop()
        if rel in copied:
            continue
        src = root / rel
        if not src.is_file():
            continue
        copied.add(rel)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_bytes().decode("utf-8", "replace")
        out.write_bytes(text.encode("utf-8"))
        for spec in _imports(text):
            if spec.startswith("."):
                stack.append(posixpath.normpath(
                    posixpath.join(posixpath.dirname(rel), spec)))
    return copied


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=HERE / "data/NatSpecGold")
    ap.add_argument("--dappscan", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "data/vendor")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    need, on_disk, demand = wanted(args.corpus)
    specs = {s for v in need.values() for s in v}
    print(f"{len(need)} files need {len(specs)} package files "
          f"({sum(demand.values())} edges)\n")

    roots = find_roots(args.dappscan, specs)
    print("donor trees chosen:")
    for prefix, trees in sorted(roots.items()):
        for root, n in trees:
            print(f"  {prefix:42} {n:3} files   {root.name[:60]}")

    def satisfied(spec: str) -> bool:
        prefix, rest = split_spec(spec)
        return any((root / rest).is_file()
                   for root, _ in roots.get(prefix, ()))

    missing = {s for s in specs if not satisfied(s)}
    print(f"\npackage files still absent: {len(missing)} of {len(specs)}")
    for s in sorted(missing)[:10]:
        print(f"  {s}")

    fixed = [r for r, m in need.items() if not (m & missing)]
    print(f"\nfiles that would fully resolve: {len(fixed)} of {len(need)}")
    if args.dry_run:
        return 0

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    prefixes, provenance, total = {}, {}, 0
    used_names: Dict[str, str] = {}
    for prefix, trees in sorted(roots.items()):
        order: List[str] = []
        for i, (root, _) in enumerate(trees):
            rests = {split_spec(s)[1] for s in specs
                     if split_spec(s)[0] == prefix
                     and (root / split_spec(s)[1]).is_file()}
            if not rests:
                continue
            # One directory per donor tree, named so a reader can see which
            # release it came from. Two prefixes sharing a tree share the
            # directory rather than copying it twice.
            name = used_names.get(str(root))
            if name is None:
                name = f"{prefix.strip('@/').replace('/', '-')}-{i}"
                used_names[str(root)] = name
                copied = copy_closure(root, rests, args.out / name)
                total += len(copied)
                print(f"  {prefix:42} -> {name}/  ({len(copied)} files)")
            else:
                copied = copy_closure(root, rests, args.out / name)
                total += len(copied)
            order.append(f"{name}/")
            provenance[f"{prefix}[{i}]"] = str(root)
        if order:
            prefixes[prefix] = order

    (args.out / INDEX).write_text(json.dumps(
        {"prefixes": prefixes, "provenance": provenance,
         "files": total,
         "note": "Dependency-only sources. Never scored, never pairs, never "
                 "retrieval units, never @inheritdoc donors. See "
                 "natspec_corpus/vendor.py."}, indent=1), encoding="utf-8")
    print(f"\n{total} files written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
