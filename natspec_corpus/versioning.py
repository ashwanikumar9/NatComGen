"""Never overwrite a result.

A results directory that is rewritten in place has one number in it — the
last one — and no way to tell whether the table you are reading came from the
run you think it did. That is a bad property for a directory whose whole job
is to be quoted in a thesis. So every run writes a fresh set: the first is
`main.md`, the second `main_2.md`, the third `main_3.md`.

Two rules make the numbering useful rather than merely non-destructive.

**The suffix is per run, not per file.** A run's tables, its LaTeX, its
figure and its manifest all carry the same number, so `main_3.md`,
`conditions_3.md` and `ablations_3.png` are the same run and can be read
together. Numbering each file independently would produce a `main_4.md` next
to a `conditions_2.md` the first time a configuration was missing, and
nothing in the directory would say they disagreed.

**The number is derived from what is on disk, not from a counter.** The index
is one past the highest version any of the run's own targets already has, so
deleting `main_2.md` and re-running produces `main_2.md` again — the
directory describes itself, and there is no hidden state to get out of step
with it.

`RUNS.md` is the index: what each number was, when it ran, and on what. It is
the one file here that *is* overwritten, because it is a view of the ledger
beside it rather than a result.

Set `NATCOMGEN_RESULT_VERSIONING=off` to restore plain overwriting.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

ENV = "NATCOMGEN_RESULT_VERSIONING"
LEDGER = "RUNS.jsonl"
INDEX = "RUNS.md"

_SUFFIX = re.compile(r"^(?P<stem>.*?)_(?P<n>[1-9][0-9]*)$")


def enabled() -> bool:
    return os.environ.get(ENV, "on").strip().lower() not in {
        "off", "0", "false", "no", "none"}


def split_version(stem: str) -> tuple:
    """`("main", 1)` for `main`, `("main", 3)` for `main_3`.

    `main_with_sigma` comes back as `("main_with_sigma", 1)` — the suffix has
    to be all digits, so a table whose name happens to end in an underscore
    word is not mistaken for version 0 of something else.
    """
    m = _SUFFIX.match(stem)
    return (m.group("stem"), int(m.group("n"))) if m else (stem, 1)


def highest(root: Path, rel: str, *, directory: bool = False) -> int:
    """The highest version of `rel` already on disk, or 0 if there is none.

    `directory=True` looks for sibling directories instead — the emit stage
    writes a folder of documented contracts, and a folder is as much a result
    as a table is.
    """
    p = Path(rel)
    d = root / p.parent
    if not d.is_dir():
        return 0
    base, _ = split_version(p.stem if not directory else p.name)
    best = 0
    for f in d.iterdir():
        if directory:
            if not f.is_dir():
                continue
            stem, n = split_version(f.name)
        else:
            if not f.is_file() or f.suffix != p.suffix:
                continue
            stem, n = split_version(f.stem)
        if stem == base:
            best = max(best, n)
    return best


class RunVersion:
    """One run's numbering, shared by every file that run writes."""

    def __init__(self, root: Path, index: int, *, active: bool = True) -> None:
        self.root = Path(root)
        self.index = index
        self.active = active
        self.written: Dict[str, str] = {}

    @classmethod
    def open(cls, root: Path, targets: Sequence[str],
             dir_targets: Sequence[str] = ()) -> "RunVersion":
        root = Path(root)
        if not enabled():
            return cls(root, 1, active=False)
        seen = [highest(root, t) for t in targets]
        seen += [highest(root, t, directory=True) for t in dir_targets]
        return cls(root, max(seen, default=0) + 1)

    def path(self, rel) -> Path:
        p = Path(rel)
        if not self.active or self.index <= 1:
            out = self.root / p
        else:
            base, _ = split_version(p.stem)
            out = self.root / p.parent / f"{base}_{self.index}{p.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def dir(self, rel) -> Path:
        """A directory for this run: `documented`, then `documented_2`."""
        p = Path(rel)
        if not self.active or self.index <= 1:
            out = self.root / p
        else:
            base, _ = split_version(p.name)
            out = self.root / p.parent / f"{base}_{self.index}"
        out.mkdir(parents=True, exist_ok=True)
        self.written[str(rel)] = out.name + "/"
        return out

    def write_text(self, rel, text: str) -> Path:
        p = self.path(rel)
        p.write_text(text, encoding="utf-8")
        self.written[str(rel)] = p.name
        return p

    def write_json(self, rel, obj, **kw) -> Path:
        return self.write_text(rel, json.dumps(obj, indent=1, **kw))

    def claim(self, rel, path: Path) -> Path:
        """Record a file written by something else — a figure, say."""
        self.written[str(rel)] = Path(path).name
        return path

    # ----------------------------------------------------------------
    # the index
    # ----------------------------------------------------------------

    def record(self, **meta) -> Optional[Path]:
        """Append this run to the ledger and rebuild `RUNS.md`."""
        if not self.active:
            return None
        row = {"run": self.index,
               "at": time.strftime("%Y-%m-%d %H:%M:%S"),
               **{k: v for k, v in meta.items() if v is not None},
               "files": dict(sorted(self.written.items()))}
        ledger = self.root / LEDGER
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        _rebuild_index(self.root)
        return ledger


def read_ledger(root: Path) -> List[dict]:
    p = Path(root) / LEDGER
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _rebuild_index(root: Path) -> Path:
    rows = read_ledger(root)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in ("run", "at", "files") and k not in keys:
                keys.append(k)

    out = ["# Results index",
           "",
           "One row per run. Files from the same run share a suffix, so "
           "`main_3.md` and `conditions_3.md` belong together. The newest run "
           "is last.",
           "",
           "| run | when | " + " | ".join(keys + ["files"]) + " |",
           "|---|---|" + "---|" * (len(keys) + 1)]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v)
            elif isinstance(v, dict):
                v = ", ".join(f"{a}={b}" for a, b in sorted(v.items()))
            cells.append(str(v).replace("|", "\\|"))
        names = sorted(set((r.get("files") or {}).values()))
        cells.append(f"{len(names)}: " + ", ".join(f"`{n}`" for n in names[:6])
                     + (" …" if len(names) > 6 else ""))
        out.append(f"| {r.get('run')} | {r.get('at', '')} | "
                   + " | ".join(cells) + " |")
    out.append("")
    p = Path(root) / INDEX
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out), encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# the one-off case
# --------------------------------------------------------------------------

def next_path(path) -> Path:
    """The given path if it is free, else `stem_2`, `stem_3`, …

    For a single file written outside a run — a tool invoked by hand with
    `--json`. Where several files belong together, use `RunVersion` so they
    share a number.
    """
    p = Path(path)
    if not enabled() or not p.exists():
        return p
    base, _ = split_version(p.stem)
    n = max(highest(p.parent, p.name), 1)
    while True:
        n += 1
        cand = p.parent / f"{base}_{n}{p.suffix}"
        if not cand.exists():
            return cand
