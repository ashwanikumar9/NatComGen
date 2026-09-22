"""An on-disk index of every function in a bulk full-source Solidity corpus.

SQLite, because the corpora worth matching against run from tens of thousands
to a million files and an in-memory dict of that size is both slow to build
and lost the moment the job is interrupted. The index is resumable: a file
already recorded under the same content hash is skipped, so re-running after
a dropped connection costs only the files that had not been reached.

Function bodies are located with `natspec_corpus.solidity`, the same parser
the corpus builder uses. That matters more than it sounds: a regex over raw
source gets parameter lists with nested mappings wrong and gets a function
containing a `}` inside a string literal wrong, and either mistake produces a
match key for a body that is not the body.

Comments are blanked before the key is taken, strings are not. SmartDoc's
tokeniser dropped comments and kept string literals, so the key has to do
the same on this side or nothing with a `require` message would ever match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from natspec_corpus.errors import LexError, ParseError          # noqa: E402
from natspec_corpus.masking import Kind, mask, scan             # noqa: E402
from natspec_corpus.solidity import find_containers, find_decls  # noqa: E402

from . import tokens as T                                        # noqa: E402

# Comments blanked, string literals kept, offsets preserved.
NO_COMMENTS = frozenset({Kind.CODE, Kind.STRING})

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  id      INTEGER PRIMARY KEY,
  path    TEXT UNIQUE NOT NULL,
  sha1    TEXT NOT NULL,
  nfuncs  INTEGER NOT NULL DEFAULT 0,
  problem TEXT
);
CREATE TABLE IF NOT EXISTS funcs (
  id       INTEGER PRIMARY KEY,
  file_id  INTEGER NOT NULL REFERENCES files(id),
  contract TEXT,
  name     TEXT NOT NULL,
  nparams  INTEGER NOT NULL,
  sig      TEXT NOT NULL,
  start    INTEGER NOT NULL,
  end      INTEGER NOT NULL,
  ekey     TEXT NOT NULL,
  bkey     TEXT NOT NULL,
  ntok     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ekey ON funcs(ekey);
CREATE INDEX IF NOT EXISTS idx_bkey ON funcs(bkey);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.executescript(SCHEMA)
    # Durability is not worth much here — the index is derived data and
    # rebuilding a shard is cheap — but throughput is, and WAL plus a relaxed
    # sync is the difference between an hour and a morning on a million files.
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=OFF")
    return con


def iter_sources(root: Path) -> Iterator[Path]:
    yield from sorted(root.rglob("*.sol"))


def read_source(path: Path) -> str:
    """Decode without newline translation.

    `Path.read_text` opens in universal-newline mode, which turns every
    `\r\n` into `\n` and so *shortens* the string. Every offset recorded
    against a CRLF file would then point a character or two past where it
    should, and the recovered body would start mid-token. Indexing and
    matching must decode identically; this is the one function both use.
    """
    return path.read_bytes().decode("utf-8", "replace")


def functions_in(src: str):
    """Yield `(contract, name, nparams, start, end, match_text)` per function.

    `match_text` is the declaration through its closing brace with comments
    blanked — exactly the span SmartDoc's tokeniser was fed.
    """
    spans = scan(src, strict=False)
    code = mask(src, spans, NO_COMMENTS)
    containers = find_containers(code, src)
    for d in find_decls(code, src, containers):
        if d.kind != "function" or d.body_end is None or not d.name:
            continue
        text = code[d.header_start:d.body_end]
        # `d.sig` is the parameter *type* key, e.g. `add(Set,bytes32)`. Arity
        # alone is not a key: a library that overloads `add` on two struct
        # types has two functions with the same name and the same count, and
        # keying on arity would fold them together and give one of them the
        # other's reference.
        yield (d.container, d.name, len(d.params), d.sig,
               d.header_start, d.body_end, text)


def add_file(con: sqlite3.Connection, path: Path, rel: str) -> int:
    raw = path.read_bytes()
    sha = hashlib.sha1(raw).hexdigest()
    row = con.execute("SELECT id, sha1 FROM files WHERE path=?", (rel,)).fetchone()
    if row and row[1] == sha:
        return 0                                    # already indexed, unchanged
    if row:
        con.execute("DELETE FROM funcs WHERE file_id=?", (row[0],))
        con.execute("DELETE FROM files WHERE id=?", (row[0],))

    src = read_source(path)
    problem, rows = None, []
    try:
        for contract, name, nparams, sig, start, end, text in functions_in(src):
            rows.append((contract, name, nparams, sig, start, end,
                         T.exact_key(text), f"{name}/{nparams}",
                         len(T.tokenise(text))))
    except (LexError, ParseError, RecursionError) as e:
        # A file the lexer refuses is a property of the crawl, not a bug to
        # crash on: the bulk corpora contain truncated and mis-encoded files.
        # Record why and move on, so the count is auditable afterwards.
        problem, rows = f"{type(e).__name__}: {e}"[:200], []

    cur = con.execute(
        "INSERT INTO files(path, sha1, nfuncs, problem) VALUES (?,?,?,?)",
        (rel, sha, len(rows), problem))
    fid = cur.lastrowid
    con.executemany(
        "INSERT INTO funcs(file_id,contract,name,nparams,sig,start,end,"
        "ekey,bkey,ntok) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(fid, *r) for r in rows])
    return len(rows)


def build(root: Path, db: Path, *, limit: Optional[int] = None,
          progress_every: int = 500) -> dict:
    con = connect(db)
    con.execute("INSERT OR REPLACE INTO meta VALUES ('root', ?)",
                (str(root.resolve()),))
    seen = nfun = failed = 0
    t0 = time.time()
    try:
        for p in iter_sources(root):
            rel = p.relative_to(root).as_posix()
            try:
                nfun += add_file(con, p, rel)
            except (OSError, UnicodeError) as e:
                failed += 1
                print(f"  ! {rel}: {e}", file=sys.stderr)
            seen += 1
            if seen % progress_every == 0:
                con.commit()
                rate = seen / max(time.time() - t0, 1e-6)
                print(f"  {seen} files, {nfun} new functions, "
                      f"{rate:.0f} files/s", file=sys.stderr)
            if limit and seen >= limit:
                break
    finally:
        con.commit()
    total = con.execute("SELECT COUNT(*) FROM funcs").fetchone()[0]
    files = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    probs = con.execute(
        "SELECT COUNT(*) FROM files WHERE problem IS NOT NULL").fetchone()[0]
    con.commit()
    con.close()
    return {"root": str(root), "db": str(db), "files_seen": seen,
            "files_indexed": files, "functions": total,
            "unparsable_files": probs, "io_failures": failed,
            "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="directory of .sol files")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.root, a.db, limit=a.limit), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
