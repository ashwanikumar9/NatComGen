"""Score a NatComGen run against SmartDoc's references, on their own metric.

Three rules, and the benchmark is worth nothing without all three.

**Same subset.** Only the matched functions can be scored, so the four
published systems are re-scored on exactly those indices too. Comparing our
number on 300 functions against their published number on 1000 would be
comparing two different test sets that happen to share a name.

**Same metric.** `bleu.py` reproduces their `evaluate.py` to 1e-9, and on the
full 1000 it returns 47.39 for `smartdoc.out` — the number in the paper.

**Same alphabet.** Their references are whitespace-tokenised prose:
punctuation stands as its own token, so `EIN .` is two tokens. A model that
writes ordinary English scores lower on `.split()` purely for attaching its
full stops. Predictions are therefore tokenised the same way before scoring,
and the raw-split number is reported alongside it so the size of that
adjustment is visible rather than buried.

`@notice` only, because SmartDoc generates only the notice sentence. The
`@dev`, `@param` and `@return` tags NatComGen produces have no counterpart in
their references and are dropped before scoring — they are not free marks,
and including them would only dilute the n-gram precision.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from natspec_corpus.natspec import parse as parse_doc     # noqa: E402
from natspec_corpus.versioning import RunVersion           # noqa: E402

from . import bleu, fetch                                  # noqa: E402

SYSTEMS = fetch.SYSTEMS

# Their NL tokeniser: punctuation separated, everything else left alone.
_PUNCT = re.compile(r"([`.,;:!?()\[\]{}\"']|--+)")


def nl_tokenise(text: str) -> str:
    return " ".join(_PUNCT.sub(r" \1 ", text).split())


def notice_of(natspec: str) -> str:
    """The `@notice` text, or untagged prose, from a generated comment."""
    if not natspec:
        return ""
    return " ".join(parse_doc(natspec).notice.split())


def seen_in_training(data: Path) -> set:
    """Indices whose reference comment appears **verbatim** in SmartDoc's own
    training set.

    434 of the 1,000 do. That is not a subtlety — it is most of the headline
    number. Scored separately, `smartdoc.out` gets 81.68 BLEU on those and
    17.53 on the 566 that are genuinely unseen, and 17.53 is an ordinary
    code-summarisation score.

    It follows from their split being random over one crawl: deployed
    Solidity is overwhelmingly copies of the same few contracts, so the same
    one-sentence comment lands on both sides of the split. Nothing here is a
    criticism of their system, which is a good system. It is a statement
    about what a single random-split BLEU number can and cannot tell you, and
    it is the reason this benchmark reports three columns instead of one.
    """
    train = data / "dataset/train/train.token.nl"
    if not train.exists():
        return set()
    seen = {" ".join(l.split()) for l in
            train.read_text(encoding="utf-8", errors="replace").splitlines()}
    refs = (data / "ref.txt").read_text(
        encoding="utf-8", errors="replace").splitlines()
    return {i for i, r in enumerate(refs) if " ".join(r.split()) in seen}


def load_map(corpus_root: Path, corpus_map: Path) -> Dict[str, int]:
    """`pair_id -> smartdoc index`.

    The run records carry `pair_id`; the staging step keyed its map by
    project, container and signature. `pairs.jsonl` is what joins them, and it is
    the corpus builder's own output, so the join cannot drift from what was
    actually built.
    """
    meta = json.loads(corpus_map.read_text())
    key_to_index = meta["map"]
    out: Dict[str, int] = {}
    missing = 0
    for line in (corpus_root / "pairs.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        key = f"{p['project']}|{p['container']}|{p['signature']}"
        if key in key_to_index:
            out[p["id"]] = key_to_index[key]
        else:
            missing += 1
    if not out:
        raise SystemExit(
            "no pair joined to a SmartDoc index — check that --corpus and "
            "--corpus-map came from the same staging run")
    return out


def predictions(runs_root: Path, pair_to_index: Dict[str, int], *,
                split: str = "val") -> Dict[tuple, Dict[int, str]]:
    """`(config, seed) -> {smartdoc index: notice}`."""
    out: Dict[tuple, Dict[int, str]] = defaultdict(dict)
    for path in sorted(runs_root.rglob(f"{split}.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = pair_to_index.get(rec.get("pair_id"))
            if idx is None or rec.get("error"):
                continue
            out[(rec.get("config", "?"), rec.get("seed", 0))][idx] = \
                notice_of(rec.get("final") or rec.get("draft") or "")
    return out


def score_subset(refs: List[str], preds: Dict[int, str], *,
                 tokenise: bool = True) -> dict:
    idx = sorted(preds)
    r = [refs[i] for i in idx]
    h = [nl_tokenise(preds[i]) if tokenise else preds[i] for i in idx]
    return bleu.leclair(r, h)


def baselines_on(refs: List[str], data: Path, idx: List[int]) -> dict:
    out = {}
    for s in SYSTEMS:
        lines = (data / f"{s}.out").read_text(
            encoding="utf-8", errors="replace").splitlines()
        out[s] = bleu.leclair([refs[i] for i in idx], [lines[i] for i in idx])
    return out


def _slice_scores(refs, get_line, idx: List[int], seen: set) -> dict:
    """`Ba` on all of `idx`, on the memorisable part, and on the rest."""
    parts = {"all": idx,
             "seen": [i for i in idx if i in seen],
             "unseen": [i for i in idx if i not in seen]}
    out = {}
    for name, ids in parts.items():
        out[name] = (bleu.leclair([refs[i] for i in ids],
                                  [get_line(i) for i in ids])["Ba"]
                     if ids else None)
        out[f"n_{name}"] = len(ids)
    return out


def report(corpus_root: Path, corpus_map: Path, data: Path,
           runs_root: Optional[Path] = None, *, split: str = "val",
           out: Optional[Path] = None) -> dict:
    _, refs = fetch.load(data)
    seen = seen_in_training(data)
    pair_to_index = load_map(corpus_root, corpus_map)
    runs_root = runs_root or (corpus_root / "runs")
    preds = predictions(runs_root, pair_to_index, split=split)
    if not preds:
        raise SystemExit(f"no run records under {runs_root}")

    # Every configuration is scored on the union of indices any of them
    # produced, so a configuration that errored on a function is not rewarded
    # for having fewer, easier items than its neighbours.
    common = sorted(set.intersection(*(set(p) for p in preds.values())))
    rows = []
    for (config, seed), p in sorted(preds.items()):
        p_common = {i: p[i] for i in common}
        rows.append({"system": f"NatComGen {config} seed{seed}",
                     "kind": "ours",
                     **score_subset(refs, p_common),
                     **_slice_scores(refs, lambda i: nl_tokenise(p_common[i]),
                                     common, seen),
                     "raw_split": score_subset(refs, p_common,
                                               tokenise=False)["Ba"]})
    base = baselines_on(refs, data, common)
    for name, sc in base.items():
        lines = (data / f"{name}.out").read_text(
            encoding="utf-8", errors="replace").splitlines()
        rows.append({"system": name, "kind": "published", **sc,
                     **_slice_scores(refs, lambda i: lines[i], common, seen)})

    full = {s: bleu.leclair(refs, (data / f"{s}.out").read_text(
        encoding="utf-8", errors="replace").splitlines()) for s in SYSTEMS}

    # Three different counts, and conflating any two of them misstates the
    # result: how many of SmartDoc's functions were recovered at all, how
    # many of those the run actually produced a comment for, and how many
    # every system is scored on. They coincide only after a complete run.
    cov = {}
    cov_file = data.parent / "coverage_test.json"
    if cov_file.exists():
        cov = json.loads(cov_file.read_text())

    result = {
        "smartdoc_total": len(refs),
        "matched": cov.get("matched"),
        "match_coverage": cov.get("coverage"),
        "scored": len(common),
        "coverage": round(len(common) / len(refs), 4),
        "split": split,
        "rows": sorted(rows, key=lambda r: -r["Ba"]),
        "published_full_1000": full,
        "seen_in_training": len(seen),
        "leakage_full": {s: _slice_scores(
            refs, lambda i, s=s: (data / f"{s}.out").read_text(
                encoding="utf-8", errors="replace").splitlines()[i],
            list(range(len(refs))), seen) for s in SYSTEMS},
    }
    if out:
        # Nothing is overwritten: run 2 writes `smartdoc_2.json` and
        # `smartdoc_2.md`, and `RUNS.md` beside them says which run was
        # which corpus, which configuration and what coverage it got. A
        # benchmark whose results directory holds only the last attempt is
        # one you cannot check a claim against.
        out = Path(out)
        stem = out.stem
        v = RunVersion.open(out.parent, [f"{stem}.json", f"{stem}.md"])
        result["run"] = v.index
        paths = {"json": v.write_json(f"{stem}.json", result),
                 "md": v.write_text(f"{stem}.md", markdown(result))}
        v.record(benchmark="smartdoc", split=split,
                 matched=result.get("matched"), scored=result["scored"],
                 systems=sorted({r["system"] for r in result["rows"]
                                 if r["kind"] == "ours"}))
        result["written"] = {k: str(p) for k, p in paths.items()}
    return result


def markdown(r: dict) -> str:
    n, total = r["scored"], r["smartdoc_total"]
    m = r.get("matched")
    head = (f"Re-grounding recovered the original contract for **{m} of "
            f"{total}** SmartDoc test functions"
            if m else f"Scored against the SmartDoc test set")
    tail = (f", and this run produced a comment for **{n}** of them."
            if m and m != n else f" — **{n} of {total}**.")
    L = ["# SmartDoc re-grounded benchmark",
         "",
         head + tail + " Σ(f) on those is computed from the real compilation "
         "unit, not reconstructed. Every system in the table below is scored "
         f"on the same {n} indices.",
         "",
         "| system | Ba | B1 | B2 | B3 | B4 | n |",
         "|---|---|---|---|---|---|---|"]
    for row in r["rows"]:
        L.append(f"| {row['system']} | {row['Ba']} | {row['B1']} | "
                 f"{row['B2']} | {row['B3']} | {row['B4']} | {row['n']} |")
    if any(row.get("n_seen") for row in r["rows"]):
        L += ["",
              "## Split by whether the reference was memorisable",
              "",
              f"Of the {n} scored functions, **{r['rows'][0]['n_seen']}** have "
              "a reference comment that appears *verbatim* in SmartDoc's own "
              "training set, because their split is random over one crawl and "
              "deployed Solidity is mostly copies of the same contracts. "
              "The `unseen` column is the one that measures generalisation.",
              "",
              "| system | seen in train | unseen |",
              "|---|---|---|"]
        for row in r["rows"]:
            L.append(f"| {row['system']} | {row.get('seen')} "
                     f"| {row.get('unseen')} |")
        L += ["",
              f"On the full {total}, the same split gives:",
              "",
              "| system | all | seen in train | unseen |", "|---|---|---|---|"]
        for s_, sc in r.get("leakage_full", {}).items():
            L.append(f"| {s_} | {sc['all']} | {sc['seen']} | {sc['unseen']} |")
        L.append("")
    L += ["",
          f"For reference, the published systems on the **full {total}**, "
          "reproduced here from their released outputs:",
          "",
          "| system | Ba |", "|---|---|"]
    for s, sc in r["published_full_1000"].items():
        L.append(f"| {s} | {sc['Ba']} |")
    L += ["",
          "A gap between a system's full-1000 score and its score on the "
          "matched subset is a property of *which* functions were "
          "recoverable, not of any system in the table. It is the first "
          "thing to look at before reading anything into the comparison.",
          ""]
    return "\n".join(L)


def main(argv=None) -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=here / "data/smartdoc_corpus")
    ap.add_argument("--corpus-map", type=Path,
                    default=here / "data/corpus_map_test.json")
    ap.add_argument("--data", type=Path, default=here / "data/smartdoc")
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, default=here / "results/smartdoc.json")
    a = ap.parse_args(argv)
    r = report(a.corpus, a.corpus_map, a.data, a.runs, split=a.split,
               out=a.out)
    print(markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
