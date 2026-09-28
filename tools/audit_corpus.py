"""Is the corpus complete enough to test the system on, and is it honest?

Seven checks, in order of how badly a failure would invalidate results. The first
four are hard invariants and make this exit non-zero; the last two are coverage
reports, where the answer is a number to know rather than a pass or a fail.

    python3 tools/audit_corpus.py                   # data/NatSpecGold
    python3 tools/audit_corpus.py --corpus DIR --json audit.json

ONE FALSE ALARM, DOCUMENTED SO NOBODY RE-DISCOVERS IT. Re-running the scorer on
each pair's `doc_raw` reports param_missing, no_text and return_missing on 89
pairs. Those 89 are exactly the pairs whose documentation is `@inheritdoc`: the
prose and the tags live in the base contract and were resolved into the
notice/dev/params/returns fields when the corpus was built, so `doc_raw` really
does carry no @param lines. The reference the system is scored against is the
resolved one, and that is what check 1 judges. Judging `doc_raw` is the mistake.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Optional

import sys

# Runnable as `python3 tools/<this>.py` from the repo root or anywhere
# else: the package sits one directory up, not on the default path.
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from natspec_corpus.gate import _declared_params, judge_text

KINDS = ("function", "constructor", "modifier", "receive", "fallback",
         "event", "error")


def load(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def as_reference(pair: dict) -> str:
    """The resolved reference, as NatSpec: what BLEU is scored against."""
    out = []
    if pair.get("notice"):
        out.append("/// @notice " + pair["notice"])
    if pair.get("dev"):
        out.append("/// @dev " + pair["dev"])
    for name, text in (pair.get("params") or {}).items():
        out.append(f"/// @param {name} {text}")
    for ret in (pair.get("returns") or []):
        name = (ret.get("name") or "").strip()
        body = (name + " " if name else "") + (ret.get("text") or "")
        out.append(("/// @return " + body).rstrip())
    return "\n".join(out)


def norm(text: Optional[str]) -> str:
    return " ".join((text or "").lower().split())


def declared_returns(pair: dict) -> Optional[int]:
    """How many values the DECLARATION returns, parsed from the pair's code.

    This exists because check 1 cannot see a missing @return. The scorer counts
    a declaration's returns from `pair["returns"]` — the very field the
    reference is built from — so the two agree by construction and a return the
    corpus simply never recorded is invisible. Parsing the signature is the only
    independent witness.

    None when the code cannot be parsed, which is reported separately rather
    than counted as agreement.
    """
    from natspec_corpus import solidity as S
    from natspec_corpus.masking import code_view, scan
    code = pair.get("code", "")
    try:
        spans = scan(code, strict=False)
        decls = S.find_decls(code_view(code, spans), code)
        if not decls:
            return None
        return len(decls[0].returns)
    except Exception:                                    # noqa: BLE001
        return None


def audit(corpus: Path) -> dict:
    pairs = load(corpus / "pairs.jsonl")
    partial_path = corpus / "pairs_partial.jsonl"
    partial = load(partial_path) if partial_path.exists() else []

    sigma_ids = set()
    sigma_path = corpus / "sigma" / "sigma.jsonl"
    if sigma_path.exists():
        for line in sigma_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sigma_ids.add(json.loads(line).get("pair_id"))

    by_split: Dict[str, List[dict]] = collections.defaultdict(list)
    for p in pairs:
        by_split[p.get("split") or "?"].append(p)
    out: Dict[str, object] = {"pairs": len(pairs), "partial": len(partial),
                              "sigma_tables": len(sigma_ids)}

    # --- 1. every gold satisfies its own declaration --------------------
    defects = collections.Counter()
    offenders = []
    for p in pairs:
        labels = judge_text(p, as_reference(p))
        if labels:
            offenders.append({"id": p["id"], "name": p["name"],
                              "file": p["file"], "defects": labels})
            for label in labels:
                defects[label.split(":")[0]] += 1
    out["gold_defects"] = dict(defects)
    out["gold_offenders"] = offenders[:20]
    out["gold_complete"] = not offenders

    # --- 2. the critic's reconstruction agrees with the corpus ----------
    # `judge_text` — and therefore ComGen's deterministic critic — recovers
    # parameter names by re-parsing the pair's own code. If that disagrees with
    # the names the corpus recorded, the critic reports defects that are not
    # there, and every defect number in every table is wrong.
    recon_bad = []
    for p in pairs:
        recorded = set((p.get("params") or {}).keys())
        if not recorded:
            continue
        recon = set(_declared_params(p))
        if recorded - recon:
            recon_bad.append({"id": p["id"], "signature": p["signature"],
                              "recorded": sorted(recorded),
                              "reconstructed": sorted(recon)})
    out["reconstruction_mismatches"] = recon_bad[:20]
    out["reconstruction_agrees"] = not recon_bad

    # --- 2b. every declared return value is recorded, and so documented -
    # The question this audit was asked: does the corpus carry a @return for
    # every value a function returns? Check 1 cannot answer it (see
    # `declared_returns`), so the signature is parsed independently and its
    # count compared with what the corpus recorded.
    ret_bad, ret_unparsed = [], 0
    for p in pairs:
        if p["kind"] not in ("function", "receive", "fallback"):
            continue
        declared = declared_returns(p)
        if declared is None:
            ret_unparsed += 1
            continue
        recorded = len(p.get("returns") or [])
        if declared != recorded:
            ret_bad.append({"id": p["id"], "signature": p["signature"],
                            "file": p["file"], "declared": declared,
                            "recorded": recorded})
    out["return_mismatches"] = ret_bad[:20]
    out["return_mismatch_count"] = len(ret_bad)
    out["returns_unparsed"] = ret_unparsed
    out["returns_complete"] = not ret_bad

    # --- 3. the splits are project-disjoint ----------------------------
    projects = {s: {p["project"] for p in v} for s, v in by_split.items()}
    overlaps = {}
    names = sorted(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(projects[a] & projects[b])
            if shared:
                overlaps[f"{a}|{b}"] = shared
    out["splits"] = {s: {"pairs": len(v), "projects": len(projects[s]),
                         "files": len({p["file"] for p in v})}
                     for s, v in sorted(by_split.items())}
    out["project_overlaps"] = overlaps
    out["splits_disjoint"] = not overlaps

    # --- 4. no reference is sitting verbatim in the training split -----
    # SmartDoc's published 47.39 rests on 434 of its 1,000 test references
    # appearing verbatim in their own training file. This is the check that
    # says whether our number is of the same kind.
    train_text = {norm(p.get("notice")) for p in by_split.get("train", ())}
    train_text |= {norm(p.get("dev")) for p in by_split.get("train", ())}
    train_text.discard("")
    leaks = {}
    for split in ("val", "test"):
        hits = [p["id"] for p in by_split.get(split, ())
                if norm(p.get("notice")) and norm(p["notice"]) in train_text]
        leaks[split] = {"n": len(hits), "of": len(by_split.get(split, ())),
                        "ids": hits[:10]}
    out["leakage"] = leaks
    out["no_leakage"] = all(v["n"] == 0 for v in leaks.values())

    # --- 5. Σ(f) coverage, which is what the evidence-grounding needs --
    coverage = {}
    for split, v in sorted(by_split.items()):
        rows = {}
        for kind in KINDS:
            k = [p for p in v if p["kind"] == kind]
            if k:
                have = sum(1 for p in k if p["id"] in sigma_ids)
                rows[kind] = {"pairs": len(k), "with_sigma": have,
                              "coverage": round(have / len(k), 3)}
        have = sum(1 for p in v if p["id"] in sigma_ids)
        rows["ALL"] = {"pairs": len(v), "with_sigma": have,
                       "coverage": round(have / len(v), 3) if v else 0.0}
        coverage[split] = rows
    out["sigma_coverage"] = coverage

    # --- 6. what the build itself reported -----------------------------
    for name in ("build_report.json", "sigma_report.json",
                 "retrieval_report.json"):
        path = corpus / name
        if path.exists():
            out[name[:-5]] = json.loads(path.read_text(encoding="utf-8"))
    return out


def report(a: dict) -> None:
    ok = "PASS"
    bad = "FAIL"
    print("# Corpus audit\n")
    print(f"{a['pairs']} complete pairs, {a['partial']} quarantined as "
          f"incomplete, {a['sigma_tables']} fact tables\n")

    print("## Hard invariants\n")
    print(f"  [{ok if a['gold_complete'] else bad}] every gold documents every "
          f"parameter and return it declares")
    if not a["gold_complete"]:
        for d in a["gold_offenders"]:
            print(f"        {d['name']} in {d['file']}: {d['defects']}")
    print(f"  [{ok if a['reconstruction_agrees'] else bad}] the deterministic "
          f"critic's parameter reconstruction agrees with the corpus")
    for m in a["reconstruction_mismatches"]:
        print(f"        {m['signature']}: recorded {m['recorded']} "
              f"reconstructed {m['reconstructed']}")
    print(f"  [{ok if a['returns_complete'] else bad}] every return value the "
          f"signature declares is recorded and documented "
          f"({a['return_mismatch_count']} mismatched, "
          f"{a['returns_unparsed']} signatures unparsed)")
    for m in a["return_mismatches"]:
        print(f"        {m['signature']} in {m['file']}: declares "
              f"{m['declared']}, corpus records {m['recorded']}")
    print(f"  [{ok if a['splits_disjoint'] else bad}] the splits share no "
          f"project")
    for k, v in a["project_overlaps"].items():
        print(f"        {k}: {v}")
    print(f"  [{ok if a['no_leakage'] else bad}] no reference appears verbatim "
          f"in the training split")
    for split, v in a["leakage"].items():
        print(f"        {split}: {v['n']} of {v['of']}")

    print("\n## Splits\n")
    print(f"  {'split':8} {'pairs':>6} {'projects':>9} {'files':>6}")
    for s, v in a["splits"].items():
        print(f"  {s:8} {v['pairs']:6} {v['projects']:9} {v['files']:6}")

    print("\n## Σ(f) coverage — what the evidence grounding can reach\n")
    for split, rows in a["sigma_coverage"].items():
        print(f"  {split}")
        for kind, v in rows.items():
            flag = "  <-- no evidence at all" if v["coverage"] == 0 else ""
            print(f"    {kind:12} {v['with_sigma']:4}/{v['pairs']:<4} "
                  f"{v['coverage']:.3f}{flag}")

    sr = a.get("sigma_report") or {}
    if sr:
        print(f"\n## Why coverage is what it is\n")
        print(f"  {sr.get('compiled')} of {sr.get('scored_files')} files "
              f"compiled; {sr.get('unresolved_imports')} had unresolved "
              f"imports; {sr.get('compile_failed')} compile failures, "
              f"{sr.get('slither_failed')} analysis failures.")
        print(f"  A fact table needs a compiling file, so file compilation is "
              f"the ceiling on Σ(f) coverage.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("data/NatSpecGold"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    if not (args.corpus / "pairs.jsonl").exists():
        raise SystemExit(f"{args.corpus} has no pairs.jsonl")
    a = audit(args.corpus)
    report(a)
    if args.json:
        args.json.write_text(json.dumps(a, indent=1), encoding="utf-8")
        print(f"\nwritten to {args.json}")
    hard = ("gold_complete", "reconstruction_agrees", "returns_complete",
            "splits_disjoint", "no_leakage")
    return 0 if all(a[k] for k in hard) else 1


if __name__ == "__main__":
    raise SystemExit(main())
