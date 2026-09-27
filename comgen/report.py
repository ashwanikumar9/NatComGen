"""ComGen's tables, written into `comgen/results/`.

The six tables NatComGen already produces are produced here unchanged, by
calling `natspec_corpus.report.write_report` with ComGen's own output directory
— so the metrics, the significance testing, the LaTeX and the result versioning
are all the same code, and a ComGen row is comparable to a C1 row because it
was computed by the same function.

Two tables are new, and they exist because ComGen's cost is no longer fixed:

  rounds  how far into its budget each configuration actually went, and what
          the extra rounds bought. G1 against G3 is the round-count result.
  cost    calls per function, measured rather than assumed. C1 is 5 by
          construction; ComGen is between 4 and 8 and the mean is the number
          that belongs next to any quality claim.

On run numbering: both new tables are opened over the same target list
`write_report` uses, plus themselves, so they take the same run number as the
main tables. Every file from one run therefore carries one number, which is the
whole point of `versioning.py` — a `rounds_3.md` that did not belong to
`main_3.md` would be worse than no table at all.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from natspec_corpus import reproduce, versioning as _V
from natspec_corpus.evaluate import aggregate, score_record
from natspec_corpus.report import markdown_table, write_report
from natspec_corpus.stats import ablation_table

from . import RESULTS
from .experiment import BY_NAME, RUN_ORDER

#: Kept in step with `report.write_report`'s own list. See the docstring.
_MAIN_TARGETS = ([f"tables/{n}.{ext}"
                  for n in ("main", "main_with_sigma", "main_without_sigma",
                            "conditions", "ablations", "seeds")
                  for ext in ("md", "tex")]
                 + ["figures/ablations.png", "manifest.json"])

_EXTRA_TARGETS = ["tables/rounds.md", "tables/cost.md"]


def _fmt(v, places: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{places}f}"
    return str(v)


def _critics_ran(rec: dict) -> tuple:
    """(deterministic ran, semantic ran) on this record's first round."""
    r0 = (rec.get("rounds") or [{}])[0]
    return (bool(r0.get("deterministic", {}).get("ran")),
            bool(r0.get("semantic", {}).get("ran")))


def rounds_rows(rows: Sequence[dict]) -> tuple:
    """Per configuration: how far the loop went and what it changed.

    A column is blank where nothing measured it. G0 runs no critic at all, so
    "clean at round 0" would otherwise read 1.000 — every draft passes a
    critique that never happened — and G5 has no deterministic critic, so
    "still blocking" would read 0.000 for the same empty reason. A blank is a
    true statement about what was measured; a 1.000 is not.
    """
    headers = ["config", "n", "rounds allowed", "mean rounds", "clean at r0",
               "improved by revision", "best round > 0", "still blocking"]
    out: List[List[str]] = []
    for name in RUN_ORDER:
        group = [r for r in rows if r.get("config") == name
                 and not r.get("error")]
        if not group:
            continue
        n = len(group)
        used = [r.get("rounds_used", 1) for r in group]

        # Any critic at all is enough to say whether a draft was clean.
        judged = [r for r in group if any(_critics_ran(r))]
        # Only the deterministic critic decides whether a defect is blocking.
        measured = [r for r in group
                    if (r.get("gate") or {}).get("passed") is not None]

        clean0 = sum(1 for r in judged
                     if (r.get("rounds") or [{}])[0].get("total") == 0)
        later = sum(1 for r in judged if r.get("best_round", 0) > 0)
        improved = 0
        for r in judged:
            rs = r.get("rounds") or []
            if len(rs) > 1 and rs[r.get("best_round", 0)]["total"] < rs[0]["total"]:
                improved += 1
        blocking = sum(1 for r in measured if not r["gate"]["passed"])

        share = lambda k, of: _fmt(k / len(of)) if of else "—"
        out.append([name, n, group[0].get("rounds_allowed", "—"),
                    _fmt(sum(used) / n, 2), share(clean0, judged),
                    share(improved, judged), share(later, judged),
                    share(blocking, measured)])
    return headers, out


def cost_rows(rows: Sequence[dict]) -> tuple:
    """Per configuration: calls per function, by prompt, measured."""
    headers = ["config", "n", "mean calls", "min", "max", "L9", "L7", "L1b",
               "L2", "R1", "L8"]
    out: List[List[str]] = []
    for name in RUN_ORDER:
        group = [r for r in rows if r.get("config") == name
                 and not r.get("error")]
        if not group:
            continue
        n = len(group)
        totals = [(r.get("calls") or {}).get("total", 0) for r in group]
        per: Dict[str, int] = defaultdict(int)
        for r in group:
            for pid, k in (r.get("calls") or {}).items():
                if pid != "total":
                    per[pid] += k
        out.append([name, n, _fmt(sum(totals) / n, 2), min(totals),
                    max(totals)]
                   + [_fmt(per.get(p, 0) / n, 2)
                      for p in ("L9", "L7", "L1b", "L2", "R1", "L8")])
    return headers, out


def write(corpus_root: Path, rows: Sequence[dict], *,
          out_dir: Optional[Path] = None, split: str = "val",
          models: Optional[dict] = None,
          baseline: str = "G1") -> Dict[str, Path]:
    """Every table, into `comgen/results/` unless told otherwise."""
    corpus_root = Path(corpus_root)
    out_dir = Path(out_dir or RESULTS)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    pairs = {json.loads(l)["id"]: json.loads(l)
             for l in (corpus_root / "pairs.jsonl").read_text(
                 encoding="utf-8").splitlines() if l.strip()}
    rows = [r for r in rows if r.get("pair_id") in pairs]
    if not rows:
        raise SystemExit("no ComGen runs to report on")

    per_seed: Dict[str, Dict[int, dict]] = defaultdict(dict)
    results: Dict[str, dict] = {}
    for name in RUN_ORDER:
        seeds = sorted({r["seed"] for r in rows if r.get("config") == name})
        if not seeds:
            continue
        for seed in seeds:
            scored = [score_record(r, pairs[r["pair_id"]]) for r in rows
                      if r.get("config") == name and r.get("seed") == seed
                      and not r.get("error")]
            per_seed[name][seed] = {
                s["pair_id"]: (s["support_rate"]
                               if s["support_rate"] is not None
                               else (s["by_kind"].get("notice") or {}).get(
                                   "bleu", 0.0))
                for s in scored}
            if seed == seeds[0]:
                results[name] = aggregate(scored)

    abl = None
    if baseline in per_seed and len(per_seed) > 1:
        abl = ablation_table(per_seed[baseline],
                             {n: v for n, v in per_seed.items()
                              if n != baseline})

    version = _V.RunVersion.open(out_dir, _MAIN_TARGETS + _EXTRA_TARGETS)
    written: Dict[str, Path] = {}
    h, r = rounds_rows(rows)
    written["tables/rounds.md"] = version.write_text(
        "tables/rounds.md",
        "# ComGen — the loop\n\n" + markdown_table(h, r)
        + "\n\n`clean at r0` is the share of functions whose first draft the "
          "Critic Group passed outright: the loop was not needed. `improved by "
          "revision` is the share where a later round had strictly fewer "
          "defects than the draft — the loop's yield. `still blocking` is the "
          "share whose winning round still carries a deterministic defect.\n")
    h, r = cost_rows(rows)
    written["tables/cost.md"] = version.write_text(
        "tables/cost.md",
        "# ComGen — calls per function\n\n" + markdown_table(h, r)
        + "\n\nMeasured, not assumed: the loop makes cost a distribution "
          "rather than a constant. C1's comparable figure is exactly 5.0. A "
          "cached call is counted here — it was issued; whether it was paid "
          "for is in the call summary.\n")

    man = reproduce.manifest(corpus_root, models=models or {},
                            seeds=sorted({r["seed"] for r in rows}),
                            configs=sorted(results), split=split)
    written.update(write_report(
        out_dir, results=results,
        labels={n: BY_NAME[n].label for n in results},
        ablations=abl, manifest=man,
        run_meta={"architecture": "comgen", "split": split,
                  "seeds": sorted({r["seed"] for r in rows}),
                  "configs": sorted(results), "functions": len(pairs),
                  "baseline": baseline,
                  "models": (models or None)}))
    return written
