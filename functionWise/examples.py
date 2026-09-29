"""Side-by-side gold and generated comments, for the qualitative section.

Every number in this package is an aggregate. A paper also needs the other
thing: a handful of real functions where a reader can see what the system
wrote and judge it. This prints them as markdown -- code, gold NatSpec, each
configuration's output, and the per-example scores that explain why the
aggregate looks the way it does.

    python3 -m functionWise.examples --split test --configs G8,G1,C1 --n 6
    python3 -m functionWise.examples --split test --pair-id 75082dfe019df055
    python3 -m functionWise.examples --split val --pick worst --n 4

`--pick` chooses which functions to show, by the first configuration's notice
BLEU:

    spread   (default) evenly spaced across the ranking -- the honest sample
    best     the ones that scored highest
    worst    the ones that scored lowest
    gap      where the configurations disagree most with each other

Prefer `spread` for the paper. Picking only the best examples is the oldest
way to make a qualitative section worthless, and a reviewer who sees four
flawless cases assumes the rest were not.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from natspec_corpus.evaluate import score_record                 # noqa: E402
from natspec_corpus.versioning import next_path                  # noqa: E402
from tools.bleu_table import load                                # noqa: E402

from functionWise import scope, surface                          # noqa: E402


def notice_bleu(rec: dict, pair: dict) -> float:
    s = score_record(rec, pair)
    return (s["by_kind"].get("notice") or {}).get("bleu", 0.0)


def gold_block(pair: dict) -> str:
    out = []
    if pair.get("notice"):
        out.append("@notice  " + pair["notice"].strip())
    if pair.get("dev"):
        out.append("@dev     " + pair["dev"].strip())
    for name, text in (pair.get("params") or {}).items():
        out.append(f"@param   {name} {(text or '').strip()}")
    for r in (pair.get("returns") or []):
        nm = (r.get("name") or "").strip()
        out.append("@return  " + (nm + " " if nm else "") + (r.get("text") or "").strip())
    return "\n".join(out) or "(no gold comment)"


def tidy(natspec: str) -> str:
    """The generated comment as plain tag lines, markers kept, noise dropped."""
    lines = []
    for raw in (natspec or "").splitlines():
        s = raw.strip()
        for lead in ("/**", "*/", "///", "*"):
            if s.startswith(lead):
                s = s[len(lead):].strip()
                break
        if s in ("", "/"):
            continue
        lines.append(s)
    return "\n".join(lines) or "(empty)"


def pick(ids: List[str], rows_by: Dict[str, Dict[str, dict]], pairs: Dict[str, dict],
         configs: Sequence[str], how: str, n: int) -> List[str]:
    first = configs[0]
    scored = []
    for pid in ids:
        rec = rows_by.get(first, {}).get(pid)
        if rec is None:
            continue
        base = notice_bleu(rec, pairs[pid])
        if how == "gap":
            vals = [notice_bleu(rows_by[c][pid], pairs[pid])
                    for c in configs if pid in rows_by.get(c, {})]
            key = max(vals) - min(vals) if len(vals) > 1 else 0.0
        else:
            key = base
        scored.append((key, pid))
    scored.sort(reverse=True)
    if not scored:
        return []
    if how in ("best", "gap"):
        return [p for _, p in scored[:n]]
    if how == "worst":
        return [p for _, p in scored[-n:]]
    step = max(1, len(scored) // max(n, 1))
    return [p for _, p in scored[::step]][:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=HERE / "data/NatSpecGold")
    ap.add_argument("--comgen-runs", type=Path, default=HERE / "comgen/results/runs")
    ap.add_argument("--natcomgen-runs", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="G8,G1,C1")
    ap.add_argument("--kinds", default="function")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--pick", default="spread",
                    choices=("spread", "best", "worst", "gap"))
    ap.add_argument("--pair-id", action="append", default=[])
    ap.add_argument("--code-lines", type=int, default=22)
    ap.add_argument("--out", type=Path, default=HERE / "results/functionwise")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    if args.natcomgen_runs is None:
        args.natcomgen_runs = args.corpus / "runs"

    kinds = scope.parse_kinds(args.kinds)
    pairs = scope.scoped(scope.load_pairs(args.corpus),
                         split=args.split, kinds=kinds)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    rows: List[dict] = []
    for root in (args.comgen_runs, args.natcomgen_runs):
        if Path(root).exists():
            rows += load(Path(root), args.split)
    rows = [r for r in rows if r.get("pair_id") in pairs and not r.get("error")
            and r.get("seed") == args.seed]
    if not rows:
        raise SystemExit(f"no {args.split} rows at seed {args.seed} under "
                         f"{args.comgen_runs} or {args.natcomgen_runs}")

    by: Dict[str, Dict[str, dict]] = {c: {} for c in configs}
    for r in rows:
        if r.get("config") in by:
            by[r["config"]][r["pair_id"]] = r
    missing = [c for c in configs if not by[c]]
    if missing:
        raise SystemExit(f"no rows for {', '.join(missing)} in {args.split} "
                         f"seed {args.seed}")

    common = set.intersection(*[set(by[c]) for c in configs]) & set(pairs)
    chosen = ([p for p in args.pair_id if p in pairs] if args.pair_id
              else pick(sorted(common), by, pairs, configs, args.pick, args.n))
    if not chosen:
        raise SystemExit("nothing to show — check --pair-id or --configs")

    doc = [f"# Qualitative examples — {args.split}, seed {args.seed}, "
           f"{args.pick}\n",
           f"Configurations: {', '.join(configs)}. "
           f"{len(common)} functions were completed by all of them; "
           f"{len(chosen)} shown.\n"]
    for pid in chosen:
        pair = pairs[pid]
        doc.append("\n---\n")
        doc.append(f"## `{pair.get('container')}.{pair.get('signature')}`\n")
        doc.append(f"`{pair.get('project')}` · `{pair.get('file')}` · "
                   f"pair `{pid}`\n")
        code = (pair.get("code") or "").splitlines()
        clipped = code[:args.code_lines]
        if len(code) > args.code_lines:
            clipped.append(f"    // … {len(code) - args.code_lines} more lines")
        doc.append("```solidity\n" + "\n".join(clipped) + "\n```\n")
        doc.append("**Gold**\n\n```\n" + gold_block(pair) + "\n```\n")
        for c in configs:
            rec = by[c].get(pid)
            if rec is None:
                continue
            s = score_record(rec, pair)
            bits = []
            for k in ("notice", "param", "return"):
                blk = s["by_kind"].get(k)
                if blk and blk.get("bleu") is not None:
                    bits.append(f"{k} BLEU {blk['bleu']:.3f}")
            if s.get("support_rate") is not None:
                bits.append(f"claim support {s['support_rate']:.3f}")
            doc.append(f"**{c}** — " + (" · ".join(bits) or "no score") + "\n\n"
                       + "```\n" + tidy(rec.get("final")) + "\n```\n")

    text = "\n".join(doc)
    print(text)
    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        path = next_path(args.out / f"examples_{args.split}_{args.pick}.md")
        path.write_text(text, encoding="utf-8")
        print(f"\nwritten to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
