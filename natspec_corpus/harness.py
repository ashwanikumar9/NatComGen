"""Prompt validation, before any full run.

Milestones M1 and M2 from the plan. Both answer questions that are cheap now
and expensive later:

  M1  Does each prompt return schema-valid JSON on the first attempt?
      A prompt that needs a retry a fifth of the time produces a distribution
      of outputs shaped partly by the repair message, which is not the thing
      being measured.

  M2  Does the semantic critic fire on defects that are known to be there?
      The corpus already carries 566 partial pairs with 1346 labelled
      defects. Feeding the critic a comment whose flaw is known is the only
      cheap measurement of whether it works at all. A critic that flags
      everything is worse than no critic, so the false-positive rate on the
      verified pairs is measured in the same pass.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from . import prompts_v3 as P
from .llm import Client, LLMError
from .runner import Context


def parse_rates(corpus_root: Path, client: Client, *, n: int = 20,
                seed: int = 0, split: str = "val") -> dict:
    """M1: run all five prompts over `n` functions and report parse rates."""
    from .runner import load_contexts
    contexts = load_contexts(corpus_root, split, with_units=False)
    rng = random.Random(seed)
    rng.shuffle(contexts)
    sample = [c for c in contexts if c.has_sigma][:n] or contexts[:n]

    failures: List[dict] = []
    for ctx in sample:
        sigma = ctx.sigma_text()
        try:
            intent = client.call(P.INTENT_REASONER, source=ctx.source,
                                 sigma=sigma).data
            gen = client.call(P.GENERATOR, source=ctx.source, sigma=sigma,
                              intent=json.dumps(intent),
                              exemplars="(none)").data
            crit = client.call(P.SEMANTIC_CRITIC, source=ctx.source,
                               sigma=sigma,
                               candidate=gen.get("natspec", "")).data
            ref = client.call(P.REFINER, source=ctx.source, sigma=sigma,
                              candidate=gen.get("natspec", ""),
                              critique=json.dumps(crit)).data
            client.call(P.CLAIM_VERIFIER, source=ctx.source, sigma=sigma,
                        candidate=ref.get("natspec", ""),
                        claims=json.dumps(gen.get("claims", [])))
        except LLMError as e:
            failures.append({"pair_id": ctx.pair["id"], "error": str(e)[:200]})

    stats = client.stats()
    return {
        "sampled": len(sample),
        "hard_failures": failures,
        "per_prompt": {k: {"calls": v["calls"],
                           "first_attempt_parse_rate":
                               round(v["first_attempt_parse_rate"], 3),
                           "mean_seconds": round(v["mean_seconds"], 2)}
                       for k, v in stats.items()},
        "gate_m1_met": all(v["first_attempt_parse_rate"] >= 0.95
                           for v in stats.values()) and not failures,
    }


def critic_calibration(corpus_root: Path, client: Client, *,
                       n_partial: int = 60, n_verified: int = 40,
                       seed: int = 0) -> dict:
    """M2: does L2 fire on known defects, and stay quiet on clean comments?

    Recall is measured per defect family rather than overall. `param_missing`
    dominates the corpus three to one, so a single overall number would be
    almost entirely a measure of that one family.
    """
    sigma_tables: Dict[str, dict] = {}
    sig = corpus_root / "sigma" / "sigma.jsonl"
    if sig.exists():
        for line in sig.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                if t.get("pair_id"):
                    sigma_tables[t["pair_id"]] = t

    def load(name: str) -> List[dict]:
        p = corpus_root / name
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()] if p.exists() else []

    rng = random.Random(seed)
    partial = [p for p in load("pairs_partial.jsonl") if p["id"] in sigma_tables]
    verified = [p for p in load("pairs.jsonl") if p["id"] in sigma_tables]
    rng.shuffle(partial)
    rng.shuffle(verified)
    partial, verified = partial[:n_partial], verified[:n_verified]

    fired: Dict[str, List[bool]] = {}
    rows: List[dict] = []
    for pair in partial:
        ctx = Context(pair=pair, table=sigma_tables.get(pair["id"]))
        try:
            out = client.call(P.SEMANTIC_CRITIC, source=ctx.source,
                              sigma=ctx.sigma_text(),
                              candidate=pair["doc_raw"]).data
        except LLMError:
            continue
        hit = bool(out.get("defects"))
        for d in pair["defects"]:
            fired.setdefault(d.split(":")[0], []).append(hit)
        rows.append({"pair_id": pair["id"], "labelled": pair["defects"],
                     "fired": hit, "reported": out.get("defects", [])})

    false_pos = 0
    checked = 0
    for pair in verified:
        ctx = Context(pair=pair, table=sigma_tables.get(pair["id"]))
        try:
            out = client.call(P.SEMANTIC_CRITIC, source=ctx.source,
                              sigma=ctx.sigma_text(),
                              candidate=pair["doc_raw"]).data
        except LLMError:
            continue
        checked += 1
        high = [d for d in out.get("defects", [])
                if d.get("severity") in ("high", "medium")]
        false_pos += int(bool(high))

    recall = {k: sum(v) / len(v) for k, v in fired.items() if v}
    fp_rate = false_pos / checked if checked else None
    return {
        "partial_checked": len(rows), "verified_checked": checked,
        "recall_by_defect": {k: round(v, 3) for k, v in sorted(recall.items())},
        "false_positive_rate": round(fp_rate, 3) if fp_rate is not None else None,
        "gate_m2_met": (
            recall.get("param_missing", 0) >= 0.70
            and recall.get("return_missing", 0) >= 0.70
            and (fp_rate is not None and fp_rate < 0.15)),
        "rows": rows[:20],
    }


def verifier_stability(corpus_root: Path, client: Client, *, n: int = 20,
                       seed: int = 0, split: str = "val") -> dict:
    """M5's check: does L8's verdict on a claim depend on the claims around it?

    Each function is verified twice, once with the claim list reversed. A
    verdict that moves means the verifier is reading position, not evidence —
    the failure mode documented for LLM judges (arXiv:2406.07791) — and each
    claim then needs its own call.
    """
    from .runner import load_contexts
    contexts = load_contexts(corpus_root, split, with_units=False)
    rng = random.Random(seed)
    rng.shuffle(contexts)
    sample = [c for c in contexts if c.has_sigma][:n]

    agree = total = 0
    flips: List[dict] = []
    for ctx in sample:
        sigma = ctx.sigma_text()
        claims = [{"text": t, "ids": []} for t in
                  [ln.strip().lstrip("/ *") for ln in
                   ctx.pair["doc_raw"].splitlines() if ln.strip()]][:6]
        if len(claims) < 2:
            continue
        try:
            a = client.call(P.CLAIM_VERIFIER, source=ctx.source, sigma=sigma,
                            candidate=ctx.pair["doc_raw"],
                            claims=json.dumps(claims)).data
            b = client.call(P.CLAIM_VERIFIER, source=ctx.source, sigma=sigma,
                            candidate=ctx.pair["doc_raw"],
                            claims=json.dumps(list(reversed(claims)))).data
        except LLMError:
            continue
        va = {v.get("claim"): v.get("verdict") for v in a.get("verdicts", [])}
        vb = {v.get("claim"): v.get("verdict") for v in b.get("verdicts", [])}
        for k in set(va) & set(vb):
            total += 1
            if va[k] == vb[k]:
                agree += 1
            else:
                flips.append({"claim": k[:80], "forward": va[k],
                              "reversed": vb[k]})
    return {"claims_compared": total,
            "agreement": round(agree / total, 3) if total else None,
            "flips": flips[:10],
            "gate_met": bool(total) and agree / total >= 0.90}
