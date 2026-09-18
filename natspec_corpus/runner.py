"""The five calls, in order, for one function — and for a whole split.

Every intermediate output is kept. Without the intent reading, the draft, the
critique and the gate verdict, a bad final comment cannot be attributed to the
generator, the critic or the refiner, and the ablations have nothing to stand
on. The record per function is the experiment's raw data.

Resumable by construction: the output is JSONL, one line per function, and a
restart skips the ids already written. A run that dies at function 300 costs
300 functions of compute, not 300 functions of work.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import prompts_v3 as P
from .errors import CorpusError
from .gate import GateResult, judge_text, run_gate
from .llm import Client, LLMError
from .views import views_for


@dataclass
class Context:
    """Everything one function needs, assembled before any model is called."""
    pair: dict
    table: Optional[dict]
    unit: Optional[Any] = None
    version: Optional[str] = None

    @property
    def has_sigma(self) -> bool:
        return bool(self.table)

    @property
    def source(self) -> str:
        return self.pair["code"]

    def sigma_text(self) -> str:
        if not self.table:
            return ("[FACTS]\n  (no fact table: this file does not compile "
                    "from the corpus alone, so no structural evidence is "
                    "available. Claim only what the source itself shows.)")
        return P.sigma_block(self.table)


def generate(ctx: Context, client: Client, index=None, *,
             top_k: int = P.RETRIEVAL_CONFIG["top_k"],
             tau: float = P.RETRIEVAL_CONFIG["threshold_tau"],
             stages: Optional[List[str]] = None) -> dict:
    """Run the pipeline for one function and return its record.

    `stages` switches calls off for the ablations: pass a subset of
    ("L7", "L1b", "L2", "L3", "L8"). L1b is always required — it is the only
    call that writes a comment.
    """
    stages = stages or ["L7", "L1b", "L2", "L3", "L8"]
    if "L1b" not in stages:
        raise CorpusError("L1b is the only stage that produces a comment")

    rec: Dict[str, Any] = {
        "pair_id": ctx.pair["id"], "file": ctx.pair["file"],
        "function": ctx.pair["name"], "signature": ctx.pair["signature"],
        "split": ctx.pair.get("split"), "has_sigma": ctx.has_sigma,
        "stages": stages, "started": time.time(),
    }
    sigma = ctx.sigma_text()

    exemplars, hits = "(retrieval disabled)", []
    if index is not None:
        hits = index.search(views_for(ctx.pair, ctx.table), top_k=top_k,
                            tau=tau)
        exemplars = P.exemplar_block(h.as_exemplar() for h in hits)
    rec["retrieved"] = [{"pair_id": h.unit.pair_id, "score": round(h.score, 4)}
                        for h in hits]

    intent_text = "(no intent reading was produced)"
    if "L7" in stages:
        c = client.call(P.INTENT_REASONER, source=ctx.source, sigma=sigma)
        rec["intent"] = c.data
        rec["call_L7"] = c.summary()
        intent_text = json.dumps(c.data, indent=1)

    c = client.call(P.GENERATOR, source=ctx.source, sigma=sigma,
                    intent=intent_text, exemplars=exemplars)
    draft = c.data.get("natspec", "")
    claims = c.data.get("claims", [])
    rec["draft"] = draft
    rec["draft_claims"] = claims
    rec["call_L1b"] = c.summary()

    final, gate = draft, None
    if "L2" in stages:
        c = client.call(P.SEMANTIC_CRITIC, source=ctx.source, sigma=sigma,
                        candidate=draft)
        rec["critique"] = c.data
        rec["call_L2"] = c.summary()

        if "L3" in stages:
            c = client.call(P.REFINER, source=ctx.source, sigma=sigma,
                            candidate=draft,
                            critique=json.dumps(rec["critique"], indent=1))
            refined = c.data.get("natspec", draft)
            rec["refined"] = refined
            rec["call_L3"] = c.summary()
            if c.data.get("claims"):
                claims = c.data["claims"]

            if ctx.unit is not None:
                gate = run_gate(ctx.unit, ctx.pair["file"], ctx.pair,
                                draft, refined, ctx.version)
            else:
                # No compilation unit: the compiler half cannot run, so the
                # gate falls back to the defect score alone. Recorded, never
                # silently treated as a pass.
                before, after = judge_text(ctx.pair, draft), judge_text(
                    ctx.pair, refined)
                gate = GateResult(passed=len(after) <= len(before),
                                  compiles=False, tags_emitted=False,
                                  score_kept=len(after) <= len(before),
                                  draft_defects=before, refined_defects=after,
                                  reasons=["no compilation unit; score only"])
            rec["gate"] = gate.to_dict()
            final = refined if gate.passed else draft
            rec["gate_kept_draft"] = not gate.passed

    rec["final"] = final
    rec["final_claims"] = claims

    if "L8" in stages:
        c = client.call(P.CLAIM_VERIFIER, source=ctx.source, sigma=sigma,
                        candidate=final, claims=json.dumps(claims, indent=1))
        rec["verification"] = c.data
        rec["call_L8"] = c.summary()
        verdicts = c.data.get("verdicts", [])
        bad = [v for v in verdicts if v.get("verdict") == "CONTRADICTED"]
        rec["gate_verdict"] = c.data.get("gate", "PASS" if not bad else "FAIL")
        rec["support_rate"] = (
            sum(1 for v in verdicts if v.get("verdict") == "SUPPORTED")
            / len(verdicts)) if verdicts else None
        if bad:
            rec["final"] = strip_claims(final, [v["claim"] for v in bad])
            rec["stripped"] = [v["claim"] for v in bad]

    rec["seconds"] = round(time.time() - rec.pop("started"), 3)
    return rec


def strip_claims(natspec: str, claims: List[str]) -> str:
    """Remove the lines carrying contradicted claims.

    Dropping a sentence is always safe; keeping a contradicted one is not.
    Matching is on a normalised substring, and a line that cannot be matched
    is left alone rather than guessed at.
    """
    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    targets = [norm(c) for c in claims if c.strip()]
    out = []
    for line in natspec.splitlines():
        body = norm(line.lstrip("/ *"))
        if body and any(t and (t in body or body in t) for t in targets):
            continue
        out.append(line)
    return "\n".join(out).strip() or natspec


# --------------------------------------------------------------------------
# a whole split
# --------------------------------------------------------------------------

def load_contexts(corpus_root: Path, split: str,
                  with_units: bool = True) -> List[Context]:
    """Pairs of `split`, each with its fact table when one exists.

    Where a fact table is missing the pipeline still runs: the evidence block
    says so explicitly and the record carries `has_sigma: false`, so those
    functions are reported separately rather than averaged in silently.
    """
    from .compile import compile_unit, unit_for

    tables: Dict[str, dict] = {}
    sig = corpus_root / "sigma" / "sigma.jsonl"
    if sig.exists():
        for line in sig.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                if t.get("pair_id"):
                    tables[t["pair_id"]] = t

    contracts = corpus_root / "contracts"

    def read(rel: str) -> Optional[str]:
        p = contracts / rel
        return p.read_text(encoding="utf-8") if p.is_file() else None

    units: Dict[str, Any] = {}
    out: List[Context] = []
    for line in (corpus_root / "pairs.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        if pair["split"] != split:
            continue
        ctx = Context(pair=pair, table=tables.get(pair["id"]))
        if with_units:
            rel = pair["file"]
            if rel not in units:
                u = unit_for(rel, read)
                res = compile_unit(u, cache_dir=corpus_root / ".compile-cache")
                units[rel] = (u, res.version) if res.ok else (None, None)
            ctx.unit, ctx.version = units[rel]
        out.append(ctx)
    return out


def run_split(corpus_root: Path, split: str, client: Client, index=None, *,
              out_path: Optional[Path] = None, limit: Optional[int] = None,
              stages: Optional[List[str]] = None,
              progress: bool = True) -> Path:
    contexts = load_contexts(corpus_root, split)
    if limit:
        contexts = contexts[:limit]
    out_path = out_path or (corpus_root / "runs" / f"{split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["pair_id"])

    with out_path.open("a", encoding="utf-8") as fh:
        for i, ctx in enumerate(contexts, 1):
            if ctx.pair["id"] in done:
                continue
            try:
                rec = generate(ctx, client, index, stages=stages)
            except LLMError as e:
                rec = {"pair_id": ctx.pair["id"], "file": ctx.pair["file"],
                       "function": ctx.pair["name"], "split": split,
                       "error": str(e)[:400]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if progress:
                print(f"  [{i}/{len(contexts)}] {ctx.pair['name']}"
                      f"{'' if ctx.has_sigma else '  (no sigma)'}")
    return out_path
