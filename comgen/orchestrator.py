"""One function, start to finish, under the ComGen architecture.

The shape, and how it differs from `natspec_corpus.runner.generate`:

    contract intent (once per file)  ─┐
    function intent                  ─┴→ Generator ──→ draft
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │              CRITIC GROUP                      │
                    │  deterministic (solc + score)   semantic (LLM)  │
                    └───────────────────────┬───────────────────────┘
                                            │ merged feedback
                            Generator (revising) ──→ next candidate
                                            │  (loop, bounded)
                                          Judge
                                            │
                                     final comment

Three things about the loop are decisions, not details, and they live here
rather than in a prompt:

STOPPING. The loop exits the moment the Critic Group is clean — the
deterministic critic passes and the semantic critic reports no defect.
Otherwise it runs to `rounds` (3 by default). The deterministic critic is the
exit guard because it costs nothing: there is no reason to spend a model call
deciding whether to stop when a compile and a score answer it for free. This is
what the removed v2 gate used to do from *outside* the loop.

BEST ROUND, NOT LAST ROUND. Every round's candidate is kept and the one with
the fewest defects wins — blocking defects first, then total. A revision that
answers one complaint by introducing two is not an improvement, and taking the
last round on faith is how v2's refiner made things worse about as often as
better on the functions where it changed anything. Ties go to the LATER round,
matching what the v2 gate did (`len(after) <= len(before)`): given equal
defect counts, the candidate that has already survived criticism is the better
bet, and its cost is spent either way.

COST IS NOW VARIABLE. Clean on round 0: five calls on the first function of a
file (L9 L7 L1b L2 L8), four on every function after it — the contract reading
is memoised per file. Full budget: nine and eight (L2 three times, R1 twice).
C1's figure is exactly five, always. So `rounds_used` and `calls` are recorded
per function; without them the cost column of the comparison is not computable
after the fact, and a quality claim with no cost beside it is not a result.

The record this returns is deliberately key-compatible with
`natspec_corpus.runner.generate`, so `evaluate.py`, `report.py`, `stats.py` and
the emission stage read a ComGen run without modification and ComGen appears as
one more column beside C1. `gate`, `gate_kept_draft`, `gate_verdict` and
`support_rate` are populated with their ComGen equivalents for exactly that
reason; the commented lines below say which.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from natspec_corpus import prompts_v3 as _P
from natspec_corpus.emit import normalise
from natspec_corpus.errors import CorpusError
from natspec_corpus.llm import Client
from natspec_corpus.runner import strip_claims
from natspec_corpus.views import views_for

from . import critics as C
from . import intent as I
from . import prompts as P

#: Rounds of criticism-and-revision, counting the first critique as round 0.
#: 3 is the budget C1's single refiner pass is being compared against; it is a
#: config knob, not a constant of the architecture.
DEFAULT_ROUNDS = 3


def generate(ctx, client: Client, index=None, *,
             top_k: int = _P.RETRIEVAL_CONFIG["top_k"],
             tau: float = _P.RETRIEVAL_CONFIG["threshold_tau"],
             stages: Optional[List[str]] = None,
             rounds: int = DEFAULT_ROUNDS,
             contract: Optional[I.ContractIntent] = None) -> dict:
    """Run ComGen for one function and return its record.

    `stages` switches parts off for the ablations; see `prompts.STAGES`. "G" is
    required — it is the only stage that writes a comment. `contract` is a
    shared `ContractIntent` memo; passing one per run is what makes the
    contract reading cost one call per file rather than one per function.
    """
    stages = list(stages or P.ALL_STAGES)
    if "G" not in stages:
        raise CorpusError("G is the only stage that produces a comment")
    if rounds < 1:
        raise CorpusError(f"rounds must be at least 1, got {rounds}")

    rec: Dict[str, Any] = {
        "pair_id": ctx.pair["id"], "file": ctx.pair["file"],
        "function": ctx.pair["name"], "signature": ctx.pair["signature"],
        "split": ctx.pair.get("split"), "has_sigma": ctx.has_sigma,
        "architecture": "comgen", "stages": stages,
        "rounds_allowed": rounds, "started": time.time(),
    }
    sigma = ctx.sigma_text()

    # ---- retrieval ------------------------------------------------------
    exemplars, hits = "(retrieval disabled)", []
    if index is not None:
        hits = index.search(views_for(ctx.pair, ctx.table), top_k=top_k,
                            tau=tau)
        exemplars = P.exemplar_block(h.as_exemplar() for h in hits)
    rec["retrieved"] = [{"pair_id": h.unit.pair_id, "score": round(h.score, 4)}
                        for h in hits]

    # ---- intent ---------------------------------------------------------
    contract_reading = None
    if "IC" in stages:
        if contract is None:
            contract = I.ContractIntent(client)
        contract_reading, summary = contract.for_context(ctx)
        rec["contract_intent"] = contract_reading
        if summary:
            rec["call_L9"] = summary

    function_reading = None
    if "I7" in stages:
        c = client.call(P.FUNCTION_INTENT, source=ctx.source, sigma=sigma)
        function_reading = c.data
        rec["intent"] = c.data
        rec["call_L7"] = c.summary()

    intent_text = I.combined(function_reading, contract_reading)

    # ---- round 0: the draft ---------------------------------------------
    c = client.call(P.GENERATOR, source=ctx.source, sigma=sigma,
                    intent=intent_text, exemplars=exemplars)
    # Normalised on arrival: the model returns NatSpec *content*, often with no
    # comment markers or with `//`, which solc cannot see. Un-normalised, the
    # deterministic critic splices bare tag text into a file, gets a syntax
    # error, and blames the comment.
    candidate = normalise(c.data.get("natspec", ""))
    claims = c.data.get("claims", [])
    rec["draft"] = candidate
    rec["draft_claims"] = claims
    rec["call_L1b"] = c.summary()

    # ---- the loop -------------------------------------------------------
    attempts: List[Dict[str, Any]] = []
    for r in range(rounds):
        det = (C.deterministic(ctx, candidate) if "CD" in stages
               else C.not_run(C.DETERMINISTIC))
        sem = C.not_run(C.SEMANTIC)
        if "CS" in stages:
            sem, call = C.semantic(ctx, candidate, client, sigma=sigma)
            rec.setdefault("call_L2", []).append(call.summary())

        group = C.merge(det, sem)
        attempts.append({
            "round": r, "natspec": candidate, "claims": claims,
            "deterministic": det.to_dict(), "semantic": sem.to_dict(),
            "group": group.to_dict(),
            "blocking": group.blocking, "total": group.total,
        })

        if group.clean:
            break
        if r == rounds - 1 or "R" not in stages:
            break

        c = client.call(P.REVISER, source=ctx.source, sigma=sigma,
                        candidate=candidate, feedback=group.feedback())
        revised = normalise(c.data.get("natspec", ""))
        rec.setdefault("call_R1", []).append(c.summary())
        if not revised:
            # The reviser returned nothing usable. Stop rather than carry an
            # empty candidate into the next round, where every critic would
            # report the same defect again.
            attempts[-1]["reviser_returned_nothing"] = True
            break
        candidate = revised
        if c.data.get("claims"):
            claims = c.data["claims"]

    best = select(attempts)
    rec["rounds"] = [_thin(a) for a in attempts]
    rec["rounds_used"] = len(attempts)
    rec["best_round"] = best["round"]
    rec["final"] = best["natspec"]
    rec["final_claims"] = best["claims"]

    # Compatibility with the existing evaluation, which reads exactly these
    # keys. `gate.refined_defects` is what fills the defect-free column, so it
    # carries the WINNING round's deterministic findings — the same defect
    # labels `score.judge` produces for C1. `gate_kept_draft` keeps its
    # meaning: the first candidate was never improved on.
    det_best = best["deterministic"]
    rec["gate"] = {
        "passed": best["blocking"] == 0,
        "compiles": det_best.get("detail", {}).get("compiles"),
        "tags_emitted": det_best.get("detail", {}).get("tags_emitted"),
        "draft_defects": [f["text"] for f
                          in attempts[0]["deterministic"]["findings"]],
        "refined_defects": [f["text"] for f in det_best["findings"]],
        "reasons": [f["why"] or f["text"] for f in det_best["findings"]],
        "applicable": det_best.get("applicable", True),
    }
    rec["gate_kept_draft"] = best["round"] == 0

    # ---- the judge ------------------------------------------------------
    if "J" in stages:
        c = client.call(P.JUDGE, source=ctx.source, sigma=sigma,
                        candidate=rec["final"],
                        claims=json.dumps(rec["final_claims"], indent=1))
        rec["judgement"] = c.data
        rec["call_L8"] = c.summary()
        verdicts = c.data.get("verdicts", [])
        bad = [v for v in verdicts if v.get("verdict") == "CONTRADICTED"]
        rec["gate_verdict"] = c.data.get("gate", "PASS" if not bad else "FAIL")
        rec["support_rate"] = (
            sum(1 for v in verdicts if v.get("verdict") == "SUPPORTED")
            / len(verdicts)) if verdicts else None
        if bad:
            rec["final"] = strip_claims(rec["final"],
                                        [v["claim"] for v in bad])
            rec["stripped"] = [v["claim"] for v in bad]

    rec["calls"] = _count_calls(rec)
    rec["seconds"] = round(time.time() - rec.pop("started"), 3)
    return rec


def select(attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The winning round: fewest blocking defects, then fewest total.

    Ties go to the later round. Two candidates with the same defect count are
    not equally good bets — the later one was written in answer to specific
    criticism of the earlier one — and the calls that produced it are already
    spent either way. This is the v2 gate's `<=` rule, kept deliberately so
    the comparison against C1 is not confounded by a changed tie-break.
    """
    if not attempts:
        raise CorpusError("no rounds were run")
    return min(attempts,
               key=lambda a: (a["blocking"], a["total"], -a["round"]))


def _thin(attempt: Dict[str, Any]) -> Dict[str, Any]:
    """A round, minus the full candidate text on rounds that did not win.

    Every intermediate is worth keeping — attribution is the whole reason the
    records exist — but the semantic critic's raw defect list repeated across
    three rounds of 40 functions of 8 configurations is most of the file. The
    text of each candidate stays; the critics' full payloads stay only where
    they carry a finding.
    """
    out = dict(attempt)
    for key in ("deterministic", "semantic"):
        v = dict(out[key])
        if not v.get("findings"):
            v["detail"] = {k: x for k, x in (v.get("detail") or {}).items()
                           if k in ("compiles", "tags_emitted", "note")}
        out[key] = v
    return out


def _count_calls(rec: Dict[str, Any]) -> Dict[str, int]:
    """Model calls this function actually cost, by prompt.

    Variable cost is the price of the loop, so it is measured rather than
    assumed. A cached call is still counted here — it was issued; whether it
    was paid for is `cached` inside the call summary.
    """
    out: Dict[str, int] = {}
    for key, value in rec.items():
        if not key.startswith("call_"):
            continue
        pid = key[len("call_"):]
        out[pid] = len(value) if isinstance(value, list) else 1
    out["total"] = sum(out.values())
    return out
