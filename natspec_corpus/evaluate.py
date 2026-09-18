"""Scoring a run against the gold references.

Scored **per field** — notice, dev, each param, each return — not as one blob
over the whole comment. A model that writes a fluent notice and invents the
parameter descriptions scores well on a blended similarity and is useless; the
per-field split is what makes that visible.

BLEU-4 and ROUGE-L are implemented here rather than pulled in, so the numbers
do not depend on which version of which NLP package happened to be installed,
and so this runs anywhere. BERTScore is optional and plugs in through
`semantic_fn`, because it needs a model download.

The two metrics that are actually the contribution are not similarity scores:
the share of claims the verifier rated SUPPORTED, and the share of comments
solc compiles and emits. Neither has anything to do with n-gram overlap with a
human reference, and neither can be reported by a system that does not carry
evidence ids.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .natspec import parse as parse_doc

_WORD = re.compile(r"[A-Za-z0-9_]+")


def tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def bleu(candidate: str, reference: str, n: int = 4) -> float:
    """Sentence BLEU-4 with add-one smoothing on higher orders.

    Smoothing matters here: NatSpec fields are short, a one-sentence @notice
    often shares no 4-gram with the reference, and unsmoothed BLEU would
    report a hard zero for a perfectly good paraphrase.
    """
    c, r = tokens(candidate), tokens(reference)
    if not c or not r:
        return 0.0
    logs = []
    for k in range(1, n + 1):
        cc = Counter(tuple(c[i:i + k]) for i in range(len(c) - k + 1))
        rc = Counter(tuple(r[i:i + k]) for i in range(len(r) - k + 1))
        total = sum(cc.values())
        if total == 0:
            # The candidate is shorter than k tokens, so this order has no
            # n-grams at all. Orders above the sentence length are skipped
            # rather than scored zero: a three-word @notice identical to its
            # reference must score 1.0, not 0.0.
            break
        overlap = sum(min(v, rc[g]) for g, v in cc.items())
        if k == 1:
            if overlap == 0:
                return 0.0
            logs.append(math.log(overlap / total))
        else:
            logs.append(math.log((overlap + 1) / (total + 1)))
    if not logs:
        return 0.0
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / max(len(c), 1))
    return bp * math.exp(sum(logs) / len(logs))


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str, beta: float = 1.2) -> float:
    c, r = tokens(candidate), tokens(reference)
    if not c or not r:
        return 0.0
    l = _lcs(c, r)
    if l == 0:
        return 0.0
    p, rec = l / len(c), l / len(r)
    return ((1 + beta ** 2) * p * rec) / (rec + beta ** 2 * p)


def exact(candidate: str, reference: str) -> float:
    return float(tokens(candidate) == tokens(reference))


# --------------------------------------------------------------------------
# field extraction
# --------------------------------------------------------------------------

def fields(natspec: str) -> Dict[str, str]:
    """Flatten a comment into comparable fields: notice, dev, param:<name>,
    return:<i>."""
    doc = parse_doc(natspec or "")
    out = {"notice": doc.notice, "dev": doc.dev}
    for name, text in doc.params.items():
        out[f"param:{name}"] = text
    for i, t in enumerate(doc.returns):
        # `@return depositID The ID` parses to arg="depositID",
        # text="depositID The ID" — the name is still at the head of the text.
        # Joining both duplicates it and costs a correct answer ~0.13 BLEU.
        text = t.text or ""
        if t.arg and text.startswith(t.arg):
            out[f"return:{i}"] = text
        else:
            out[f"return:{i}"] = " ".join(x for x in (t.arg or "", text) if x)
    return {k: v for k, v in out.items() if v and v.strip()}


def reference_fields(pair: dict) -> Dict[str, str]:
    out = {"notice": pair.get("notice", ""), "dev": pair.get("dev", "")}
    for name, text in (pair.get("params") or {}).items():
        out[f"param:{name}"] = text
    for i, r in enumerate(pair.get("returns") or []):
        out[f"return:{i}"] = " ".join(
            x for x in (r.get("name") or "", r.get("text") or "") if x)
    return {k: v for k, v in out.items() if v and v.strip()}


# --------------------------------------------------------------------------
# scoring a run
# --------------------------------------------------------------------------

def score_record(rec: dict, pair: dict,
                 semantic_fn: Optional[Callable[[str, str], float]] = None
                 ) -> dict:
    ref = reference_fields(pair)
    got = fields(rec.get("final", ""))
    per_field: Dict[str, dict] = {}
    for key, gold in ref.items():
        cand = got.get(key, "")
        row = {"bleu": bleu(cand, gold), "rouge_l": rouge_l(cand, gold),
               "exact": exact(cand, gold), "present": bool(cand.strip())}
        if semantic_fn is not None and cand.strip():
            row["semantic"] = semantic_fn(cand, gold)
        per_field[key] = row

    kinds = {"notice": [], "dev": [], "param": [], "return": []}
    for key, row in per_field.items():
        kinds[key.split(":", 1)[0]].append(row)

    def mean(rows: List[dict], k: str) -> Optional[float]:
        vals = [r[k] for r in rows if k in r]
        return sum(vals) / len(vals) if vals else None

    return {
        "pair_id": rec.get("pair_id"), "has_sigma": rec.get("has_sigma"),
        "fields": per_field,
        "by_kind": {k: {"n": len(v), "bleu": mean(v, "bleu"),
                        "rouge_l": mean(v, "rouge_l"),
                        "coverage": mean(v, "present")}
                    for k, v in kinds.items() if v},
        "extra_fields": sorted(set(got) - set(ref)),
        "support_rate": rec.get("support_rate"),
        "gate_verdict": rec.get("gate_verdict"),
        "gate_kept_draft": rec.get("gate_kept_draft"),
        "defects": rec.get("gate", {}).get("refined_defects"),
    }


def aggregate(scores: List[dict]) -> dict:
    """Means, with the no-fact-table functions reported separately.

    They are a different experimental condition — source only, no structural
    evidence — and folding them into one average would understate the system
    and overstate the baseline at the same time.
    """
    def block(rows: List[dict]) -> dict:
        if not rows:
            return {"n": 0}
        out: Dict[str, object] = {"n": len(rows)}
        for kind in ("notice", "dev", "param", "return"):
            b = [r["by_kind"][kind] for r in rows if kind in r["by_kind"]]
            if b:
                out[kind] = {
                    "n": sum(x["n"] for x in b),
                    "bleu": sum(x["bleu"] for x in b) / len(b),
                    "rouge_l": sum(x["rouge_l"] for x in b) / len(b),
                    "coverage": sum(x["coverage"] for x in b) / len(b)}
        sup = [r["support_rate"] for r in rows if r["support_rate"] is not None]
        if sup:
            out["claim_support_rate"] = sum(sup) / len(sup)
        gv = [r["gate_verdict"] for r in rows if r["gate_verdict"]]
        if gv:
            out["verifier_pass_rate"] = sum(1 for g in gv if g == "PASS") / len(gv)
        kd = [r["gate_kept_draft"] for r in rows
              if r["gate_kept_draft"] is not None]
        if kd:
            out["gate_kept_draft_rate"] = sum(kd) / len(kd)
        clean = [r for r in rows if r["defects"] is not None]
        if clean:
            out["defect_free_rate"] = sum(
                1 for r in clean if not r["defects"]) / len(clean)
        return out

    return {"all": block(scores),
            "with_sigma": block([s for s in scores if s["has_sigma"]]),
            "without_sigma": block([s for s in scores if not s["has_sigma"]])}


def evaluate_run(corpus_root: Path, run_path: Path,
                 semantic_fn: Optional[Callable[[str, str], float]] = None
                 ) -> dict:
    pairs = {}
    for line in (corpus_root / "pairs.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            pairs[p["id"]] = p

    scores, errors = [], 0
    for line in run_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error"):
            errors += 1
            continue
        pair = pairs.get(rec["pair_id"])
        if pair:
            scores.append(score_record(rec, pair, semantic_fn))
    out = aggregate(scores)
    out["errors"] = errors
    out["scored"] = len(scores)
    return out
