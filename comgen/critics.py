"""The Critic Group: two critics, one box, one merged verdict.

Both critics read the generator's draft directly — neither is downstream of the
other, and neither sees the other's findings. That is the architectural point:
in v2 the deterministic check sat *after* the refiner and could only veto it,
so a structural defect cost a whole extra round to discover. Here it is read at
the same moment as the semantic critique and travels back in the same feedback.

They are not equals when they disagree. The deterministic critic attaches the
comment to the real file, compiles it, and compares what solc emitted against
the declaration; its findings are facts about the compiler's behaviour. The
semantic critic is a model reading evidence. So the merge is not a union:

  1. Every deterministic finding is MUST FIX and survives the merge intact.
  2. A semantic finding that would undo a deterministic one is DROPPED, with
     the reason recorded. "Delete the @param, the evidence doesn't describe
     that argument" cannot be allowed to answer "solc dropped this @param" —
     following it produces a comment that satisfies the critic and documents
     less than the declaration requires. This is the only place the two can
     actually contradict each other and it is resolved here, in code, rather
     than left to the generator's judgement in a prompt.
  3. What remains is ordered CONTRADICTED before UNSUPPORTED, high before
     medium before low, so a round that only partly succeeds fixes the worst
     item first.

The deterministic critic costs nothing — no model call — which is why it is
also the loop's exit guard in `orchestrator.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from natspec_corpus.gate import judge_text, solc_emits
from natspec_corpus.llm import Client

from . import prompts as P

DETERMINISTIC = "deterministic"
SEMANTIC = "semantic"

#: Severity ranking used to order the merged feedback. Deterministic findings
#: are all `blocking`; the semantic critic's own severities follow.
_RANK = {"blocking": 0, "high": 1, "medium": 2, "low": 3}

#: A deterministic defect label carries its subject after a colon:
#: `param_missing:amount`. These are the labels whose subject is a tag the
#: comment is *required* to carry, and therefore the ones a semantic "delete
#: it" finding must not be allowed to undo.
_REQUIRING = ("param_missing", "param_empty", "return_missing",
              "return_empty")


@dataclass
class Finding:
    """One thing wrong with the candidate, from one critic."""
    source: str
    text: str
    severity: str
    why: str = ""
    ids: List[str] = field(default_factory=list)
    subject: Optional[str] = None      # the tag or param a finding is about

    @property
    def blocking(self) -> bool:
        return self.severity == "blocking"

    def to_dict(self) -> dict:
        d = {"source": self.source, "text": self.text,
             "severity": self.severity}
        if self.why:
            d["why"] = self.why
        if self.ids:
            d["ids"] = list(self.ids)
        if self.subject:
            d["subject"] = self.subject
        return d


@dataclass
class CriticVerdict:
    """What one critic concluded about one candidate."""
    name: str
    ran: bool
    findings: List[Finding] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)
    #: False when the check could not apply — no compilation unit for the
    #: deterministic critic, the stage switched off for either. Never
    #: silently reported as a pass.
    applicable: bool = True

    @property
    def ok(self) -> bool:
        """Clean. A critic that did not run is not clean; it is unknown."""
        return self.ran and not self.findings

    def to_dict(self) -> dict:
        return {"name": self.name, "ran": self.ran,
                "applicable": self.applicable,
                "ok": self.ok,
                "findings": [f.to_dict() for f in self.findings],
                "detail": self.detail}


# --------------------------------------------------------------------------
# the deterministic critic — no model, no cost
# --------------------------------------------------------------------------

def deterministic(ctx, candidate: str) -> CriticVerdict:
    """Compile the candidate into the real file and score its tags.

    Two checks, both borrowed whole from `natspec_corpus.gate`, so ComGen's
    notion of a structural defect is the same one the corpus was labelled with:

      * `solc_emits` — the file still compiles with this comment attached, and
        solc's devdoc/userdoc still carries the tags the comment claims. A
        malformed @param is not a compile error; it is a tag the compiler
        quietly drops, which is worse.
      * `judge_text` — the same `score.judge` that labels the corpus, run on
        the candidate against the real declaration.

    Without a compilation unit the first check cannot run. It is reported as
    not applicable — never as a pass.
    """
    v = CriticVerdict(name=DETERMINISTIC, ran=True)

    for label in judge_text(ctx.pair, candidate):
        kind, _, subject = label.partition(":")
        v.findings.append(Finding(
            source=DETERMINISTIC, text=label, severity="blocking",
            why=_explain(kind, subject), subject=subject or None))

    if ctx.unit is None:
        v.applicable = False
        v.detail = {"compiles": None, "tags_emitted": None,
                    "note": "no compilation unit for this file; the compiler "
                            "half of this critic could not run"}
        return v

    compiles, tags, detail = solc_emits(
        ctx.unit, ctx.pair["file"], ctx.pair, candidate, ctx.version)
    v.detail = {"compiles": compiles, "tags_emitted": tags, **detail}
    if not compiles:
        v.findings.append(Finding(
            source=DETERMINISTIC, text="does_not_compile", severity="blocking",
            why="solc rejects the file with this comment attached: "
                f"{detail.get('errors')}"))
    elif not tags:
        dropped = detail.get("dropped_params") or []
        for name in dropped:
            v.findings.append(Finding(
                source=DETERMINISTIC, text=f"tag_dropped:{name}",
                severity="blocking", subject=name,
                why=f"solc emitted no devdoc entry for @param {name}: the "
                    f"name does not match the declaration, or the tag is "
                    f"malformed"))
        if not dropped:
            v.findings.append(Finding(
                source=DETERMINISTIC, text="notice_dropped",
                severity="blocking",
                why="the comment claims a @notice that solc did not emit to "
                    "userdoc"))
    return v


def _explain(kind: str, subject: str) -> str:
    """Plain wording for a defect label, for the generator to act on."""
    return {
        "no_text": "the comment has no prose at all: no @notice and no @dev",
        "text_short": "the prose is too short to say anything",
        "name_echo": "the prose only repeats the function's own name",
        "placeholder": "the comment contains a TODO/TBD placeholder",
        "param_missing": f"@param {subject} is declared but not documented",
        "param_empty": f"@param {subject} is present but has no description",
        "param_unknown": f"@param {subject} documents an argument the "
                         f"declaration does not have",
        "return_missing": "a declared return value is not documented",
        "return_extra": "more @return tags than the declaration returns",
        "return_empty": "a @return tag has no description",
    }.get(kind, kind)


# --------------------------------------------------------------------------
# the semantic critic — one model call
# --------------------------------------------------------------------------

def semantic(ctx, candidate: str, client: Client,
             sigma: Optional[str] = None) -> Tuple[CriticVerdict, Any]:
    """Audit the candidate's claims against Σ(f). Returns (verdict, call)."""
    v = CriticVerdict(name=SEMANTIC, ran=True)
    call = client.call(P.SEMANTIC_CRITIC, source=ctx.source,
                       sigma=sigma if sigma is not None else ctx.sigma_text(),
                       candidate=candidate)
    data = call.data or {}
    for d in data.get("defects", []):
        quote = (d.get("quote") or "").strip()
        v.findings.append(Finding(
            source=SEMANTIC,
            text=f"{d.get('verdict', 'UNSUPPORTED')}: {quote}",
            severity=d.get("severity", "medium"),
            why=d.get("why", ""), ids=list(d.get("ids") or []),
            subject=_tag_subject(quote)))
    v.detail = {
        "defects": data.get("defects", []),
        "missing_high_value_facts": data.get("missing_high_value_facts", []),
    }
    return v, call


_TAG = re.compile(r"@(param|return)\s+(\w+)?")


def _tag_subject(quote: str) -> Optional[str]:
    """The tag a semantic finding is about, if it quotes one."""
    m = _TAG.search(quote or "")
    if not m:
        return None
    return m.group(2) or m.group(1)


def not_run(name: str) -> CriticVerdict:
    """A critic switched off by an ablation. Unknown, not clean."""
    return CriticVerdict(name=name, ran=False, applicable=False)


# --------------------------------------------------------------------------
# the merge
# --------------------------------------------------------------------------

@dataclass
class GroupVerdict:
    """What the Critic Group, as one unit, says about one candidate."""
    must_fix: List[Finding] = field(default_factory=list)
    should_fix: List[Finding] = field(default_factory=list)
    omissions: List[dict] = field(default_factory=list)
    suppressed: List[dict] = field(default_factory=list)
    deterministic_applicable: bool = True
    semantic_ran: bool = True

    @property
    def findings(self) -> List[Finding]:
        return self.must_fix + self.should_fix

    @property
    def clean(self) -> bool:
        """Nothing left to fix. The loop's exit condition.

        Omissions do not block: they are suggestions, and a comment that is
        merely less complete than it could be is not defective. Requiring them
        would make the loop run its full budget on almost every function.
        """
        return not self.must_fix and not self.should_fix

    @property
    def blocking(self) -> int:
        return len(self.must_fix)

    @property
    def total(self) -> int:
        return len(self.must_fix) + len(self.should_fix)

    def to_dict(self) -> dict:
        return {"must_fix": [f.to_dict() for f in self.must_fix],
                "should_fix": [f.to_dict() for f in self.should_fix],
                "omissions": self.omissions,
                "suppressed": self.suppressed,
                "blocking": self.blocking, "total": self.total,
                "clean": self.clean,
                "deterministic_applicable": self.deterministic_applicable,
                "semantic_ran": self.semantic_ran}

    def feedback(self) -> str:
        """The block handed to the generator. Precedence is on the page."""
        out: List[str] = []
        if self.must_fix:
            out.append("MUST FIX — from the deterministic critic. These were "
                       "produced by compiling your comment into the real "
                       "file. They are not opinions.")
            for i, f in enumerate(self.must_fix, 1):
                out.append(f"  {i}. [{f.text}] {f.why}")
        if self.should_fix:
            out.append("")
            out.append("SHOULD FIX — from the semantic critic. Claims the "
                       "evidence does not support. Act on each unless you can "
                       "name a row that supports it.")
            for i, f in enumerate(self.should_fix, 1):
                ids = f" (rows: {', '.join(f.ids)})" if f.ids else ""
                out.append(f"  {i}. [{f.severity}] {f.text}{ids}")
                if f.why:
                    out.append(f"       why: {f.why}")
        if self.omissions:
            out.append("")
            out.append("HIGH-VALUE OMISSIONS — optional. Add one only if it is "
                       "a revert condition, an ordering hazard or a "
                       "non-obvious effect.")
            for i, o in enumerate(self.omissions, 1):
                ids = f" (rows: {', '.join(o.get('ids') or [])})"
                out.append(f"  {i}. {o.get('text', '')}{ids}")
        if not out:
            out.append("No defects found. Return the comment unchanged.")
        return "\n".join(out)


def merge(det: CriticVerdict, sem: CriticVerdict) -> GroupVerdict:
    """One verdict from both critics. Deterministic findings are final."""
    g = GroupVerdict(deterministic_applicable=det.applicable,
                     semantic_ran=sem.ran)
    g.must_fix = list(det.findings)

    required = {f.subject for f in det.findings
                if f.subject and f.text.split(":", 1)[0] in _REQUIRING}
    required |= {f.subject for f in det.findings
                 if f.subject and f.text.startswith("tag_dropped")}

    keep: List[Finding] = []
    for f in sem.findings:
        if f.subject and f.subject in required:
            g.suppressed.append({
                **f.to_dict(),
                "dropped_because": f"the deterministic critic requires "
                                   f"@param/@return {f.subject}; acting on "
                                   f"this would remove a tag the declaration "
                                   f"needs"})
            continue
        keep.append(f)

    keep.sort(key=lambda f: (0 if f.text.startswith("CONTRADICTED") else 1,
                             _RANK.get(f.severity, 9)))
    g.should_fix = keep
    g.omissions = list(sem.detail.get("missing_high_value_facts") or [])
    return g
