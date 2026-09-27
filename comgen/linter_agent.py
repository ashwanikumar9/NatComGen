"""The Linter Agent: the last check, on the file rather than the comment.

Every check before this one looks at a comment. This one looks at the file the
comments were written into, which catches a different class of failure — one
that cannot be seen from any single comment and that the per-function critics
are structurally blind to:

  * the code changed. Placing a comment must not touch a byte of code. An
    off-by-one in an offset shows up here and nowhere else.
  * the file stopped compiling. Each comment compiled on its own, spliced into
    the file alone; twelve of them together can still break it.
  * a comment landed on the wrong declaration. Compiling and comparing solc's
    devdoc against the declarations is the only check that catches this, and it
    is the failure a reader would most trust and be most misled by.
  * a comment was dropped in placement. The comment is fine, the emitted file
    simply does not carry it.

All four come from `natspec_corpus.verify_file.verify`, which is why this
module is thin. It is a separate agent in the architecture rather than a step
inside the Aggregator because its verdict is about the deliverable: the
Aggregator's job is to produce a documented file, and this one's job is to
refuse to hand it over.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from natspec_corpus.verify_file import verify

from .aggregator import Assembled


@dataclass
class LintResult:
    rel: str
    ok: bool
    report: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"rel": self.rel, "ok": self.ok, "issues": self.issues,
                **self.report}


def lint(a: Assembled) -> LintResult:
    """Check one documented file against the original it came from."""
    report = verify(a.rel, a.original, a.emitted, a.unit_sources,
                    a.version).to_dict()
    issues: List[str] = []
    if not report.get("code_unchanged"):
        issues.append("code changed: placement altered a byte outside a "
                      "comment")
    if not report.get("compiles"):
        issues.append("the documented file does not compile")
    if not report.get("placement_ok"):
        issues.append("a comment is attached to the wrong declaration, or was "
                      "dropped in placement")
    exposed, documented = report.get("exposed", 0), report.get("documented", 0)
    if exposed and documented < a.placed:
        issues.append(f"{a.placed} comments placed but solc reports "
                      f"{documented} documented members")
    return LintResult(rel=a.rel, ok=not issues, report=report, issues=issues)


def summarise(results: Iterable[LintResult]) -> dict:
    """What to print, and what the run record keeps.

    `accepted` is the number that matters and it is deliberately the strict
    one: a file with any issue is not accepted. An aggregate that counted
    partial successes would make a placement bug look like a rounding error.
    """
    rows = list(results)
    by_issue: Dict[str, int] = {}
    for r in rows:
        for i in r.issues:
            by_issue[i.split(":")[0]] = by_issue.get(i.split(":")[0], 0) + 1
    return {"files": len(rows),
            "accepted": sum(1 for r in rows if r.ok),
            "rejected": sum(1 for r in rows if not r.ok),
            "issues": by_issue}
