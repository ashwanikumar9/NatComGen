"""Shared fixtures for the ComGen tests.

No model is reachable here and none is wanted. Every test in this package is
about the orchestration — the merge precedence, when the loop stops, which
round wins, what the record carries — and every one of those can be wrong while
the model is perfectly good. A scripted backend makes each of them a decidable
question instead of a judgement call about model output.
"""
from __future__ import annotations

from typing import Any, Dict, List

from natspec_corpus.llm import Client, MockBackend
from natspec_corpus.runner import Context

# A comment with nothing wrong with it: prose long enough to say something,
# both parameters documented, the return documented. `score.judge` reports no
# defect on this, which is what makes it the "clean" case.
CLEAN = ("/// @notice Adds two unsigned integers and returns the sum.\n"
         "/// @param a the first addend\n"
         "/// @param b the second addend\n"
         "/// @return the sum of the two addends")

# Four deterministic defects: the prose is too short, neither parameter is
# documented, the return is not documented.
DIRTY = "/// @notice Adds."

# Exactly one defect: the return is undocumented.
ONE_DEFECT_A = ("/// @notice Adds two unsigned integers and returns the sum.\n"
                "/// @param a the first addend\n"
                "/// @param b the second addend")

# Exactly one defect, a different one: parameter b is undocumented.
ONE_DEFECT_B = ("/// @notice Adds two unsigned integers and returns the sum.\n"
                "/// @param a the first addend\n"
                "/// @return the sum of the two addends")

# Two defects.
TWO_DEFECTS = ("/// @notice Adds two unsigned integers and returns the sum.\n"
               "/// @param a the first addend")


def pair() -> dict:
    code = ("function add(uint256 a, uint256 b) public pure "
            "returns (uint256) { return a + b; }")
    return {"id": "p1", "file": "x/C.sol", "name": "add", "kind": "function",
            "signature": "add(uint256,uint256)", "container": "C",
            "container_kind": "contract", "split": "val",
            "visibility": "public", "mutability": "pure",
            "code": code, "code_start": 0, "doc_spans": [],
            "notice": "Adds two unsigned integers.", "dev": "",
            "params": {"a": "the first addend", "b": "the second addend"},
            "returns": [{"name": None, "text": "the sum"}],
            "doc_raw": "/// @notice Adds two unsigned integers."}


def table() -> dict:
    return {"pair_id": "p1", "signature": "add(uint256,uint256)",
            "facts": [{"id": "F1", "kind": "returns",
                       "text": "returns uint256"}],
            "nodes": [], "edges": [], "paths": [], "calls": [], "deps": [],
            "reverts": [], "events": [], "interaction_order": None,
            "ast_types": {}}


def ctx(*, with_sigma: bool = True) -> Context:
    return Context(pair=pair(), table=table() if with_sigma else None,
                   unit=None, version=None)


def claims(text: str = "Adds two unsigned integers.") -> List[dict]:
    return [{"text": text, "ids": ["F1"]}]


def intent_reply() -> dict:
    return {"purpose": "Adds two unsigned integers.", "purpose_ids": ["F1"],
            "caller": None, "caller_ids": [], "preconditions": [],
            "effects": [], "returns_meaning": [], "unknowns": []}


def contract_reply() -> dict:
    return {"role": "Holds a total and exposes arithmetic on it.",
            "vocabulary": ["total"], "roles": [], "invariants": [],
            "do_not_claim": ["no fee logic", "no oracle"]}


def draft(natspec: str) -> dict:
    return {"natspec": natspec, "claims": claims(), "used_examples": False}


def revision(natspec: str, changed: bool = True) -> dict:
    return {"natspec": natspec, "changed": changed, "addressed": [],
            "declined": [], "claims": claims()}


def critique(*defects: dict) -> dict:
    return {"defects": list(defects), "missing_high_value_facts": []}


def defect(quote: str, verdict: str = "UNSUPPORTED",
           severity: str = "medium", why: str = "nothing states it") -> dict:
    return {"quote": quote, "verdict": verdict, "severity": severity,
            "why": why, "ids": []}


def judgement(verdict: str = "SUPPORTED", gate: str = "PASS") -> dict:
    return {"verdicts": [{"claim": "Adds two unsigned integers.",
                          "rows": ["F1"], "verdict": verdict}],
            "gate": gate}


class Script:
    """Scripted replies per prompt id.

    A list with more than one entry is consumed in order; a one-entry list
    repeats forever. So `Script(R1=[a, b])` gives the first revision `a` and
    every later one `b`, which is how a loop of unknown length is scripted
    without guessing how many rounds a test will take.
    """

    def __init__(self, **by_id: List[Any]) -> None:
        self.q: Dict[str, List[Any]] = {k: list(v) for k, v in by_id.items()}
        self.seen: List[str] = []

    def __call__(self, pid: str, request: dict) -> Any:
        self.seen.append(pid)
        if pid not in self.q:
            raise AssertionError(f"unscripted call to {pid}")
        q = self.q[pid]
        return q.pop(0) if len(q) > 1 else q[0]

    def count(self, pid: str) -> int:
        return self.seen.count(pid)


def client(script: Script, **kw) -> Client:
    return Client(MockBackend(script), **kw)


def default_script(*, drafts=(CLEAN,), revisions=(CLEAN,),
                   critiques=None, gate: str = "PASS") -> Script:
    return Script(
        L9=[contract_reply()],
        L7=[intent_reply()],
        L1b=[draft(d) for d in drafts],
        L2=list(critiques) if critiques else [critique()],
        R1=[revision(r) for r in revisions],
        L8=[judgement(gate=gate)],
    )
