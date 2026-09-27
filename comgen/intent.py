"""Contract-level intent: one call per file, reused by every function in it.

Why the architecture grew this. Reading the C1 records, the largest single
class of invented claim is not a wrong fact about the function — it is a right
fact about the function stated in the vocabulary of machinery that is not
there. A function that writes one mapping gets described as "updating the
vault's share accounting", because a 7B model handed one function and told to
say what it is for will supply the surrounding system from its training data.
Σ(f) cannot fix this: the fact table is scoped to the function, so it has
nothing to say about what the contract is or is not.

So the contract is read once, from its own source, and the answer travels into
every function's generator prompt — including `do_not_claim`, which is the
half that actually does the work. Naming the machinery the file does NOT
contain is what lets the generator decline to invent it.

Cost. One call per FILE, memoised here and cached on disk like any other call,
so a 20-function contract pays 1/20th of a call per function. That is why this
sits in the architecture at all: the same correction applied per function would
have cost as much as the generator.

Comments are stripped before the model sees the file. A contract's own
documentation would otherwise leak the reference text straight into the
generator's prompt, which would make every number in the evaluation worthless.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from natspec_corpus.llm import Client
from natspec_corpus.verify_file import strip_comments

from . import prompts as P

#: A whole file at 8192 num_ctx is a real risk of silent truncation by the
#: server, which is worse than a truncation we declare ourselves.
MAX_CHARS = 12000


def file_source(ctx) -> Tuple[str, bool]:
    """The file this function lives in, comments stripped. (text, truncated)

    Falls back to the function's own source when there is no compilation unit,
    in which case the contract reading is thinner but not wrong — it is still
    only ever asked to report what it can see.
    """
    rel = ctx.pair["file"]
    src = None
    if ctx.unit is not None:
        src = ctx.unit.sources.get(rel)
    if src is None:
        src = ctx.pair.get("code", "")
    try:
        src = strip_comments(src)
    except Exception:                                    # noqa: BLE001
        pass
    if len(src) > MAX_CHARS:
        return (src[:MAX_CHARS]
                + "\n// [truncated: this file is longer than the window; the "
                  "tail was not shown]", True)
    return src, False


class ContractIntent:
    """Memoised contract readings for one run.

    Keyed by file path. The on-disk call cache already makes a repeat free,
    but a memo here also avoids re-rendering and re-hashing a 12KB request
    once per function, which on a 40-function file is not nothing.
    """

    def __init__(self, client: Client, *, enabled: bool = True) -> None:
        self.client = client
        self.enabled = enabled
        self._by_file: Dict[str, Optional[dict]] = {}
        self._calls: Dict[str, dict] = {}
        self.hits = 0
        self.misses = 0

    def for_context(self, ctx) -> Tuple[Optional[dict], Optional[dict]]:
        """(reading, call summary). Both None when switched off."""
        if not self.enabled:
            return None, None
        rel = ctx.pair["file"]
        if rel in self._by_file:
            self.hits += 1
            return self._by_file[rel], None
        self.misses += 1
        src, truncated = file_source(ctx)
        call = self.client.call(P.CONTRACT_INTENT, rel=rel, source=src)
        data = dict(call.data or {})
        if truncated:
            data["_truncated"] = True
        self._by_file[rel] = data
        self._calls[rel] = call.summary()
        return data, call.summary()

    def stats(self) -> dict:
        return {"files": len(self._by_file), "reused": self.hits,
                "calls": self.misses}


def as_text(data: Optional[dict]) -> str:
    """Render a contract reading for a prompt, or say plainly that there is
    none. An empty string here would leave the generator with a bare header
    and no way to tell 'nothing found' from 'not asked'."""
    if not data:
        return "(no contract reading was produced)"
    out = [f"role: {data.get('role') or 'unknown'}"]
    if data.get("vocabulary"):
        out.append("this contract's own terms: "
                   + ", ".join(str(v) for v in data["vocabulary"][:20]))
    for r in (data.get("roles") or [])[:8]:
        out.append(f"privileged caller: {r.get('name')} ({r.get('evidence')})")
    for inv in (data.get("invariants") or [])[:8]:
        out.append(f"invariant: {inv}")
    for nc in (data.get("do_not_claim") or [])[:12]:
        out.append(f"DO NOT CLAIM: {nc}")
    if data.get("_truncated"):
        out.append("(the file was truncated before this reading; absence of a "
                   "mechanism here is weaker evidence than usual)")
    return "\n".join(out)


def combined(function_intent: Optional[dict],
             contract: Optional[dict]) -> str:
    """The `intent` input to the generator: the contract first, then the
    function. Contract first on purpose — it is the frame the function is read
    in, and a model reads the top of a block more reliably than the middle."""
    parts = ["[Contract]", as_text(contract), "",
             "[Function]",
             json.dumps(function_intent, indent=1) if function_intent
             else "(no function intent reading was produced)"]
    return "\n".join(parts)
