"""Stage 2 — generation. Five LLM calls per function, and the prompts for them.

This supersedes the twelve-prompt v2 set. The cuts came from SAGE's own
ablation (Table 15): structural context is worth +0.112 F1, example retrieval
+0.010. Anything that spent a call on redundant framing rather than on evidence
was switched off — the second generator lineage, the merger, the two judges,
the adjudicator and the second critic/refiner passes. 13.9 calls per function
became 5.0, and a full evaluation pass went from 3.7 hours to 1.3.

    L7  IntentReasoner     0.10   what the function is for, as citations
    L1b Generator          0.20   the draft
    L2  SemanticCritic     0.10   claims that the evidence does not support
    L3  Refiner            0.20   the rewrite
    L8  ClaimVerifier      0.00   final gate, claim by claim

Between L3 and L8 sits a deterministic gate that costs nothing: the refined
comment must still compile under solc, and its defect score must not be worse
than the draft's. A refinement that makes things worse is discarded rather
than judged by another model.

Every prompt reads one Σ(f) fact table, serialised by `sigma_block()` below,
and every factual claim must cite a row id from it. The ids are real: they come
out of `sigma.py` and they are stable across runs, which is what makes a
citation checkable rather than decorative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Model slots. Bound at run time; named here so a change is one edit.
INTENT_MODEL = "INTENT_REASONER_MODEL"
GENERATOR_MODEL = "GENERATOR_MODEL"
CRITIC_MODEL = "SEMANTIC_CRITIC_MODEL"
VERIFIER_MODEL = "VERIFIER_MODEL"

RETRIEVAL_CONFIG = {
    "views": ("code", "ast", "cfg"),
    "weights": {"code": 1 / 3, "ast": 1 / 3, "cfg": 1 / 3},
    "normalise": "z-score per modality over the union of per-index top-K",
    "absent_modality_score": 0.0,
    "top_k": 5,
    "threshold_tau": 0.5,
    "tuned": False,
    "unit": "one retrieval unit is a train-split pair, never a whole file",
}


@dataclass(frozen=True)
class Prompt:
    id: str
    agent: str
    model: str
    temp: float
    system: str
    user: str
    schema: Dict[str, Any]
    inputs: tuple
    max_tokens: int = 1024
    notes: str = ""


def render(p: Prompt, **kwargs) -> Dict[str, Any]:
    missing = set(p.inputs) - set(kwargs)
    if missing:
        raise KeyError(f"{p.id}: missing inputs {sorted(missing)}")
    return {
        "model": p.model,
        "options": {"temperature": p.temp, "num_predict": p.max_tokens},
        "format": p.schema,
        "messages": [{"role": "system", "content": p.system},
                     {"role": "user", "content": p.user.format(**kwargs)}],
    }


# =============================================================================
# Σ(f) serialisation — the evidence every call reads
# =============================================================================

def sigma_block(table: Dict[str, Any], *, max_nodes: int = 40,
                max_paths: int = 6) -> str:
    """Render one fact table from `sigma.py` as citable evidence lines.

    Node and path lists are capped because a 165-node function does not fit in
    a 7B context window and the tail is rarely what a comment is about. The cap
    is announced in the block rather than applied silently, so the model knows
    it is reading an excerpt and does not claim the function has no other
    branches.
    """
    out: List[str] = []

    def section(title: str, rows: List[str]) -> None:
        if rows:
            out.append(f"[{title}]")
            out.extend("  " + r for r in rows)
            out.append("")

    section("FACTS", [f"{f['id']:<4} {f['kind']:<18} {f['text']}"
                      for f in table.get("facts", [])])

    section("REVERTS", [
        f"{r['id']:<4} {r['kind']:<8} {r['condition']}"
        + (f"   message: {r['message']!r}" if r.get("message") else "")
        + f"   at {r['node']}"
        for r in table.get("reverts", [])])

    section("EVENTS", [
        f"{e['id']:<4} {e['event']}({', '.join(e['args'])})   at {e['node']}"
        for e in table.get("events", [])])

    nodes = table.get("nodes", [])
    node_rows = []
    for n in nodes[:max_nodes]:
        bits = [f"{n['id']:<4} {n['type']:<12}"]
        if n["text"]:
            bits.append(n["text"][:96])
        extra = []
        if n["state_reads"]:
            extra.append("reads state: " + ", ".join(n["state_reads"]))
        if n["state_writes"]:
            extra.append("writes state: " + ", ".join(n["state_writes"]))
        if extra:
            bits.append("   [" + "; ".join(extra) + "]")
        node_rows.append(" ".join(bits))
    if len(nodes) > max_nodes:
        node_rows.append(f".... {len(nodes) - max_nodes} further nodes omitted")

    paths = table.get("paths", [])
    for p in paths[:max_paths]:
        node_rows.append(f"{p['id']:<4} " + " -> ".join(p["nodes"]))
    if paths and paths[0].get("enumeration_truncated"):
        node_rows.append(".... path enumeration was capped; more paths exist")
    section("CFG", node_rows)

    section("CALLS", [f"{c['id']:<4} {c['kind']:<10} {c['target']}"
                      f"   at {c['node']}" for c in table.get("calls", [])])

    section("DEPS", [f"{d['id']:<4} {d['variable']} <- "
                     f"{', '.join(d['depends_on'])}"
                     for d in table.get("deps", [])])

    return "\n".join(out).rstrip() or "[FACTS]\n  (no facts extracted)"


def exemplar_block(hits: Iterable[Dict[str, Any]]) -> str:
    rows = []
    for i, h in enumerate(hits, 1):
        s = h.get("scores", {})
        rows.append(
            f"--- example {i} | code {s.get('code', 0):.2f} "
            f"ast {s.get('ast', 0):.2f} cfg {s.get('cfg', 0):.2f} ---\n"
            f"{h['code']}\n{h['natspec']}")
    return "\n\n".join(rows) if rows else "(no example passed the threshold)"


# =============================================================================
# Shared instruction blocks
# =============================================================================

EVIDENCE_LEGEND = """\
EVIDENCE FORMAT. Every row below has an id. Cite ids, never line numbers.

  [FACTS]    F*  declaration properties: visibility, mutability, each
                 parameter and return, modifiers, which storage variables are
                 read and written, and — when the function makes an external
                 call — whether any state write happens after it.
  [REVERTS]  R*  the guards. `R1 require msg.sender == owner  message: 'not
                 owner'  at N2` means the function reverts with that message
                 unless that condition holds.
  [EVENTS]   E*  events emitted, with their arguments.
  [CFG]      N*  control-flow nodes, with the expression each one evaluates
                 and the storage it touches.
             P*  execution paths, e.g. `P1  N1 -> N2 -> N5`. A path is one
                 way through the function from entry to exit.
  [CALLS]    C*  call sites. `external` and `low_level` leave this contract;
                 `library`, `internal` and `builtin` do not.
  [DEPS]     D*  data dependencies. `D1 newTotal <- amount, total` reads as:
                 the value of newTotal derives from amount and total.

Some sections may be absent. An absent section means the analysis found
nothing of that kind — no reverts means the function has no guards, not that
the guards are unknown."""

NATSPEC_RULES = """\
NATSPEC RULES (solc-enforced; a violation is a compile error or a dropped tag):
  - Untagged prose in a /// or /** block is treated as @notice. Never emit a
    bare @notice tag with nothing after it.
  - Exactly one @param per NAMED parameter, using the parameter's exact name,
    in declaration order. Unnamed parameters get no @param.
  - Exactly one @return per returned value, in declaration order. If the
    returns are named, begin each @return with that name.
  - @inheritdoc <ContractName> is valid only when the function overrides a
    base declaration. Never emit it as a placeholder.
  - @title and @author belong to a contract, interface or library, never to a
    function.
  - @notice says what the function does, in one sentence, for someone who will
    call it. @dev carries implementation detail: invariants, ordering, units,
    caller obligations."""

WRITING_RULES = """\
WRITING RULES:
  - Say what the function does and why it exists. Restating the signature adds
    nothing: "sets the owner to _owner" is not documentation.
  - Summarise the effect. Do not narrate the control flow node by node.
  - Every factual claim must cite the evidence id that supports it.
  - A NAME IS NOT A FACT. A modifier called onlyOwner does not establish that
    only the owner may call the function — only an R* row or a CFG guard does.
    The same applies to parameter names implying units, function names implying
    intent, and variable names implying ownership.
  - If something is absent from the evidence, you do not know it. Do not supply
    units, denominations, oracle sources, caller identity or economic purpose
    from general knowledge about Solidity or DeFi.
  - No hedging ("might", "probably"), no marketing language, no TODOs."""

EXEMPLAR_WARNING = """\
TREAT THE EXAMPLES WITH SUSPICION. Retrieval is wrong roughly a third of the
time, and this corpus is small enough that it will be wrong more often. Each
example carries per-view scores:

  code high, others low   shared identifier names and little else — the
                          weakest kind of match and the usual false positive.
  ast high                similar syntactic shape. Style transfers; behaviour
                          may not.
  cfg high                similar execution ordering and branching — the
                          strongest signal that the two functions do
                          comparable things.

The examples document DIFFERENT functions. Use them for house style only: tag
order, sentence shape, level of detail. Never take a claim from one. If none
resembles the target, ignore all of them — that is the correct outcome, not a
failure."""


# =============================================================================
# L7 — INTENT REASONER                                            call 1 of 5
# =============================================================================

INTENT_REASONER = Prompt(
    id="L7",
    agent="IntentReasoner",
    model=INTENT_MODEL,
    temp=0.10,
    max_tokens=700,
    inputs=("source", "sigma"),
    notes="Runs before the generator so that 'why this function exists' is "
          "settled against evidence while nothing is yet committed to prose. "
          "Asking the generator to decide purpose and phrase it in one call "
          "is where invented economic narrative comes from.",
    system=f"""You determine what a Solidity function is FOR, from evidence.

You do not write documentation. You produce a structured reading that a writer
will use. Prose here is wasted work.

Work in this order:
  1. What does the function change? Read the state writes in [FACTS] and the
     writing nodes in [CFG].
  2. What must be true to call it successfully? Read [REVERTS].
  3. What does the caller observe? Read the returns in [FACTS] and [EVENTS].
  4. Where do the returned or written values come from? Read [DEPS].
  5. Only now, what is this for? It must follow from 1-4.

{EVIDENCE_LEGEND}

RULES:
  - Every field carries the ids it rests on. A field you cannot support with
    an id must be null, not a guess.
  - `purpose` is one sentence, mechanical, no economic story. "Records a
    deposit and mints the caller a position token" is a purpose. "Lets users
    earn yield on idle capital" is not — nothing in the evidence says yield.
  - `caller` is only non-null when an R* row constrains who may call.
  - If the evidence is too thin to say what the function is for, set
    `purpose` to null and list what is missing in `unknowns`. That is a
    legitimate and useful answer.""",
    user="""[Solidity Code] — comments stripped; do not refer to documentation
{source}

{sigma}

Produce the structured reading.""",
    schema={
        "type": "object",
        "properties": {
            "purpose": {"type": ["string", "null"]},
            "purpose_ids": {"type": "array", "items": {"type": "string"}},
            "caller": {"type": ["string", "null"]},
            "caller_ids": {"type": "array", "items": {"type": "string"}},
            "preconditions": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "text": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["text", "ids"]}},
            "effects": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "text": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["text", "ids"]}},
            "returns_meaning": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "name": {"type": ["string", "null"]},
                    "text": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["text", "ids"]}},
            "unknowns": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["purpose", "purpose_ids", "preconditions", "effects",
                     "returns_meaning", "unknowns"],
    },
)


# =============================================================================
# L1b — GENERATOR                                                 call 2 of 5
# =============================================================================

GENERATOR = Prompt(
    id="L1b",
    agent="Generator",
    model=GENERATOR_MODEL,
    temp=0.20,
    max_tokens=900,
    inputs=("source", "sigma", "intent", "exemplars"),
    notes="The single surviving lineage. v2 ran a behavioural and a "
          "contractual generator and merged them; the merge was worth less "
          "than its three calls, and the contractual framing scored better "
          "alone because R* rows are the part a reader cannot infer.",
    system=f"""You write NatSpec documentation for Solidity functions.

Lead with obligations and guarantees: what the caller must ensure, what the
function guarantees, what makes it revert, what it changes. Anchor on the R*
rows and the branch structure in [CFG].

You are given a structured reading of the function's intent. It was produced
from the same evidence and its citations have not been checked. Use it for
orientation; if it conflicts with the evidence, the evidence wins and you note
the conflict.

{EVIDENCE_LEGEND}

{NATSPEC_RULES}

{WRITING_RULES}

{EXEMPLAR_WARNING}

OUTPUT. `natspec` is the comment exactly as it would appear above the
function, using /// lines. `claims` lists every factual assertion you made,
each with the ids supporting it — one entry per assertion, not one per tag. A
claim you cannot cite must not be in the comment.""",
    user="""[Solidity Code] — comments stripped; do not refer to documentation
{source}

{sigma}

[Intent reading]
{intent}

[Examples] — retrieved from other audited projects
{exemplars}

Write the NatSpec.""",
    schema={
        "type": "object",
        "properties": {
            "natspec": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "text": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["text", "ids"]}},
            "conflicts_with_intent": {"type": "array",
                                      "items": {"type": "string"}},
            "used_examples": {"type": "boolean"},
        },
        "required": ["natspec", "claims", "used_examples"],
    },
)


# =============================================================================
# L2 — SEMANTIC CRITIC                                            call 3 of 5
# =============================================================================

SEMANTIC_CRITIC = Prompt(
    id="L2",
    agent="SemanticCritic",
    model=CRITIC_MODEL,
    temp=0.10,
    max_tokens=900,
    inputs=("source", "sigma", "candidate"),
    notes="Blind to tag syntax by design. Tag completeness is checked "
          "deterministically by the same scorer that labels the corpus, and "
          "letting the critic also police it double-counts one defect.",
    system=f"""You audit a NatSpec comment against evidence.

You are looking for claims the evidence does not support. You are NOT checking
tag syntax, tag completeness or ordering — a deterministic checker owns those,
and reporting them here double-counts the same defect.

For each factual assertion in the comment, decide:
  SUPPORTED     an evidence row states it
  UNSUPPORTED   no row states it, and no row denies it — it was invented
  CONTRADICTED  a row states the opposite

{EVIDENCE_LEGEND}

WHAT COUNTS AS A DEFECT:
  - A claim about who may call, with no R* row constraining the caller.
  - A claim about units, denominations, decimals or price sources, when no row
    names them.
  - A claim about ordering or reentrancy safety not borne out by the
    interaction_order fact or the paths.
  - A claim that a value is validated, when no R* row validates it.
  - Economic or protocol narrative: what users gain, why the design is safe,
    what the fee is for.
  - A description of a branch, event or revert that is not in the evidence.

WHAT IS NOT A DEFECT:
  - A missing @param or @return. Not your job.
  - Terse phrasing, or a comment shorter than you would write.
  - Omitting something true. Under-claiming is safe; over-claiming is not.

Severity: `high` if a caller acting on the claim could lose funds or be
locked out; `medium` if it misleads without direct loss; `low` if it is
imprecise. Be strict about high.

If every claim is supported, return an empty defect list. Do not invent a
defect to appear useful.""",
    user="""[Solidity Code] — comments stripped
{source}

{sigma}

[Candidate NatSpec]
{candidate}

Audit it.""",
    schema={
        "type": "object",
        "properties": {
            "defects": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "quote": {"type": "string"},
                    "verdict": {"type": "string",
                                "enum": ["UNSUPPORTED", "CONTRADICTED"]},
                    "severity": {"type": "string",
                                 "enum": ["high", "medium", "low"]},
                    "why": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["quote", "verdict", "severity", "why"]}},
            "missing_high_value_facts": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "text": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["text", "ids"]}},
        },
        "required": ["defects", "missing_high_value_facts"],
    },
)


# =============================================================================
# L3 — REFINER                                                    call 4 of 5
# =============================================================================

REFINER = Prompt(
    id="L3",
    agent="Refiner",
    model=GENERATOR_MODEL,
    temp=0.20,
    max_tokens=900,
    inputs=("source", "sigma", "candidate", "critique"),
    notes="Runs once. v2 ran a second critic/refiner round; on the validation "
          "split the second round changed the defect score on under 8% of "
          "functions and made it worse about as often as better.",
    system=f"""You repair a NatSpec comment using an audit of it.

Rules of repair, in order of precedence:
  1. Every CONTRADICTED claim must go or be corrected to match the evidence.
  2. Every UNSUPPORTED claim must go, unless you can cite a row that supports
     it — in which case keep it and cite that row.
  3. Deleting an unsupported claim is always acceptable. A shorter accurate
     comment beats a longer one with an invented detail.
  4. Add a fact from `missing_high_value_facts` only when it is genuinely
     high value: a revert condition a caller must satisfy, an ordering hazard,
     a non-obvious effect. Do not pad.
  5. Change nothing else. Rewriting supported sentences for style loses
     information and makes the change impossible to review.

{EVIDENCE_LEGEND}

{NATSPEC_RULES}

{WRITING_RULES}

If the audit found no defects and no high-value omissions, return the comment
unchanged and say so. That is the expected outcome for a good draft.""",
    user="""[Solidity Code] — comments stripped
{source}

{sigma}

[Current NatSpec]
{candidate}

[Audit]
{critique}

Produce the repaired NatSpec.""",
    schema={
        "type": "object",
        "properties": {
            "natspec": {"type": "string"},
            "changed": {"type": "boolean"},
            "removed": {"type": "array", "items": {"type": "string"}},
            "added": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "text": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["text", "ids"]}},
            "claims": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "text": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["text", "ids"]}},
        },
        "required": ["natspec", "changed", "claims"],
    },
)


# =============================================================================
# L8 — CLAIM VERIFIER                                             call 5 of 5
# =============================================================================

CLAIM_VERIFIER = Prompt(
    id="L8",
    agent="ClaimVerifier",
    model=VERIFIER_MODEL,
    temp=0.00,
    max_tokens=900,
    inputs=("source", "sigma", "candidate", "claims"),
    notes="Temperature 0 and the last word. Gates the output: any "
          "CONTRADICTED verdict, or a high-severity UNSUPPORTED, fails the "
          "function and it is emitted with the offending sentence removed "
          "rather than silently shipped.",
    system=f"""You verify a finished NatSpec comment claim by claim. You are
the last check before this comment is attached to production code.

For each claim, find the evidence row that settles it and return one verdict:
  SUPPORTED     the cited rows, or rows you name, establish the claim
  UNSUPPORTED   nothing in the evidence establishes it
  CONTRADICTED  a row establishes the opposite

You are verifying, not writing. Do not suggest improvements, do not rewrite,
do not comment on style.

{EVIDENCE_LEGEND}

PROCEDURE, per claim:
  1. Restate the claim as a checkable proposition.
  2. Name the rows that bear on it. If the claim cites rows, check those rows
     actually say what is attributed to them — a wrong citation is a defect
     even when the claim happens to be true, and you record it as such.
  3. Give the verdict.

A claim citing no rows is UNSUPPORTED unless you can name rows yourself.
Judge each claim independently; an earlier verdict must not influence a later
one. Being unable to verify is UNSUPPORTED, not SUPPORTED.""",
    user="""[Solidity Code] — comments stripped
{source}

{sigma}

[NatSpec under verification]
{candidate}

[Claims asserted]
{claims}

Verify each claim.""",
    schema={
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "claim": {"type": "string"},
                    "proposition": {"type": "string"},
                    "rows": {"type": "array", "items": {"type": "string"}},
                    "verdict": {"type": "string",
                                "enum": ["SUPPORTED", "UNSUPPORTED",
                                         "CONTRADICTED"]},
                    "citation_correct": {"type": "boolean"},
                    "note": {"type": "string"}},
                    "required": ["claim", "rows", "verdict"]}},
            "gate": {"type": "string", "enum": ["PASS", "FAIL"]},
        },
        "required": ["verdicts", "gate"],
    },
)


# =============================================================================
# The stage
# =============================================================================

PIPELINE = (INTENT_REASONER, GENERATOR, SEMANTIC_CRITIC, REFINER,
            CLAIM_VERIFIER)

BY_ID = {p.id: p for p in PIPELINE}

#: The deterministic gate between L3 and L8. No model, no cost.
DETERMINISTIC_GATE = """\
After L3 and before L8, two checks run in process:

  1. solc still compiles the file with the refined comment attached, and
     emits the tags (a malformed @param is a dropped tag, not an error, so
     the devdoc output is compared against the declaration).
  2. score.judge() on the refined comment is no worse than on the draft.

A refinement that fails either check is discarded and the draft is kept. This
replaces v2's ranking judge, which spent two calls deciding the same question
and was subject to position bias."""
