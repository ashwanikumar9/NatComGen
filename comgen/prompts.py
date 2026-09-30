"""ComGen's prompts. Four reused as they are, two new.

Reused from `natspec_corpus.prompts_v3`, imported rather than copied so a fix
to a rule reaches both systems:

    L7  FunctionIntent    what this function is for, as citations
    L1b Generator         the draft (round 0 only)
    L2  SemanticCritic    claims the evidence does not support
    L8  Judge             the last word, claim by claim

New here:

    L9  ContractIntent    what the *contract* is for — one call per file,
                          shared by every function in it
    R1  Reviser           the generator repairing its own draft from the
                          merged feedback of both critics

PROMPT IDS ARE CACHE KEYS. `CallCache.key` hashes the id together with the
rendered request, so a reused prompt keeps NatComGen's id and a ComGen call
that happens to be identical to a C1 call is free. The *agent* name is what
the architecture renamed — ClaimVerifier became the Judge — and that is a
label, not a key, so it changes freely.

There is no refiner. R1 carries GENERATOR_MODEL and calls itself the generator
in the second person, because architecturally it is the same agent reading
criticism of its own work. The difference from v2's L3 refiner is not cosmetic:
L3 received one critic's audit and ran once, R1 receives the merged verdict of
both critics — including the deterministic one, whose findings are not
negotiable — and runs until the critics are satisfied or the round budget is
spent.
"""
from __future__ import annotations

from dataclasses import replace

from natspec_corpus.prompts_v3 import (
    EVIDENCE_LEGEND,
    EXEMPLAR_WARNING,
    GENERATOR_MODEL,
    INTENT_MODEL,
    NATSPEC_RULES,
    WRITING_RULES,
    Prompt,
    render,
    sigma_block,
    exemplar_block,
)
from natspec_corpus import prompts_v3 as _P

# Reused verbatim. Same object, same id, same cache line.
FUNCTION_INTENT = _P.INTENT_REASONER          # L7
GENERATOR = _P.GENERATOR                      # L1b (patched below)
SEMANTIC_CRITIC = _P.SEMANTIC_CRITIC          # L2

#: The ClaimVerifier under its architectural name. `replace` keeps the id — and
#: therefore the cache line — while the agent label changes.
JUDGE = replace(_P.CLAIM_VERIFIER, agent="Judge")   # L8



# =============================================================================
# ACCESS: the one amendment ComGen makes to the shared writing rules
# =============================================================================
#
# The shared rule says a name is not a fact and gives `onlyOwner` as the
# example. It is right that a modifier's name does not establish WHO may call
# the function. It is wrong to conclude that nothing may be said: the modifier
# is written on the declaration and is already an F* row, so that the function
# is GATED is a fact. Refusing to state it drops something the corpus itself
# documents on roughly half of its gated functions, and it is the single piece
# of a NatSpec comment a downstream reader most needs.
#
# The amendment is patched into ComGen's own copies of the prompts. It is NOT
# written back into `natspec_corpus.prompts_v3`, so NatComGen's prompts, and
# therefore its cached calls and its published numbers, are untouched.

_OLD_RULE_HEAD = "  - A NAME IS NOT A FACT."

ACCESS_RULE = """\
  - A NAME IS NOT A FACT, WITH ONE EXCEPTION. A modifier called onlyOwner does
    not establish WHO may call the function — only an R* row or a CFG guard
    does. But an F* modifier row does establish THAT the function is gated, and
    you must say so. Write it as one clause naming both the restriction and the
    modifier it comes from: "Can only be called by the owner (`onlyOwner`)
    [F7]". Name the principal only when the modifier's name carries one; when
    it does not, name the modifier alone: "Callable only when `auth` permits
    [F4]". This clause goes in @dev — open a @dev tag if the comment has none.
    NEVER put it inside a @param or @return: solc attributes it to that
    parameter, which both misfiles the restriction and corrupts the
    parameter's description. Write it once, on its own line, with its citation
    on the same line.
  - Not every modifier restricts the caller. `nonReentrant` and `lock` are
    reentrancy guards; `onlyLive`, `whenNotPaused` and `onlyOrchestrated` gate
    on contract state, not on who is calling. Do not describe those as caller
    restrictions.
  - The rest of the name rule stands: parameter names implying units, function
    names implying intent, and variable names implying ownership are still not
    facts."""


#: Measured on the training split by `functionWise.conventions`, and narrowed
#: to the two mismatches that measurement actually found. Length is NOT among
#: them: the generated comments already sit at 0.88-1.00x of gold's median in
#: every field, so a budget would have bought nothing and risked the brevity
#: penalty. The rules below are only what the corpus and the runs disagree on.
STYLE_RULES = """\
  - DO NOT PREPEND AN ARTICLE BY REFLEX. Just under half of @param and @return
    descriptions in this corpus open with a bare noun — "amount of tokens to
    deposit", "address of the vault" — and the runs open with "The" on 77-86%
    of them. Use "The" only where the phrase needs it.
  - Never open @dev with "The". No @dev in this corpus does.
  - @dev carries caller obligations and revert conditions: who may call this,
    and what makes it revert. Write one whenever the evidence shows either.
    Do not use it to restate what @notice already said."""


def _extract_old_rule(text: str) -> str:
    """The shared bullet, taken from the rendered prompt rather than retyped.

    Retyping it would mean a silent no-op the day someone reflows that
    paragraph, and a prompt patch that quietly does nothing is worse than no
    patch: every number downstream would be attributed to a change that never
    reached the model.
    """
    start = text.index(_OLD_RULE_HEAD)
    nxt = text.index("\n  - ", start + len(_OLD_RULE_HEAD))
    return text[start:nxt]


def with_access_rule(prompt):
    """A copy of `prompt` whose writing rules permit stating the gate."""
    old = _extract_old_rule(prompt.system)
    patched = prompt.system.replace(old, ACCESS_RULE + "\n" + STYLE_RULES, 1)
    if patched == prompt.system:                       # pragma: no cover
        raise RuntimeError(
            f"the access rule did not apply to prompt {prompt.id}: the shared "
            f"writing rules have changed and the patch is now a no-op")
    return replace(prompt, system=patched)


WRITING_RULES_ACCESS = WRITING_RULES.replace(
    _extract_old_rule(WRITING_RULES + "\n  - "),
    ACCESS_RULE + "\n" + STYLE_RULES, 1)

# =============================================================================
# L9 — CONTRACT INTENT                                     one call per file
# =============================================================================

CONTRACT_INTENT = Prompt(
    id="L9",
    agent="ContractIntent",
    model=INTENT_MODEL,
    temp=0.10,
    max_tokens=600,
    inputs=("rel", "source"),
    notes="One call per file, not per function, and cached — a 20-function "
          "contract pays for this once. It exists because the single largest "
          "class of invented claim in the C1 records is a function described "
          "in terms of a protocol role the function itself does not show: "
          "'the vault's accounting' when the function only writes a mapping. "
          "Naming the contract's role once, from the contract, means the "
          "generator no longer has to guess it from one function.",
    system=f"""You determine what a Solidity CONTRACT is for, from its code.

You are read once per file and your answer is given to a writer who will then
document each function separately. You are the only part of the system that
sees the whole file, so say the things that are true of the contract and
invisible from any single function — and nothing else.

{EVIDENCE_LEGEND}

Report:
  - `role`: one sentence on what this contract is, in mechanical terms. What
    it stores and what it lets callers do. Not what the protocol is for.
  - `vocabulary`: the terms this contract's own code uses for its domain
    objects — identifier and struct names, event names, the units a value is
    held in when the code names them. This is the wording the writer should
    reuse, so take it from the code, not from your own phrasing.
  - `roles`: privileged callers the contract defines (an owner, a role
    constant, a registry it checks), each with the identifier that shows it.
  - `invariants`: things the contract keeps true across calls, only where the
    code enforces them.
  - `do_not_claim`: things a reader might assume that this contract does NOT
    establish — no fee logic, no oracle, no upgrade path. This list is as
    useful as the others; it is what stops a function being documented in the
    vocabulary of machinery that is not here.

RULES:
  - Everything you report must be visible in the source you were given. An
    imported interface you cannot see is an unknown, not an assumption.
  - No economic story. "Holds deposits and issues shares against them" is a
    role; "lets users earn yield" is not.
  - If the file is an interface or a library with no state, say so plainly —
    that is a useful answer and the lists may be empty.""",
    user="""[File] {rel}

[Solidity Source] — comments stripped; do not refer to documentation
{source}

Describe the contract.""",
    schema={
        "type": "object",
        "properties": {
            "role": {"type": ["string", "null"]},
            "vocabulary": {"type": "array", "items": {"type": "string"}},
            "roles": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "evidence": {"type": "string"}},
                    "required": ["name", "evidence"]}},
            "invariants": {"type": "array", "items": {"type": "string"}},
            "do_not_claim": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["role", "vocabulary", "do_not_claim"],
    },
)


# =============================================================================
# R1 — REVISER (the generator, revising)          once per round after round 0
# =============================================================================

REVISER = Prompt(
    id="R1",
    agent="Generator",
    model=GENERATOR_MODEL,
    temp=0.20,
    max_tokens=900,
    inputs=("source", "sigma", "candidate", "feedback"),
    notes="Replaces v2's L3 refiner. Two differences that matter. It receives "
          "the merged feedback of both critics, with the deterministic "
          "findings marked non-negotiable, so a structural defect can no "
          "longer be argued away by a model that prefers its own sentence. "
          "And it can run more than once: the loop exits when the critics are "
          "satisfied, so a comment with two independent defects gets two "
          "attempts instead of one combined attempt.",
    system=f"""You wrote the comment below. Two critics have now read it and
you are repairing it.

The feedback comes from two sources and they do not have equal standing:

  MUST FIX — from the deterministic critic. These are not opinions. They were
  produced by attaching your comment to the real file, compiling it, and
  comparing what the compiler emitted against the declaration. If it says a
  @param was dropped, the compiler dropped it; the parameter name is wrong or
  the tag is malformed. Fix the tag. Do not delete the parameter to make the
  complaint go away, and do not argue.

  SHOULD FIX — from the semantic critic. Claims the evidence does not support.
  Act on these unless a row you can name supports the claim, in which case
  keep the claim and cite that row.

Order of repair:
  1. Every MUST FIX item.
  2. Every CONTRADICTED claim goes or is corrected to match the evidence.
  3. Every UNSUPPORTED claim goes, unless you can cite a row for it.
  4. A high-value omission is added only if it is genuinely high value: a
     revert condition the caller must satisfy, an ordering hazard, a
     non-obvious effect. Do not pad.
  5. A tag the declaration does not support is DELETED, never filled in.
     `@return null`, `@param x none`, an empty tag — each of these still
     carries the tag, so it still fails, and you will be told the same thing
     again next round. If the function returns nothing, the comment has no
     @return line.
  6. Change nothing else. Rewriting a sentence nobody objected to loses
     information and makes your change impossible to review.

Deleting an unsupported claim is always acceptable. A shorter accurate comment
beats a longer one with an invented detail.

{EVIDENCE_LEGEND}

{NATSPEC_RULES}

{WRITING_RULES_ACCESS}

If there is nothing to fix, return the comment unchanged and set `changed` to
false. That is a legitimate answer and the loop will stop.""",
    user="""[Solidity Code] — comments stripped
{source}

{sigma}

[Your current NatSpec]
{candidate}

[Feedback]
{feedback}

Produce the repaired NatSpec.""",
    schema={
        "type": "object",
        "properties": {
            "natspec": {"type": "string"},
            "changed": {"type": "boolean"},
            "addressed": {"type": "array", "items": {"type": "string"}},
            "declined": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "item": {"type": "string"},
                    "why": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["item", "why"]}},
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
# The architecture, as a table
# =============================================================================

#: Every prompt ComGen can issue. Not a running order — the orchestrator's
#: loop decides how many times R1 and L2 are issued.
PROMPTS = (CONTRACT_INTENT, FUNCTION_INTENT, GENERATOR, SEMANTIC_CRITIC,
           REVISER, JUDGE)

BY_ID = {p.id: p for p in PROMPTS}

#: Stage codes, as used in `Config.stages` and in a record's "stages" field.
#: CD is free — no model, no cost — but it is named here so an ablation can
#: switch it off and the record says which critics were in the group.
STAGES = {
    "IC": "L9  contract intent, one call per file",
    "I7": "L7  function intent",
    "G":  "L1b generator, round 0",
    "CD": "--  deterministic critic (solc + defect score); no model call",
    "CS": "L2  semantic critic",
    "R":  "R1  generator revising; once per round after the first",
    "J":  "L8  judge",
}

ALL_STAGES = ("IC", "I7", "G", "CD", "CS", "R", "J")

__all__ = ["FUNCTION_INTENT", "GENERATOR", "SEMANTIC_CRITIC", "JUDGE",
           "CONTRACT_INTENT", "REVISER", "PROMPTS", "BY_ID", "STAGES",
           "ALL_STAGES", "render", "sigma_block", "exemplar_block",
           "EVIDENCE_LEGEND"]


# Applied last, so `with_access_rule` and the prompt objects both exist. The
# generator is the only agent that writes a comment from scratch; the reviser
# carries the rules through its own f-string above.
GENERATOR = with_access_rule(GENERATOR)
PROMPTS = tuple(GENERATOR if p.id == GENERATOR.id else p for p in PROMPTS)
BY_ID = {p.id: p for p in PROMPTS}
