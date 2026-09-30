"""Caller gates: what a function's modifiers establish, and what they do not.

`WRITING_RULES` says a name is not a fact, and gives `onlyOwner` as the
example. That rule is right about one thing and wrong about another, and the
distinction is the whole of this module:

  * A modifier's name does not establish **who** may call the function. Only a
    guard does, and the guard lives in the modifier's body, which is not in
    this function's nodes.
  * A modifier's *application* does establish **that the function is gated**.
    It is written on the declaration, it is already an `F*` row, and refusing
    to say so is not caution — it is dropping a fact the corpus itself
    documents on 52% of gated functions.

So this module reads the gates off the declaration, classifies them, and
offers a principal derived from the name — clearly labelled as name-derived,
so a writing rule can require the sentence to disclose its source.

One thing the corpus taught us, and the reason this is not just a regex for
`only`: **not every `only*` modifier is a caller gate.** Of 225 modifier
applications in NatSpecGold, 123 restrict the caller, 63 are reentrancy locks
and 36 are state preconditions. `onlyLive` and `onlyEnabledRoute` say nothing
about who may call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

CALLER = "caller"
REENTRANCY = "reentrancy"
STATE = "state"
OTHER = "other"

#: Reentrancy locks. A comment may say the function is protected; a detector
#: may use it later. Kept separate from caller gates deliberately.
_REENTRANCY = {"nonreentrant", "noreentrancy", "lock", "locked", "nonreentrancy"}

#: Modifiers that gate on contract or argument STATE, not on the caller.
#: Observed in the corpus: onlyLive, whenNotPaused, onlyOrchestrated,
#: onlyEnabledRoute. Listing them beats guessing, because `onlyLive` looks
#: exactly like a caller gate to a pattern that only knows the word `only`.
_STATE_ROOTS = {"live", "active", "open", "closed", "initialized", "enabled",
                "disabled", "orchestrated", "paused", "unpaused", "started",
                "ended", "expired", "valid", "existing", "ready", "settled",
                "matured", "enabledroute", "notpaused"}

#: Name fragments that denote a principal — someone who calls. The value is
#: the phrase a comment should use.
_PRINCIPALS = {
    "owner": "owner", "admin": "admin", "gov": "governance",
    "governor": "governance", "governance": "governance",
    "operator": "operator", "controller": "controller", "manager": "manager",
    "minter": "minter", "burner": "burner", "hub": "hub",
    "factory": "factory", "keeper": "keeper", "guardian": "guardian",
    "pauser": "pauser", "authority": "authority",
    "authorized": "authorised caller", "authorised": "authorised caller",
    "holder": "holder", "delegate": "delegate", "creator": "creator",
    "farmer": "farmer", "bot": "bot", "role": "role", "whitelisted": "whitelisted caller",
    "self": "contract itself", "eoa": "externally owned account",
}

#: Bare names that are caller gates without an `only` prefix.
_BARE_CALLER = {"auth", "authorized", "authorised", "requiresauth",
                "restricted", "onlyrole", "permissioned"}

_PIECE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z]+|[0-9]+")
_SENTENCE = re.compile(r"[.;\n]")

#: Words that turn a mention of a principal into a statement about who may
#: call. "the owner is set to _owner" is not an access statement.
#: `allow` and `permit` are in here because "Allows the owner to revoke the
#: vesting" is the dominant Ethereum idiom for stating access control, and a
#: cue list without them scores a corpus that prefers that phrasing at near
#: zero — which would not be a finding, it would be a measurement artifact.
#: The cost is a false positive on "Allows the owner to be changed", where the
#: principal is the object of the action rather than its subject. Accepting a
#: few of those is the right trade against missing a whole idiom.
_RESTRICTION = re.compile(
    r"\bonly\b|\brestrict\w*\b|\bpermission\w*\b|\bmust be\b|\bcan(?:not)? be "
    r"called\b|\bcallable\b|\bcaller\b|\brequires?\b|\bauthori[sz]\w*\b|"
    r"\breverts?\b|\bgated?\b|\ballow\w*\b|\bpermit\w*\b", re.I)


def _pieces(name: str) -> List[str]:
    return [p.lower() for p in _PIECE.findall(name or "")]


def classify(name: str) -> str:
    """caller | reentrancy | state | other, from the modifier's name alone."""
    low = (name or "").lower()
    if low in _REENTRANCY:
        return REENTRANCY
    if low in _BARE_CALLER:
        return CALLER
    parts = _pieces(name)
    if not parts:
        return OTHER
    if parts[0] == "when" or (parts[0] == "only" and len(parts) > 1
                              and "".join(parts[1:]) in _STATE_ROOTS):
        return STATE
    if parts[0] == "only":
        rest = parts[1:]
        if any(p in _STATE_ROOTS for p in rest) and not any(
                p in _PRINCIPALS for p in rest):
            return STATE
        return CALLER
    return OTHER


def principal(name: str) -> Optional[str]:
    """The phrase a comment should use for who may call, from the NAME.

    Name-derived and therefore weaker than a guard. Returns None when the name
    carries no recognisable principal, and the writing rule then requires the
    modifier to be named instead of a principal invented.
    """
    parts = [p for p in _pieces(name) if p != "only"]
    hits = [_PRINCIPALS[p] for p in parts if p in _PRINCIPALS]
    if not hits:
        return None
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    if len(out) == 1:
        return out[0]
    # `onlyHolderOrDelegate` names alternatives; `onlyFactoryOwner` names one
    # compound principal. The literal `Or` in the name is what separates them,
    # and guessing wrong turns one role into two.
    joiner = " or " if "or" in parts else " "
    return joiner.join(out)


@dataclass(frozen=True)
class Gate:
    name: str                      # the modifier as written
    kind: str                      # caller | reentrancy | state | other
    principal: Optional[str]       # name-derived, may be None
    fact_id: Optional[str] = None  # the F* row it came from, when known

    @property
    def tokens(self) -> List[str]:
        """Words whose presence in a comment counts as naming this gate."""
        out = [self.name.lower()]
        if self.principal:
            out += [w for w in re.split(r"[ ,]+", self.principal) if len(w) > 2]
        return out


# --------------------------------------------------------------------------
# reading the gates
# --------------------------------------------------------------------------

_KEYWORDS = {"external", "public", "internal", "private", "view", "pure",
             "payable", "returns", "override", "virtual", "memory", "calldata",
             "storage", "constant", "immutable"}


def gates_from_table(table: Optional[dict]) -> List[Gate]:
    """From Σ(f): the `F*` rows whose kind is `modifier`."""
    out: List[Gate] = []
    for row in (table or {}).get("facts", []) or []:
        if row.get("kind") != "modifier":
            continue
        name = (row.get("text") or "").strip()
        if name:
            out.append(Gate(name, classify(name), principal(name),
                            row.get("id")))
    return out


def gates_from_code(code: str) -> List[Gate]:
    """Fallback for a function with no fact table: read the declaration.

    Everything between the parameter list's closing paren and the body's
    opening brace is modifiers, visibility, mutability and `returns(...)`.
    """
    src = code or ""
    i = src.find(")")
    if i < 0:
        return []
    depth, j = 0, -1
    for k, ch in enumerate(src):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "{" and depth == 0:
            j = k
            break
    head = src[i + 1:j] if j > i else ""
    head = re.sub(r"returns\s*\([^)]*\)", " ", head)
    head = re.sub(r"\([^)]*\)", " ", head)        # modifier arguments
    out: List[Gate] = []
    for name in re.findall(r"[A-Za-z_]\w*", head):
        if name.lower() in _KEYWORDS or name in _KEYWORDS:
            continue
        out.append(Gate(name, classify(name), principal(name)))
    return out


_SENDER = re.compile(r"\bmsg\.sender\b|\b_msgSender\s*\(\s*\)")


def gates_from_guards(table: Optional[dict]) -> List[Gate]:
    """`R*` rows that compare the caller — the honest source the rules wanted.

    A contract may gate in the body rather than with a modifier:
    `require(msg.sender == HUB)`, or an internal `_validateCallerIsHub()`.
    Thirteen functions in NatSpecGold document exactly such a restriction with
    no modifier on the declaration, so a checker that only reads modifiers
    calls those correct comments inventions. These rows close that hole, and
    unlike a modifier name they are evidence in the original sense.
    """
    out: List[Gate] = []
    for row in (table or {}).get("reverts", []) or []:
        cond = str(row.get("condition") or row.get("text") or "")
        if not _SENDER.search(cond):
            continue
        names = [n for n in re.findall(r"[A-Za-z_]\w*", cond)
                 if n not in ("msg", "sender", "require", "revert", "assert",
                              "_msgSender", "address", "bool", "string")]
        who = next((_PRINCIPALS[p] for n in names for p in _pieces(n)
                    if p in _PRINCIPALS), None)
        if who is None and names:
            who = " ".join(_pieces(names[0]))
        out.append(Gate(name=cond.strip()[:60], kind=CALLER, principal=who,
                        fact_id=row.get("id")))
    return out


def gates_for(pair: dict, table: Optional[dict] = None) -> List[Gate]:
    """Every gate on this declaration: modifiers plus caller guards.

    Σ(f) first. With no fact table the declaration is read directly, which
    still finds the modifiers — the part that is visible without compiling.
    """
    mods = gates_from_table(table) or gates_from_code(pair.get("code", ""))
    return mods + gates_from_guards(table)


def caller_gates(pair: dict, table: Optional[dict] = None) -> List[Gate]:
    return [g for g in gates_for(pair, table) if g.kind == CALLER]


# --------------------------------------------------------------------------
# does the comment say so?
# --------------------------------------------------------------------------

def states(comment: str, gate: Gate) -> bool:
    """True when some sentence both names the gate and restricts the caller.

    Both conditions, in the same sentence. "Sets the owner to `_owner`"
    mentions an owner and restricts nothing; "can only be called by the owner"
    does both. Checking them separately over the whole comment would count the
    first as an access statement on any function with an `owner` parameter.
    """
    for sentence in _SENTENCE.split(comment or ""):
        if not _RESTRICTION.search(sentence):
            continue
        low = sentence.lower()
        if any(tok in low for tok in gate.tokens):
            return True
    return False


def undeclared(pair: dict, table: Optional[dict], comment: str) -> List[Gate]:
    """Caller gates the comment fails to state. The defect list."""
    return [g for g in caller_gates(pair, table) if not states(comment, g)]


def invented(pair: dict, table: Optional[dict], comment: str) -> List[str]:
    """Principals the comment restricts to that no gate on this function shows.

    The other direction, and the one that matters for a downstream consumer:
    a comment asserting "only the owner may call this" on an ungated function
    is worse than silence.
    """
    gates = gates_for(pair, table)
    known = {t for g in gates for t in g.tokens}
    out: List[str] = []
    for sentence in _SENTENCE.split(comment or ""):
        if not re.search(r"\bonly\b|\bcan only be called\b|\brestricted to\b",
                         sentence, re.I):
            continue
        low = sentence.lower()
        for word, phrase in _PRINCIPALS.items():
            if re.search(rf"\b{re.escape(word)}\b", low) and word not in known \
                    and phrase not in known:
                out.append(word)
    seen, uniq = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq

def misplaced(pair: dict, table: Optional[dict], comment: str) -> List[str]:
    """Fields that state a caller gate but are not @notice or @dev.

    A restriction filed under `@param amount` is attributed by solc to that
    parameter, so the tag now describes the wrong thing and the parameter's
    own description is polluted. The comment still *says* who may call, which
    is why whole-comment gate recall does not catch this — hence a check of
    its own.

    Observed at 3% of functions for G1 and 10% for G8 when the writing rule
    said to use @dev "unless the function has no @dev": with no @dev present
    the model appended the clause to whichever tag came last.
    """
    from .evaluate import fields as _fields
    gates = caller_gates(pair, table)
    if not gates:
        return []
    out: List[str] = []
    for key, text in _fields(comment or "").items():
        if key in ("notice", "dev"):
            continue
        if any(states(text, g) for g in gates):
            out.append(key)
    return out
