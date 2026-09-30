"""Caller-gate detection: what a modifier establishes and what it does not."""
from __future__ import annotations

import pytest

from natspec_corpus import access as A


@pytest.mark.parametrize("name,kind", [
    ("onlyOwner", A.CALLER), ("onlyGov", A.CALLER), ("onlyAdmin", A.CALLER),
    ("onlyHolderOrDelegate", A.CALLER), ("auth", A.CALLER),
    ("nonReentrant", A.REENTRANCY), ("lock", A.REENTRANCY),
    ("onlyLive", A.STATE), ("whenNotPaused", A.STATE),
    ("onlyOrchestrated", A.STATE), ("onlyEnabledRoute", A.STATE),
])
def test_classification(name, kind):
    assert A.classify(name) == kind


def test_a_state_gate_is_not_a_caller_gate():
    """The reason this is a lexicon and not a regex for `only`."""
    assert A.classify("onlyLive") != A.CALLER
    assert A.principal("onlyLive") is None


@pytest.mark.parametrize("name,who", [
    ("onlyOwner", "owner"), ("onlyGov", "governance"),
    ("onlyFactoryOwner", "factory owner"),
    ("onlyHolderOrDelegate", "holder or delegate"),
    ("auth", None), ("nonReentrant", None),
])
def test_principal_is_derived_only_where_the_name_carries_one(name, who):
    assert A.principal(name) == who


def test_mentioning_a_principal_is_not_stating_a_restriction():
    gate = A.Gate("onlyOwner", A.CALLER, "owner")
    assert not A.states("Sets the owner to the given address.", gate)
    assert not A.states("Returns the owner.", gate)
    assert A.states("Can only be called by the owner.", gate)
    assert A.states("Reverts unless the caller is the owner.", gate)


def test_restriction_and_principal_must_share_a_sentence():
    gate = A.Gate("onlyOwner", A.CALLER, "owner")
    split = "Returns the owner. This can only be called during the sale."
    assert not A.states(split, gate)


def test_gates_are_read_from_the_declaration_without_a_fact_table():
    pair = {"code": "function setFee(uint256 f) external onlyOwner "
                    "nonReentrant returns (bool) { fee = f; }"}
    got = {(g.name, g.kind) for g in A.gates_for(pair, None)}
    assert ("onlyOwner", A.CALLER) in got
    assert ("nonReentrant", A.REENTRANCY) in got
    assert not any(g.name in ("external", "returns", "bool") for g in
                   A.gates_for(pair, None))


def test_modifier_arguments_are_not_mistaken_for_modifiers():
    pair = {"code": "function f() external onlyRole(ADMIN_ROLE) { }"}
    names = {g.name for g in A.gates_for(pair, None)}
    assert "ADMIN_ROLE" not in names
    assert "onlyRole" in names


def test_sigma_is_preferred_over_the_declaration():
    pair = {"code": "function f() external onlyOwner { }"}
    table = {"facts": [{"id": "F7", "kind": "modifier", "text": "onlyGov"}]}
    got = A.gates_for(pair, table)
    assert [g.name for g in got] == ["onlyGov"]
    assert got[0].fact_id == "F7"


def test_a_caller_guard_in_the_body_counts_as_a_gate():
    """Thirteen correct NatSpecGold comments document exactly this shape."""
    table = {"facts": [], "reverts": [
        {"id": "R2", "kind": "require", "condition": "msg.sender == HUB"}]}
    got = A.gates_for({"code": "function f() external { }"}, table)
    assert got and got[0].kind == A.CALLER and got[0].fact_id == "R2"


def test_undeclared_and_invented_are_opposite_directions():
    pair = {"code": "function f(uint256 x) external onlyOwner { }"}
    assert [g.name for g in A.undeclared(pair, None, "Does a thing.")] \
        == ["onlyOwner"]
    assert A.undeclared(pair, None, "Only the owner may call this.") == []
    open_fn = {"code": "function f() external { }"}
    assert A.invented(open_fn, None, "Can only be called by the admin.") \
        == ["admin"]
    assert A.invented(open_fn, None, "Sets the admin address.") == []


def test_the_allows_idiom_counts_as_stating_the_restriction():
    """"Allows the owner to revoke the vesting" is how much of the Ethereum
    corpus states access control. A cue list without it scores a corpus that
    prefers that phrasing near zero, which is a measurement artifact and not a
    finding."""
    gate = A.Gate("onlyOwner", A.CALLER, "owner")
    assert A.states("Allows the owner to revoke the vesting.", gate)
    assert A.states("Permits the owner to withdraw.", gate)
    assert not A.states("Sets the owner to the given address.", gate)
    assert not A.states("Returns the owner.", gate)


def test_a_gate_stated_inside_a_param_tag_is_misplaced():
    """solc attributes it to that parameter, so the tag describes the wrong
    thing and the parameter's own description is polluted. Whole-comment gate
    recall cannot see this, which is why it has its own check."""
    pair = {"code": "function mint ( address to , uint256 amount ) "
                    "public onlyOwner ( ) { }"}
    bad = ("/// @notice Mints tokens.\n"
           "/// @param to The address\n"
           "/// @param amount The amount Can only be called by the owner\n")
    good = ("/// @notice Mints tokens.\n"
            "/// @dev Can only be called by the owner (`onlyOwner`).\n"
            "/// @param to The address\n"
            "/// @param amount The amount\n")
    assert A.misplaced(pair, None, bad) == ["param:amount"]
    assert A.misplaced(pair, None, good) == []
    # it still counts as stated — the two checks answer different questions
    gate = A.caller_gates(pair, None)[0]
    assert A.states(bad, gate)
