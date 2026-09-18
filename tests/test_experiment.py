"""Stage 4: the ablation matrix.

The properties that matter here are all about isolation — one configuration
must not change what another sees — and about the cache, because the run plan
depends on a prefix-sharing configuration costing only its extra calls.
"""
import json

import pytest

from natspec_corpus.cache import CallCache
from natspec_corpus.experiment import (BY_NAME, CONFIGS, Config,
                                       ExperimentError, RUN_ORDER, drop_all,
                                       drop_families, keep_all, load_results,
                                       ordered, run_config, run_matrix)
from natspec_corpus.llm import Client, MockBackend
from natspec_corpus.runner import Context

TABLE = {
    "facts": [{"id": "F1", "kind": "visibility", "text": "external"},
              {"id": "F2", "kind": "interaction_order", "text": "reenter"}],
    "nodes": [{"id": "N1", "type": "ENTRYPOINT", "src": [0, 1], "text": "",
               "reads": [], "writes": [], "state_reads": [],
               "state_writes": []}],
    "edges": [], "paths": [], "calls": [{"id": "C1", "kind": "builtin",
                                         "target": "require", "node": "N1"}],
    "deps": [{"id": "D1", "variable": "a", "depends_on": ["b"]}],
    "reverts": [{"id": "R1", "kind": "require", "condition": "x",
                 "message": "m", "node": "N1", "text": "require(x)"}],
    "events": [], "ast_types": ["FunctionDefinition"],
}


# -- the filters -----------------------------------------------------------

def test_dropping_a_family_leaves_the_others():
    out = drop_families("deps")(TABLE)
    assert out["deps"] == []
    assert out["reverts"] and out["calls"]


def test_dropping_reverts_also_drops_the_interaction_verdict():
    """`interaction_order` is an F-row derived from the guards; leaving it
    behind would mean the R* ablation still carries the reentrancy finding."""
    out = drop_families("reverts")(TABLE)
    assert out["reverts"] == []
    assert [f["kind"] for f in out["facts"]] == ["visibility"]


def test_a_filter_never_mutates_the_table_it_was_given():
    """The same Context list is reused for every configuration."""
    before = json.dumps(TABLE, sort_keys=True)
    drop_families("deps", "reverts")(TABLE)
    drop_all(TABLE)
    assert json.dumps(TABLE, sort_keys=True) == before


def test_drop_all_removes_the_table_entirely():
    assert drop_all(TABLE) is None
    assert keep_all(TABLE) is TABLE


def test_an_unknown_family_is_refused():
    with pytest.raises(ExperimentError, match="unknown row families"):
        drop_families("nodez")


# -- the matrix ------------------------------------------------------------

def test_every_configuration_is_in_the_run_order():
    assert {c.name for c in CONFIGS} == set(RUN_ORDER)


def test_the_full_system_runs_first_so_the_cache_is_warm():
    assert RUN_ORDER[0] == "C1"
    assert [c.name for c in ordered()][0] == "C1"


def test_ordering_is_by_run_order_not_by_request_order():
    assert [c.name for c in ordered(["C0", "C6", "C1"])] == ["C1", "C6", "C0"]


def test_an_unknown_configuration_is_refused():
    with pytest.raises(ExperimentError, match="unknown configurations"):
        ordered(["C9"])


def test_the_baseline_is_one_call_with_no_evidence():
    c = BY_NAME["C0"]
    assert c.calls == 1 and not c.retrieval and c.sigma(TABLE) is None


def test_call_counts_match_the_plan():
    assert {n: BY_NAME[n].calls for n in RUN_ORDER} == {
        "C1": 5, "C2": 5, "C3": 5, "C4": 5, "C5": 5,
        "C6": 3, "C7": 4, "C0": 1}


# -- running ---------------------------------------------------------------

GOOD = {
    "L7": {"purpose": "p", "purpose_ids": ["F1"], "caller": None,
           "caller_ids": [], "preconditions": [], "effects": [],
           "returns_meaning": [], "unknowns": []},
    "L1b": {"natspec": "/// @notice Adds.", "claims": [{"text": "p",
                                                        "ids": ["F1"]}],
            "used_examples": False},
    "L2": {"defects": [], "missing_high_value_facts": []},
    "L3": {"natspec": "/// @notice Adds.", "changed": False, "removed": [],
           "claims": [{"text": "p", "ids": ["F1"]}]},
    "L8": {"verdicts": [{"claim": "p", "rows": ["F1"],
                         "verdict": "SUPPORTED"}], "gate": "PASS"},
}


def pair(i=1):
    return {"id": f"p{i}", "file": "x/C.sol", "name": "add", "kind": "function",
            "signature": "add(uint256)", "container": "C",
            "container_kind": "contract", "split": "val",
            "visibility": "public", "mutability": "pure",
            "code": "function add(uint256 a) public pure {}",
            "code_start": 0, "doc_spans": [], "notice": "Adds.", "dev": "",
            "params": {}, "returns": [], "doc_raw": "/// @notice Adds."}


def backend():
    return MockBackend(lambda pid, req: GOOD[pid])


def test_a_configuration_writes_one_record_per_function(tmp_path):
    ctxs = [Context(pair=pair(1), table=TABLE),
            Context(pair=pair(2), table=TABLE)]
    p = run_config(tmp_path, BY_NAME["C1"], Client(backend()),
                   contexts=ctxs, out_root=tmp_path / "runs")
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    assert len(recs) == 2
    assert all(r["config"] == "C1" and r["seed"] == 0 for r in recs)


def test_only_the_configured_stages_are_called(tmp_path):
    b = backend()
    c = Client(b)
    run_config(tmp_path, BY_NAME["C6"], c, contexts=[Context(pair=pair(),
                                                             table=TABLE)],
               out_root=tmp_path / "runs")
    assert [x.prompt_id for x in c.log] == ["L7", "L1b", "L8"]


def test_the_sigma_ablation_reaches_the_prompt(tmp_path):
    """C2 removes the fact table; the evidence block must say so rather than
    quietly shipping an empty one."""
    b = backend()
    run_config(tmp_path, BY_NAME["C2"], Client(b),
               contexts=[Context(pair=pair(), table=TABLE)],
               out_root=tmp_path / "runs")
    sent = b.calls[0]["messages"][-1]["content"]
    assert "no fact table" in sent
    assert "R1" not in sent


def test_the_r_star_ablation_removes_only_the_guards(tmp_path):
    b = backend()
    run_config(tmp_path, BY_NAME["C4"], Client(b),
               contexts=[Context(pair=pair(), table=TABLE)],
               out_root=tmp_path / "runs")
    sent = b.calls[0]["messages"][-1]["content"]
    assert "[REVERTS]" not in sent
    assert "D1" in sent and "[CALLS]" in sent


def test_one_configuration_does_not_change_what_the_next_one_sees(tmp_path):
    ctxs = [Context(pair=pair(), table=TABLE)]
    run_config(tmp_path, BY_NAME["C4"], Client(backend()), contexts=ctxs,
               out_root=tmp_path / "runs")
    b = backend()
    run_config(tmp_path, BY_NAME["C1"], Client(b), contexts=ctxs,
               out_root=tmp_path / "runs")
    assert "[REVERTS]" in b.calls[0]["messages"][-1]["content"]


def test_a_resumed_configuration_skips_what_it_finished(tmp_path):
    ctxs = [Context(pair=pair(1), table=TABLE),
            Context(pair=pair(2), table=TABLE)]
    out = tmp_path / "runs"
    run_config(tmp_path, BY_NAME["C1"], Client(backend()), contexts=ctxs[:1],
               out_root=out)
    b = backend()
    p = run_config(tmp_path, BY_NAME["C1"], Client(b), contexts=ctxs,
                   out_root=out)
    assert len([l for l in p.read_text().splitlines() if l.strip()]) == 2
    assert all("p1" not in json.dumps(c) for c in b.calls)


# -- the cache, which the run plan depends on -----------------------------

def test_a_prefix_sharing_configuration_costs_only_its_extra_calls(tmp_path):
    """C6 issues the same L7 and L1b requests as C1. If that were not true the
    84-hour matrix would not come down to 35."""
    cache = CallCache(tmp_path / "cache")
    ctxs = [Context(pair=pair(), table=TABLE)]
    b1 = backend()
    run_config(tmp_path, BY_NAME["C1"], Client(b1, cache=cache), contexts=ctxs,
               out_root=tmp_path / "r1")
    assert len(b1.calls) == 5

    b2 = backend()
    c2 = Client(b2, cache=cache)
    run_config(tmp_path, BY_NAME["C6"], c2, contexts=ctxs,
               out_root=tmp_path / "r2")
    assert [x.prompt_id for x in c2.log] == ["L7", "L1b", "L8"]
    assert [x.cached for x in c2.log] == [True, True, True]
    assert b2.calls == []


def test_two_seeds_never_share_a_cached_call(tmp_path):
    """The seed reaches the cache key as well as the sampler; sharing would
    make the seed axis measure nothing."""
    cache = CallCache(tmp_path / "cache")
    ctxs = [Context(pair=pair(), table=TABLE)]
    b = backend()
    for seed in (0, 1):
        run_config(tmp_path, BY_NAME["C6"], Client(b, cache=cache, seed=seed),
                   contexts=ctxs, seed=seed, out_root=tmp_path / "runs")
    assert len(b.calls) == 6          # three calls, twice, no sharing
    assert {c["options"]["seed"] for c in b.calls} == {0, 1}


def test_the_matrix_runs_every_configuration_and_seed(tmp_path):
    import natspec_corpus.experiment as E
    ctxs = [Context(pair=pair(), table=TABLE)]
    E_load = E.load_contexts
    E.load_contexts = lambda root, split: ctxs
    try:
        paths = run_matrix(tmp_path, lambda s: Client(backend(), seed=s),
                           seeds=(0, 1), names=["C1", "C6"],
                           out_root=tmp_path / "runs")
    finally:
        E.load_contexts = E_load
    assert set(paths) == {"C1", "C6"}
    assert all(len(v) == 2 for v in paths.values())
    rows = load_results(tmp_path / "runs")
    assert len(rows) == 4
    assert {(r["config"], r["seed"]) for r in rows} == {
        ("C1", 0), ("C1", 1), ("C6", 0), ("C6", 1)}
