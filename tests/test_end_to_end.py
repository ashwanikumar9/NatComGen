"""The whole chain, in one test: sources -> corpus -> Σ(f) -> index -> run -> score.

Every other test file checks a component in isolation. This one checks that
the stages actually fit together, which is a different question and the one
that fails in practice: an offset convention that two modules disagree about,
a field one stage writes and the next never reads, a split label that does not
survive the trip. It also exercises the failure paths in place — a retry
inside the runner, a call that never parses, and a resumed run — rather than
against a mock in a vacuum.

It runs a real compiler and a real Slither analysis over a small synthetic
project, so it is slower than the rest of the suite and worth every second.
"""
import json
from pathlib import Path

import pytest

from natspec_corpus import build as corpus_build
from natspec_corpus import sigma_build
from natspec_corpus.cache import CallCache
from natspec_corpus.compile import installed_versions
from natspec_corpus.evaluate import evaluate_run
from natspec_corpus.llm import Client, MockBackend
from natspec_corpus.retrieve import build_index, load_units
from natspec_corpus.runner import load_contexts, run_split

pytestmark = pytest.mark.skipif(not installed_versions(),
                                reason="no solc available")

VAULT = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Math.sol";

interface IVault {
    /// @notice Deposits `amount` of the caller's tokens into the vault.
    /// @param amount the number of tokens to deposit
    /// @return newTotal the vault total after the deposit
    function deposit(uint256 amount) external returns (uint256 newTotal);
}

contract Vault is IVault {
    uint256 public total;
    address public owner;

    event Deposited(address indexed who, uint256 amount);

    /// @inheritdoc IVault
    function deposit(uint256 amount) external returns (uint256 newTotal) {
        require(amount > 0, "zero amount");
        total += Math.double(amount);
        emit Deposited(msg.sender, amount);
        newTotal = total;
    }

    /// @notice Returns the current vault total.
    /// @return the total recorded so far
    function peek() external view returns (uint256) {
        return total;
    }
}
"""

MATH = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library Math {
    /// @notice Doubles the given value.
    /// @param x the value to double
    /// @return the value multiplied by two
    function double(uint256 x) internal pure returns (uint256) {
        return x * 2;
    }
}
"""

ENGINE = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Engine {
    mapping(address => uint256) public balance;

    /// @notice Credits `who` with `amount` inside the engine.
    /// @param who the account to credit
    /// @param amount the amount to add to that account
    function credit(address who, uint256 amount) external {
        require(who != address(0), "zero address");
        balance[who] += amount;
    }

    /// @notice Reads the recorded balance of an account.
    /// @param who the account to read
    /// @return the balance recorded for that account
    function balanceOf(address who) external view returns (uint256) {
        return balance[who];
    }
}
"""


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    """Two audit folders whose names map to a train slug and a val slug."""
    root = tmp_path_factory.mktemp("src")
    train = root / "Trail_of_Bits-UniswapV3Core" / "v3-core-abc" / "contracts"
    val = root / "Trail_of_Bits-Primitive" / "rmm-core-def" / "contracts"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    (train / "Vault.sol").write_text(VAULT)
    (train / "Math.sol").write_text(MATH)
    (val / "Engine.sol").write_text(ENGINE)
    return root


@pytest.fixture(scope="module")
def corpus(sources, tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus") / "NatSpecGold"
    report = corpus_build.build(sources, out)
    return out, report


@pytest.fixture(scope="module")
def with_sigma(corpus):
    out, _ = corpus
    return out, sigma_build.build(out)


# -- stage 1 ---------------------------------------------------------------

def test_the_corpus_builds_and_all_invariants_hold(corpus):
    out, report = corpus
    assert report["quarantined"] == {}
    assert report["pairs_verified"] >= 5
    for name in ("pairs.jsonl", "manifest.json", "splits.json",
                 "index_allowlist.json", "README.md"):
        assert (out / name).exists(), name


def test_inheritdoc_is_resolved_across_files(corpus):
    """Vault.deposit carries only `@inheritdoc IVault`; its prose lives in the
    interface. If resolution breaks, the pair silently becomes undocumented."""
    out, report = corpus
    pairs = [json.loads(l) for l in (out / "pairs.jsonl").read_text().splitlines()]
    dep = [p for p in pairs if p["name"] == "deposit" and p["container"] == "Vault"]
    assert dep, "Vault.deposit produced no verified pair"
    assert dep[0]["inheritdoc"] == "IVault"
    assert "Deposits" in dep[0]["notice"]
    assert set(dep[0]["params"]) == {"amount"}


def test_project_split_labels_survive(corpus):
    out, _ = corpus
    pairs = [json.loads(l) for l in (out / "pairs.jsonl").read_text().splitlines()]
    by_split = {p["project"]: p["split"] for p in pairs}
    assert by_split["uniswap-v3-core"] == "train"
    assert by_split["primitive-rmm"] == "val"


# -- stage 1b --------------------------------------------------------------

def test_sigma_runs_and_joins_by_offset(with_sigma):
    out, report = with_sigma
    assert report["slither_failed"] == 0
    assert report["compile_failed"] == 0
    assert report["joined"] >= 5
    assert report["invariants"] == "all hold"


def test_sigma_carries_reverts_events_and_the_ast_view(with_sigma):
    out, _ = with_sigma
    tables = [json.loads(l) for l in
              (out / "sigma" / "sigma.jsonl").read_text().splitlines()]
    dep = next(t for t in tables
               if t["function"] == "deposit" and t["contract"] == "Vault")
    assert any("zero amount" in r["message"] for r in dep["reverts"])
    assert any(e["event"] == "Deposited" for e in dep["events"])
    assert dep["ast_types"][0] == "FunctionDefinition"
    assert "StructuredDocumentation" not in dep["ast_types"]


# -- retrieval -------------------------------------------------------------

def test_index_covers_only_train_and_all_three_views_populate(with_sigma):
    """Every indexed unit must carry all three views. A unit with an empty CFG
    view is an interface declaration that survived dedup over its own
    implementation — the bug the body-preference ordering exists to stop."""
    out, _ = with_sigma
    idx = build_index(out)
    assert {u.project for u in idx.units} == {"uniswap-v3-core"}
    assert all(u.views["ast"] for u in idx.units)
    assert all(u.views["cfg"] for u in idx.units)


def test_a_val_query_retrieves_from_train_only(with_sigma):
    out, _ = with_sigma
    idx = build_index(out)
    val = load_units(out, "val")
    hits = idx.search(val[0].views, tau=None)
    assert hits and all(h.unit.project == "uniswap-v3-core" for h in hits)


# -- the run, with retries and failures in place --------------------------

def gold_comment(pair):
    lines = []
    if pair["notice"]:
        lines.append("/// @notice " + pair["notice"])
    if pair["dev"]:
        lines.append("/// @dev " + pair["dev"])
    for k, v in pair["params"].items():
        lines.append(f"/// @param {k} {v}")
    for r in pair["returns"]:
        body = " ".join(x for x in (r.get("name") or "", r.get("text") or "")
                        if x)
        lines.append("/// @return " + body)
    return "\n".join(lines)


def make_handler(golds, *, flaky_prompt="L1b", doomed_function=None):
    """A backend that misbehaves the way a real small model misbehaves:
    prose around the JSON on the first attempt, and one function it never
    manages at all."""
    state = {"seen": set()}

    def handler(pid, req):
        user = req["messages"][-1]["content"]
        pair_id = next((k for k, v in golds.items() if v["code"] in user), None)
        if doomed_function and pair_id and golds[pair_id]["name"] == doomed_function:
            return "I am afraid I cannot comply with that request."
        key = (pid, pair_id)
        if pid == flaky_prompt and key not in state["seen"]:
            state["seen"].add(key)
            return "Sure! Here is the documentation you asked for."
        gold = gold_comment(golds[pair_id]) if pair_id else "/// @notice x"
        if pid == "L7":
            return {"purpose": "p", "purpose_ids": ["F1"], "caller": None,
                    "caller_ids": [], "preconditions": [], "effects": [],
                    "returns_meaning": [], "unknowns": []}
        if pid in ("L1b", "L3"):
            return {"natspec": gold, "claims": [{"text": "p", "ids": ["F1"]}],
                    "used_examples": False, "changed": False, "removed": []}
        if pid == "L2":
            return {"defects": [], "missing_high_value_facts": []}
        return {"verdicts": [{"claim": "p", "rows": ["F1"],
                              "verdict": "SUPPORTED"}], "gate": "PASS"}

    return handler


@pytest.fixture(scope="module")
def run(with_sigma, tmp_path_factory):
    out, _ = with_sigma
    golds = {c.pair["id"]: c.pair
             for c in load_contexts(out, "val", with_units=False)}
    backend = MockBackend(make_handler(golds, doomed_function="balanceOf"))
    client = Client(backend, cache=CallCache(tmp_path_factory.mktemp("cache")))
    path = run_split(out, "val", client, build_index(out),
                     out_path=out / "runs" / "val.jsonl", progress=False)
    return out, path, client


def test_every_function_produces_a_record(run):
    out, path, _ = run
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    n_val = len(load_contexts(out, "val", with_units=False))
    assert len(recs) == n_val


def test_a_retry_inside_the_chain_is_recorded_not_hidden(run):
    """The generator returns prose on its first attempt for every function.
    The run must still succeed, and the parse rate must show it."""
    _, _, client = run
    stats = client.stats()
    assert stats["L1b"]["first_attempt_parse_rate"] == 0.0
    assert stats["L7"]["first_attempt_parse_rate"] == 1.0


def test_a_function_the_model_never_manages_becomes_an_error_row(run):
    """One bad function must not take the run down with it."""
    _, path, _ = run
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    bad = [r for r in recs if r.get("error")]
    assert len(bad) == 1 and bad[0]["function"] == "balanceOf"
    assert "no schema-valid JSON" in bad[0]["error"]


def test_good_functions_still_complete(run):
    _, path, _ = run
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    ok = [r for r in recs if not r.get("error")]
    assert ok
    for r in ok:
        assert r["final"].strip()
        assert r["gate"]["compiles"] is True
        assert r["gate"]["tags_emitted"] is True


def test_the_record_carries_the_whole_trace(run):
    _, path, _ = run
    rec = next(json.loads(l) for l in path.read_text().splitlines()
               if not json.loads(l).get("error"))
    for key in ("intent", "draft", "critique", "refined", "gate",
                "verification", "retrieved", "call_L1b"):
        assert key in rec, key
    assert rec["call_L1b"]["attempts"] == 2


# -- resume ----------------------------------------------------------------

def test_a_resumed_run_tops_up_without_redoing_work(run, tmp_path):
    out, path, _ = run
    recs = path.read_text().splitlines()
    partial = tmp_path / "val.jsonl"
    partial.write_text("\n".join(recs[:1]) + "\n")

    golds = {c.pair["id"]: c.pair
             for c in load_contexts(out, "val", with_units=False)}
    backend = MockBackend(make_handler(golds, doomed_function="balanceOf"))
    run_split(out, "val", Client(backend), build_index(out),
              out_path=partial, progress=False)

    after = [json.loads(l) for l in partial.read_text().splitlines()]
    assert len(after) == len(recs)
    done_first = json.loads(recs[0])["pair_id"]
    assert sum(1 for r in after if r["pair_id"] == done_first) == 1
    # the already-done function was never sent to the model again
    sent = [c["messages"][-1]["content"] for c in backend.calls]
    first_pair = golds[done_first]
    assert not any(first_pair["code"] in s for s in sent)


# -- scoring ---------------------------------------------------------------

def test_the_run_scores_against_the_gold_references(run):
    out, path, _ = run
    result = evaluate_run(out, path)
    assert result["errors"] == 1
    assert result["scored"] >= 1
    block = result["all"]
    assert block["notice"]["bleu"] == pytest.approx(1.0)
    assert block["notice"]["coverage"] == pytest.approx(1.0)
    assert block["claim_support_rate"] == pytest.approx(1.0)


def test_functions_with_and_without_a_fact_table_are_reported_apart(run):
    out, path, _ = run
    result = evaluate_run(out, path)
    assert result["with_sigma"]["n"] + result["without_sigma"]["n"] == \
        result["scored"]


# -- ablations -------------------------------------------------------------

def test_an_ablation_runs_the_whole_chain_with_fewer_calls(with_sigma,
                                                           tmp_path):
    out, _ = with_sigma
    golds = {c.pair["id"]: c.pair
             for c in load_contexts(out, "val", with_units=False)}
    backend = MockBackend(make_handler(golds, flaky_prompt=None))
    client = Client(backend)
    path = run_split(out, "val", client, None,
                     out_path=tmp_path / "ablation.jsonl",
                     stages=["L1b"], progress=False)
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    assert all(set(r["stages"]) == {"L1b"} for r in recs)
    assert all(r["retrieved"] == [] for r in recs)
    assert set(client.stats()) == {"L1b"}


def test_dedup_keeps_the_implementation_not_the_interface(corpus):
    """After @inheritdoc resolution an interface declaration and its
    implementation carry identical prose, so one is dropped. It must be the
    declaration: the implementation is the only one with a body, and so the
    only one Σ(f) can describe. Before this was ordered explicitly the
    survivor was whichever file sorted first alphabetically."""
    out, _ = corpus
    pairs = [json.loads(l) for l in (out / "pairs.jsonl").read_text().splitlines()]
    dep = [p for p in pairs if p["name"] == "deposit"]
    assert len(dep) == 1
    assert dep[0]["container"] == "Vault"
    assert dep[0]["inheritdoc"] == "IVault"
    assert "{" in dep[0]["code"]


# -- stage 3: the run's comments, emitted into the file -------------------

def test_stage2_output_feeds_stage3_emission(run):
    """The last link in the chain: take what the runner produced, strip the
    file's documentation, write the generated comments in, and require the
    result to compile with solc reading back what was written.

    This is the join stages 2 and 3 have never been exercised across — a
    record addresses a declaration by offset, and the offsets it carries are
    from the *original* file, not the stripped one."""
    from natspec_corpus.assemble import document_file
    from natspec_corpus.compile import compile_unit, unit_for
    from natspec_corpus.emit import Comment
    from natspec_corpus.evaluate import fields
    from natspec_corpus.extract import build_file
    from natspec_corpus.verify_file import (strip_comments, strip_doc_comments,
                                            verify)

    out, path, _ = run
    contracts = out / "contracts"
    pairs = {json.loads(l)["id"]: json.loads(l)
             for l in (out / "pairs.jsonl").read_text().splitlines() if l.strip()}
    recs = [json.loads(l) for l in path.read_text().splitlines()
            if l.strip() and not json.loads(l).get("error")]
    assert recs

    by_file = {}
    for rec in recs:
        by_file.setdefault(rec["file"], []).append(rec)

    checked = 0
    for rel, group in by_file.items():
        def read(r):
            p = contracts / r
            return p.read_text() if p.is_file() else None

        u = unit_for(rel, read)
        res = compile_unit(u, cache_dir=out / ".compile-cache")
        if not res.ok:
            continue
        src = u.sources[rel]

        # The generated comments, keyed by (container, signature) — never by
        # offset, because stripping moves every offset in the file.
        want = {}
        for rec in group:
            pair = pairs[rec["pair_id"]]
            f = fields(rec["final"])
            want[(pair["container"], pair["signature"])] = Comment(
                notice=f.get("notice", ""), dev=f.get("dev", ""),
                params={k.split(":", 1)[1]: v for k, v in f.items()
                        if k.startswith("param:")},
                returns=[v for k, v in sorted(f.items())
                         if k.startswith("return:")])

        stripped = strip_doc_comments(src, rel=rel)
        m = build_file(rel, stripped)
        by = {}
        for d in m.decls:
            by.setdefault((d.container, d.sig), d.header_start)
        cm = {by[k]: v for k, v in want.items() if k in by and not v.empty}
        if not cm:
            continue

        emitted = document_file(stripped, cm, mode="fill_gaps")

        # nothing but comments changed
        assert strip_comments(src) == strip_comments(emitted), rel
        # it compiles and solc reads the comments back
        rep = verify(rel, src, emitted, u.sources, res.version)
        assert rep.compiles, (rel, rep.errors)
        assert rep.code_unchanged and rep.placement_ok, rel
        # every comment landed on the declaration it was written for
        got = {(a.decl.container, a.decl.sig): a.doc.notice
               for a in build_file(rel, emitted).attachments}
        for key, comment in want.items():
            if key in by and not comment.empty and comment.notice:
                expected = (comment.notice if isinstance(comment.notice, str)
                            else " ".join(comment.notice))
                assert got.get(key) == expected, (rel, key)
        checked += 1
    assert checked >= 1


def test_emission_of_generated_comments_is_idempotent(run):
    from natspec_corpus.assemble import document_file
    from natspec_corpus.emit import Comment
    from natspec_corpus.extract import build_file
    from natspec_corpus.verify_file import strip_doc_comments

    out, path, _ = run
    rec = next(json.loads(l) for l in path.read_text().splitlines()
               if l.strip() and not json.loads(l).get("error"))
    src = (out / "contracts" / rec["file"]).read_text()
    stripped = strip_doc_comments(src, rel=rec["file"])
    m = build_file(rec["file"], stripped)
    d = m.decls[0]
    c = Comment(notice="Generated.", params={p.name: "x" for p in d.params
                                             if p.name})
    once = document_file(stripped, {d.header_start: c}, mode="replace")
    m2 = build_file(rec["file"], once)
    d2 = next(x for x in m2.decls if x.sig == d.sig and x.container == d.container)
    twice = document_file(once, {d2.header_start: c}, mode="replace")
    assert once == twice


# -- resuming an interrupted analysis --------------------------------------

def test_sigma_reuses_its_shards_and_produces_the_same_tables(with_sigma):
    """The second run of a finished stage must analyse nothing and still write
    byte-identical fact tables — otherwise 'resume' is a different pipeline
    than the one that was interrupted."""
    root, first = with_sigma
    before = (root / "sigma" / "sigma.jsonl").read_text(encoding="utf-8")
    again = sigma_build.build(root)
    after = (root / "sigma" / "sigma.jsonl").read_text(encoding="utf-8")
    assert after == before
    assert again["reused"] == again["analysed"] > 0
    assert again["functions"] == first["functions"]
    assert again["joined"] == first["joined"]


def test_sigma_resumes_after_losing_half_its_shards(with_sigma):
    """What a killed run leaves behind: some files analysed, some not. The
    rerun must redo only the missing ones and land on the same output."""
    root, _ = with_sigma
    sigma_build.build(root)                       # make sure all shards exist
    before = (root / "sigma" / "sigma.jsonl").read_text(encoding="utf-8")
    shards = sorted((root / "sigma" / "parts").glob("*.json"))
    assert len(shards) >= 2
    shards[0].unlink()
    report = sigma_build.build(root)
    assert report["reused"] == report["analysed"] - 1
    assert (root / "sigma" / "sigma.jsonl").read_text(encoding="utf-8") == before


def test_a_shard_left_truncated_by_a_kill_is_redone_not_trusted(with_sigma):
    root, _ = with_sigma
    sigma_build.build(root)
    before = (root / "sigma" / "sigma.jsonl").read_text(encoding="utf-8")
    shards = sorted((root / "sigma" / "parts").glob("*.json"))
    shards[0].write_text('[{"id": "F1", "cha', encoding="utf-8")
    report = sigma_build.build(root)
    assert report["reused"] == report["analysed"] - 1
    assert (root / "sigma" / "sigma.jsonl").read_text(encoding="utf-8") == before


def test_editing_a_contract_invalidates_only_its_own_shard(with_sigma,
                                                           tmp_path_factory):
    """A cache that served stale fact tables after an edit would be worse than
    no cache at all, so this checks the invalidation rather than the hit."""
    root, _ = with_sigma
    sigma_build.build(root)
    target = next(p for p in (root / "contracts").rglob("*.sol")
                  if "Engine" in p.name)
    original = target.read_text(encoding="utf-8")
    try:
        target.write_text(original.replace("contract Engine",
                                           "contract Engine /* edited */"),
                          encoding="utf-8")
        report = sigma_build.build(root)
        assert report["reused"] == report["analysed"] - 1
    finally:
        target.write_text(original, encoding="utf-8")
        sigma_build.build(root)


def test_resume_false_ignores_the_shards_entirely(with_sigma):
    root, _ = with_sigma
    sigma_build.build(root)
    report = sigma_build.build(root, resume=False)
    assert report["reused"] == 0 and report["analysed"] > 0
