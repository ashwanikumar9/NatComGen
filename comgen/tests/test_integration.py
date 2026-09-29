"""ComGen inside NatComGen: the seams, not the parts.

Three things have to hold for this package to be worth putting in the same
repository rather than forking it, and none of them is visible from a unit test
of either side:

  1. A ComGen record flows through the existing evaluation unmodified, so
     ComGen is one more column beside C1 rather than a separate result nobody
     can compare.
  2. ComGen's results land inside `comgen/`, so a ComGen run cannot disturb the
     C0–C7 records it is being compared against.
  3. The C0–C7 configurations are byte-identical to the ones that produced the
     published numbers, despite `run_config` having grown a parameter.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from natspec_corpus import experiment as NX
from natspec_corpus.evaluate import aggregate, score_record
from natspec_corpus.report import markdown_table

import comgen
from comgen import experiment as X
from comgen import orchestrator as O
from comgen import report as R
from comgen.tests import fixtures as F


# --------------------------------------------------------------------------
# 1. the record is readable by the existing evaluation
# --------------------------------------------------------------------------

def comgen_record(**kw) -> dict:
    rec = O.generate(F.ctx(), F.client(F.default_script(**kw)), None)
    rec["config"] = "G1"
    rec["seed"] = 0
    return rec


def test_score_record_reads_a_comgen_record():
    s = score_record(comgen_record(), F.pair())
    assert s["pair_id"] == "p1"
    assert s["support_rate"] == 1.0
    assert s["gate_verdict"] == "PASS"
    assert s["by_kind"]["notice"]["bleu"] > 0


def test_the_defect_column_is_populated_from_the_winning_round():
    """`evaluate` reads `rec["gate"]["refined_defects"]`. If ComGen left that
    key out, the defect-free column of every table would silently read as
    None — a blank cell where a regression should have been visible."""
    clean = score_record(comgen_record(), F.pair())
    dirty = score_record(
        comgen_record(drafts=(F.DIRTY,), revisions=(F.DIRTY,)), F.pair())
    assert clean["defects"] == []
    assert len(dirty["defects"]) == 4


def test_aggregate_produces_a_full_row():
    """Including the `with_sigma` / `without_sigma` split, which is what makes
    a ComGen row line up with a C-series row in the main table."""
    rows = [score_record(comgen_record(), F.pair()) for _ in range(3)]
    agg = aggregate(rows)
    assert set(agg) == {"all", "with_sigma", "without_sigma"}
    assert agg["all"]["n"] == 3
    assert agg["all"]["verifier_pass_rate"] == 1.0
    assert agg["all"]["claim_support_rate"] == 1.0
    assert agg["all"]["defect_free_rate"] == 1.0
    assert agg["with_sigma"]["n"] == 3
    assert agg["without_sigma"]["n"] == 0


# --------------------------------------------------------------------------
# 2. results stay inside comgen/
# --------------------------------------------------------------------------

def test_results_default_inside_the_package():
    assert comgen.RUNS.is_relative_to(comgen.ROOT)
    assert comgen.TABLES.is_relative_to(comgen.ROOT)
    assert comgen.ROOT.name == "comgen"


def test_a_run_writes_where_it_was_told_and_nowhere_else(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    out = tmp_path / "comgen-runs"
    client = F.client(F.default_script())
    path = X.run_one(corpus, "G1", client, split="val", seed=0,
                     out_root=out, contexts=[F.ctx()])
    assert path == out / "G1" / "seed0" / "val.jsonl"
    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["config"] == "G1"
    assert rec["architecture"] == "comgen"
    assert rec["pair_id"] == "p1"
    # Nothing was written into the corpus root.
    assert list(corpus.iterdir()) == []


def test_a_run_resumes_instead_of_duplicating(tmp_path):
    out = tmp_path / "runs"
    for _ in range(2):
        X.run_one(tmp_path, "G1", F.client(F.default_script()), split="val",
                  seed=0, out_root=out, contexts=[F.ctx()])
    lines = [l for l in (out / "G1" / "seed0" / "val.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1, "the second run must skip the id already written"


def test_the_round_budget_reaches_the_pipeline(tmp_path):
    """G3 exists to cut the loop to one round. If `pipeline_for` did not bind
    ROUNDS, G1 and G3 would be the same configuration and the round-count
    result would be a tautology."""
    for name, expect in (("G1", 3), ("G3", 1)):
        s = F.default_script(drafts=(F.DIRTY,), revisions=(F.DIRTY,))
        X.run_one(tmp_path, name, F.client(s), split="val", seed=0,
                  out_root=tmp_path / name, contexts=[F.ctx()])
        rec = json.loads((tmp_path / name / name / "seed0" / "val.jsonl")
                         .read_text(encoding="utf-8").strip())
        assert rec["rounds_allowed"] == expect
        assert rec["rounds_used"] == expect


def test_switched_off_stages_are_honoured_per_configuration(tmp_path):
    s = F.default_script()
    X.run_one(tmp_path, "G6", F.client(s), split="val", seed=0,
              out_root=tmp_path / "r", contexts=[F.ctx()])
    assert s.count("L9") == 0, "G6 removes the contract intent call"


def test_sigma_is_filtered_on_the_way_into_the_prompt(tmp_path):
    """G2 drops Σ(f). It must drop it from the prompt, never from the table on
    disk — the same rule the C-series ablations run under."""
    s = F.default_script()
    client = F.client(s)
    ctx = F.ctx()
    X.run_one(tmp_path, "G2", client, split="val", seed=0,
              out_root=tmp_path / "r", contexts=[ctx])
    sent = [c for c in client.backend.calls if c["_prompt_id"] == "L1b"][0]
    assert "no fact table" in sent["messages"][-1]["content"]
    assert ctx.table is not None, "the context's own table must be untouched"


# --------------------------------------------------------------------------
# 3. the C-series is unchanged
# --------------------------------------------------------------------------

def test_the_c_series_configurations_are_untouched():
    """`run_config` grew a `generate_fn` parameter for ComGen's sake. These are
    the numbers that parameter must not have moved."""
    assert [c.name for c in NX.CONFIGS] == ["C1", "C0", "C2", "C3", "C4", "C5",
                                            "C6", "C7"]
    assert NX.RUN_ORDER == ["C1", "C6", "C7", "C5", "C3", "C4", "C2", "C0"]
    assert NX.FULL.stages == ("L7", "L1b", "L2", "L3", "L8")
    assert NX.BY_NAME["C0"].stages == ("L1b",)
    assert NX.BY_NAME["C7"].stages == ("L1b", "L2", "L3", "L8")


def test_run_config_still_defaults_to_the_five_call_runner():
    import inspect
    from natspec_corpus.runner import generate as five_call
    for fn in (NX.run_config, NX.run_matrix):
        assert inspect.signature(fn).parameters["generate_fn"].default \
            is five_call, "the default must stay the architecture the " \
                          "published ablations were run with"


def test_comgen_reuses_the_shared_prompt_ids_so_the_cache_still_hits():
    """A reused prompt keeps NatComGen's id, because the id is half the cache
    key. Renaming the ClaimVerifier to Judge must not have cost the L8 cache."""
    from comgen import prompts as P
    from natspec_corpus import prompts_v3 as V
    assert P.JUDGE.id == V.CLAIM_VERIFIER.id == "L8"
    assert P.JUDGE.agent == "Judge"
    assert P.JUDGE.system == V.CLAIM_VERIFIER.system
    # The generator is deliberately NOT shared any more: ComGen amends the
    # "a name is not a fact" rule so a caller gate may be stated, and that
    # amendment must not reach NatComGen's prompts, its cache lines, or its
    # published numbers. Everything else is still the same object.
    assert P.GENERATOR is not V.GENERATOR
    assert P.GENERATOR.id == V.GENERATOR.id == "L1b"
    assert "WITH ONE EXCEPTION" in P.GENERATOR.system
    assert "WITH ONE EXCEPTION" not in V.GENERATOR.system
    assert P.FUNCTION_INTENT is V.INTENT_REASONER
    assert P.SEMANTIC_CRITIC is V.SEMANTIC_CRITIC
    assert P.SEMANTIC_CRITIC is V.SEMANTIC_CRITIC
    assert {p.id for p in P.PROMPTS} == {"L9", "L7", "L1b", "L2", "R1", "L8"}


# --------------------------------------------------------------------------
# the new tables
# --------------------------------------------------------------------------

def rows_for_report():
    out = []
    for i, kw in enumerate(({}, {"drafts": (F.DIRTY,),
                                 "revisions": (F.CLEAN,)},
                            {"drafts": (F.DIRTY,), "revisions": (F.DIRTY,)})):
        rec = O.generate(F.ctx(), F.client(F.default_script(**kw)), None)
        rec.update(config="G1", seed=0, pair_id=f"p{i}")
        out.append(rec)
    return out


def test_rounds_table_counts_what_the_loop_did():
    headers, rows = R.rounds_rows(rows_for_report())
    assert rows and rows[0][0] == "G1"
    table = dict(zip(headers, rows[0]))
    assert table["n"] == 3
    assert table["clean at r0"] == "0.333"        # one of three needed nothing
    assert table["improved by revision"] == "0.333"   # one was repaired
    assert table["still blocking"] == "0.333"     # one never came clean
    assert markdown_table(headers, rows).startswith("| config |")


def test_cost_table_measures_calls_rather_than_assuming_them():
    headers, rows = R.cost_rows(rows_for_report())
    table = dict(zip(headers, rows[0]))
    assert float(table["min"]) == 5.0 or table["min"] == 5
    assert int(table["max"]) == 9, "three rounds: L9 L7 L1b L2x3 R1x2 L8"
    assert float(table["mean calls"]) > 5


def test_the_extra_tables_share_the_main_tables_run_number():
    """Every file from one run carries one number. A `rounds_3.md` that did not
    belong to `main_3.md` would be worse than no table at all."""
    assert set(R._EXTRA_TARGETS).isdisjoint(R._MAIN_TARGETS)
    from natspec_corpus.report import write_report
    import inspect
    src = inspect.getsource(write_report)
    for name in ("main", "conditions", "ablations", "seeds"):
        assert f'"{name}"' in src, \
            f"report.write_report no longer writes {name}; _MAIN_TARGETS in " \
            f"comgen/report.py is out of step and the run numbers will drift"


# --------------------------------------------------------------------------
# a critic that did not run measured nothing
#
# From the first full run: G0 (no critic at all) and G5 (no deterministic
# critic) both scored a perfect 1.000 defect-free rate, for the sole reason
# that nobody looked. `evaluate.aggregate` counts a record as defect-free when
# `gate.refined_defects` is empty, and a switched-off critic produces an empty
# finding list. An unmeasured quantity must be blank, never perfect.
# --------------------------------------------------------------------------

def test_a_switched_off_deterministic_critic_records_no_defect_measurement():
    rec = O.generate(F.ctx(), F.client(F.default_script(drafts=(F.DIRTY,))),
                     None, stages=["I7", "G", "CS", "R", "J"])
    assert rec["gate"]["measured"] is False
    assert rec["gate"]["refined_defects"] is None
    assert rec["gate"]["draft_defects"] is None
    assert rec["gate"]["passed"] is None
    assert rec["gate"]["applicable"] is False


def test_the_defect_free_rate_excludes_a_config_that_never_measured():
    """The bug, end to end: G0's draft is defective and its defect-free rate
    must be absent, not 1.000."""
    rec = O.generate(F.ctx(), F.client(F.default_script(drafts=(F.DIRTY,))),
                     None, stages=["G", "J"])
    s = score_record(rec, F.pair())
    assert s["defects"] is None
    assert "defect_free_rate" not in aggregate([s])["all"], \
        "an unmeasured config must not appear in the defect-free column"


def test_a_running_deterministic_critic_still_reports_normally():
    rec = O.generate(F.ctx(), F.client(F.default_script(drafts=(F.DIRTY,),
                                                        revisions=(F.CLEAN,))),
                     None)
    assert rec["gate"]["measured"] is True
    assert rec["gate"]["refined_defects"] == []
    assert rec["gate"]["passed"] is True
    assert aggregate([score_record(rec, F.pair())])["all"][
        "defect_free_rate"] == 1.0


def test_the_loop_table_blanks_columns_nothing_measured():
    def rec_for(stages, **kw):
        r = O.generate(F.ctx(), F.client(F.default_script(**kw)), None,
                       stages=stages)
        r.update(config="G0" if stages == ["G", "J"] else "G5", seed=0)
        return r

    no_critic = rec_for(["G", "J"], drafts=(F.DIRTY,))
    headers, rows = R.rounds_rows([no_critic])
    row = dict(zip(headers, rows[0]))
    assert row["clean at r0"] == "—", "no critic ran; nothing was judged"
    assert row["still blocking"] == "—"

    semantic_only = rec_for(["I7", "G", "CS", "R", "J"], drafts=(F.DIRTY,))
    headers, rows = R.rounds_rows([semantic_only])
    row = dict(zip(headers, rows[0]))
    assert row["clean at r0"] != "—", "the semantic critic did judge the draft"
    assert row["still blocking"] == "—", \
        "only the deterministic critic decides what is blocking"


def test_g8_removes_both_components_the_ablations_found_inert():
    """G8 is the shipping candidate: no retrieval, no semantic critic.

    Chosen after seeing the val ablations — retrieval at p=.887 and the
    semantic critic at p=1.000, the latter costing 1.33 calls per function —
    so val is its selection set and the test split is where its number counts.
    """
    cfg = X.BY_NAME["G8"]
    assert cfg.retrieval is False
    assert "CS" not in cfg.stages, "the semantic critic is gone"
    assert "CD" in cfg.stages, "the deterministic critic is what drives the loop"
    assert "R" in cfg.stages and "J" in cfg.stages
    assert X.ROUNDS["G8"] == X.DEFAULT_ROUNDS


def test_g8_runs_after_g7_so_it_reuses_those_calls(tmp_path):
    """Both drop retrieval, so their L9/L7/L1b requests are identical and the
    cache serves them — which is why G8 costs about an hour rather than three."""
    assert X.RUN_ORDER.index("G8") == X.RUN_ORDER.index("G7") + 1
    s = F.default_script()
    X.run_one(tmp_path, "G8", F.client(s), split="val", seed=0,
              out_root=tmp_path / "r", contexts=[F.ctx()])
    assert s.count("L2") == 0, "no semantic critic call"
    assert s.count("L1b") == 1
