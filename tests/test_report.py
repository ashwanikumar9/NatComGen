"""Stage 4: tables and figures.

Generated from a synthetic result set, so the shape of the write-up is fixed
before any real number exists and cannot be tuned to flatter one.
"""
import json
from pathlib import Path

import pytest

from natspec_corpus.report import (CONDITIONS, PRIMARY, SECONDARY,
                                   ablation_figure, ablation_table_rows,
                                   condition_table, flatten, latex_table,
                                   main_table, markdown_table, seed_table,
                                   write_report)


def block(n=10, support=0.9, bleu=0.4):
    return {"n": n, "claim_support_rate": support, "defect_free_rate": 0.8,
            "verifier_pass_rate": 0.95,
            "notice": {"n": n, "bleu": bleu, "rouge_l": 0.5, "coverage": 1.0},
            "param": {"n": 2 * n, "bleu": bleu - 0.1, "rouge_l": 0.4,
                      "coverage": 0.9},
            "return": {"n": n, "bleu": bleu - 0.2, "rouge_l": 0.3,
                       "coverage": 0.8}}


def result(**kw):
    return {"all": block(**kw), "with_sigma": block(n=6, **kw),
            "without_sigma": block(n=4, **kw)}


RESULTS = {"C1": result(), "C2": result(support=0.6, bleu=0.3),
           "C0": result(support=0.4, bleu=0.2)}
LABELS = {"C0": "zero-shot", "C1": "full system", "C2": "no Σ(f)"}

ABLATIONS = {
    "comparisons": {
        "C2": {"n": 10, "diff": -0.12, "ci_low": -0.18, "ci_high": -0.06,
               "p_value": 0.001, "p_adjusted": 0.002, "reject": True,
               "cliffs_delta": -0.5, "magnitude": "large",
               "within_noise": False, "significant": True},
        "C5": {"n": 10, "diff": -0.002, "ci_low": -0.02, "ci_high": 0.016,
               "p_value": 0.04, "p_adjusted": 0.08, "reject": False,
               "cliffs_delta": -0.03, "magnitude": "negligible",
               "within_noise": True, "significant": False},
    },
    "seed_spread": {"full": {"mean": 0.9, "sd": 0.01, "seeds": 3},
                    "C2": {"mean": 0.78, "sd": 0.02, "seeds": 3},
                    "C5": {"mean": 0.898, "sd": 0.01, "seeds": 3}},
}


# -- flattening ------------------------------------------------------------

def test_flatten_exposes_every_metric_a_table_needs():
    f = flatten(block())
    for key, _ in PRIMARY:
        assert key in f
    for key, _ in SECONDARY:
        assert key in f


def test_a_missing_metric_flattens_to_none_not_zero():
    """A metric that was never computed must not read as a score of zero."""
    f = flatten({"n": 3})
    assert f["claim_support_rate"] is None and f["notice_bleu"] is None


# -- formatting ------------------------------------------------------------

def test_markdown_table_is_well_formed():
    md = markdown_table(["a", "b"], [[1, 0.5], [2, None]])
    lines = md.splitlines()
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | 0.500 |"
    assert lines[3].endswith("| — |")


def test_latex_escapes_what_latex_needs_escaped():
    tex = latex_table(["Σ(f) & more", "b"], [["a_b", "50%"]])
    assert r"\Sigma" in tex and r"\&" in tex and r"\_" in tex and r"\%" in tex
    assert tex.startswith(r"\begin{table}") and tex.rstrip().endswith(
        r"\end{table}")


def test_latex_column_spec_matches_the_header_count():
    tex = latex_table(["a", "b", "c"], [[1, 2, 3]])
    assert r"\begin{tabular}{lrr}" in tex


# -- the tables ------------------------------------------------------------

def test_primary_metrics_come_before_secondary_ones():
    """The ordering is the argument: n-gram overlap is not the finding."""
    headers, _ = main_table(RESULTS, LABELS)
    assert headers.index("Claim support") < headers.index("notice BLEU")


def test_main_table_has_one_row_per_configuration():
    headers, rows = main_table(RESULTS, LABELS)
    assert len(rows) == 3
    assert {r[0] for r in rows} == set(LABELS.values())


def test_main_table_can_be_restricted_to_one_condition():
    _, rows = main_table(RESULTS, LABELS, condition="with_sigma")
    assert all(r[1] == 6 for r in rows)


def test_condition_table_reports_all_three_conditions():
    headers, rows = condition_table(RESULTS["C1"])
    assert [r[0] for r in rows] == [label for _, label in CONDITIONS]


def test_ablation_table_puts_noise_beside_the_p_value():
    """A corrected p below 0.05 on a difference inside the seed spread is not
    a result, and the columns sit together so that cannot be missed."""
    headers, rows = ablation_table_rows(ABLATIONS, LABELS)
    assert abs(headers.index("p (Holm)") - headers.index("within noise")) <= 3
    row = next(r for r in rows if r[0] == "no retrieval") \
        if any(r[0] == "no retrieval" for r in rows) else rows[1]
    assert "yes" in row or row[-1] == "yes"


def test_ablation_rows_carry_the_interval_not_just_the_point():
    _, rows = ablation_table_rows(ABLATIONS, LABELS)
    assert any("[" in str(c) and "," in str(c) for r in rows for c in r)


def test_seed_table_lists_every_configuration():
    _, rows = seed_table(ABLATIONS)
    assert {r[0] for r in rows} == {"full", "C2", "C5"}


# -- the figure ------------------------------------------------------------

def test_the_figure_is_written(tmp_path):
    p = ablation_figure(ABLATIONS, LABELS, tmp_path / "f" / "ablations.png")
    assert p.exists() and p.stat().st_size > 1000


def test_the_figure_refuses_an_empty_comparison_set(tmp_path):
    with pytest.raises(ValueError, match="no comparisons"):
        ablation_figure({"comparisons": {}}, LABELS, tmp_path / "x.png")


# -- the whole report ------------------------------------------------------

def test_write_report_emits_every_table_in_both_formats(tmp_path):
    written = write_report(tmp_path, results=RESULTS, labels=LABELS,
                           ablations=ABLATIONS,
                           manifest={"corpus_sha1": "abc"})
    for name in ("main", "main_with_sigma", "main_without_sigma",
                 "conditions", "ablations", "seeds"):
        assert (tmp_path / "tables" / f"{name}.md").exists(), name
        assert (tmp_path / "tables" / f"{name}.tex").exists(), name
    assert (tmp_path / "figures" / "ablations.png").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["corpus_sha1"] \
        == "abc"


def test_a_report_without_ablations_still_produces_the_main_tables(tmp_path):
    write_report(tmp_path, results=RESULTS, labels=LABELS)
    assert (tmp_path / "tables" / "main.md").exists()
    assert not (tmp_path / "tables" / "ablations.md").exists()


def test_tables_are_regenerated_not_appended(tmp_path):
    write_report(tmp_path, results=RESULTS, labels=LABELS)
    first = (tmp_path / "tables" / "main.md").read_text()
    write_report(tmp_path, results=RESULTS, labels=LABELS)
    assert (tmp_path / "tables" / "main.md").read_text() == first
