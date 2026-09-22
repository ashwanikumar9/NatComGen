"""Result files are never overwritten, and a run's files stay together."""
import json
import os

import pytest

from natspec_corpus import versioning as V
from natspec_corpus.report import write_report

RESULTS = {"C1": {"n": 2, "claim_support": {"mean": 0.6}, "by_condition": {}},
           "C5": {"n": 2, "claim_support": {"mean": 0.5}, "by_condition": {}}}
LABELS = {"C1": "full system", "C5": "no retrieval"}


def test_split_version_reads_a_numeric_suffix():
    assert V.split_version("main") == ("main", 1)
    assert V.split_version("main_7") == ("main", 7)


def test_split_version_leaves_a_worded_suffix_alone():
    """`main_with_sigma` is a table name, not version `with_sigma` of `main`."""
    assert V.split_version("main_with_sigma") == ("main_with_sigma", 1)
    assert V.split_version("main_without_sigma_3") == ("main_without_sigma", 3)


def test_next_path_leaves_a_free_name_alone(tmp_path):
    assert V.next_path(tmp_path / "r.json") == tmp_path / "r.json"


def test_next_path_steps_past_what_exists(tmp_path):
    (tmp_path / "r.json").write_text("1")
    assert V.next_path(tmp_path / "r.json").name == "r_2.json"
    (tmp_path / "r_2.json").write_text("2")
    assert V.next_path(tmp_path / "r.json").name == "r_3.json"


def test_report_numbers_each_run(tmp_path):
    names = []
    for _ in range(3):
        w = write_report(tmp_path, results=RESULTS, labels=LABELS,
                         manifest={}, run_meta={"split": "val"})
        names.append(w["main.md"].name)
    assert names == ["main.md", "main_2.md", "main_3.md"]
    assert (tmp_path / "tables/main.md").read_text() != ""


def test_nothing_from_an_earlier_run_is_touched(tmp_path):
    first = write_report(tmp_path, results=RESULTS, labels=LABELS,
                         manifest={"run": 1})
    before = first["main.md"].read_text()
    stamp = first["manifest.json"].read_text()
    write_report(tmp_path, results={"C1": RESULTS["C1"]}, labels=LABELS,
                 manifest={"run": 2})
    assert first["main.md"].read_text() == before
    assert first["manifest.json"].read_text() == stamp


def test_a_runs_files_share_one_number(tmp_path):
    """The point of numbering per run rather than per file.

    `conditions` is missing from run 2 because C1 is absent. Numbering each
    file on its own would then give run 3 a `conditions_2.md` sitting beside
    a `main_3.md`, and nothing on disk would say they were different runs.
    """
    write_report(tmp_path, results=RESULTS, labels=LABELS, manifest={})
    write_report(tmp_path, results={"C5": RESULTS["C5"]}, labels=LABELS,
                 manifest={})
    w = write_report(tmp_path, results=RESULTS, labels=LABELS, manifest={})
    assert w["main.md"].name == "main_3.md"
    assert w["conditions.md"].name == "conditions_3.md"
    assert w["manifest.json"].name == "manifest_3.json"
    assert not (tmp_path / "tables/conditions_2.md").exists()


def test_the_returned_map_is_keyed_by_the_logical_name(tmp_path):
    write_report(tmp_path, results=RESULTS, labels=LABELS, manifest={})
    w = write_report(tmp_path, results=RESULTS, labels=LABELS, manifest={})
    assert "main.md" in w and w["main.md"].name == "main_2.md"


def test_the_index_lists_every_run(tmp_path):
    write_report(tmp_path, results=RESULTS, labels=LABELS, manifest={},
                 run_meta={"split": "val", "seeds": [0]})
    write_report(tmp_path, results=RESULTS, labels=LABELS, manifest={},
                 run_meta={"split": "test", "seeds": [0, 1]})
    rows = V.read_ledger(tmp_path)
    assert [r["run"] for r in rows] == [1, 2]
    assert [r["split"] for r in rows] == ["val", "test"]
    index = (tmp_path / "RUNS.md").read_text()
    assert "main_2.md" in index and "| 1 |" in index and "| 2 |" in index


def test_a_deleted_version_is_reused(tmp_path):
    """The number comes from disk, not from a counter.

    A hidden counter would keep climbing after a cleanup and leave gaps that
    the directory itself could not explain.
    """
    write_report(tmp_path, results=RESULTS, labels=LABELS, manifest={})
    w = write_report(tmp_path, results=RESULTS, labels=LABELS, manifest={})
    assert w["main.md"].name == "main_2.md"
    for p in tmp_path.rglob("*_2.*"):
        p.unlink()
    again = write_report(tmp_path, results=RESULTS, labels=LABELS, manifest={})
    assert again["main.md"].name == "main_2.md"


def test_directories_are_versioned_too(tmp_path):
    for expected in ("documented", "documented_2", "documented_3"):
        v = V.RunVersion.open(tmp_path, ["emission_report.json"],
                              ["documented"])
        v.write_json("emission_report.json", [])
        assert v.dir("documented").name == expected
        v.record(stage="emit")


def test_the_switch_restores_overwriting(tmp_path, monkeypatch):
    monkeypatch.setenv(V.ENV, "off")
    a = write_report(tmp_path, results=RESULTS, labels=LABELS, manifest={})
    b = write_report(tmp_path, results=RESULTS, labels=LABELS, manifest={})
    assert a["main.md"] == b["main.md"] == tmp_path / "tables/main.md"
    assert not (tmp_path / "RUNS.jsonl").exists()
