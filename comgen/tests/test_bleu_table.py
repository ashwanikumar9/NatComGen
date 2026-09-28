"""tools/bleu_table.py — the matched-function comparison.

The point of the tool is that every row is scored on the same functions. Two
`main.md` files cannot be compared directly: NatComGen's run 3 has rows with n
between 33 and 40 because some functions errored, ComGen's has 40, and a
difference between two such rows is partly a difference of function sets. So
the intersection logic is what these tests are about.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import bleu_table as T                                # noqa: E402

from comgen.tests import fixtures as F                           # noqa: E402


def write(root: Path, config: str, seed: int, records) -> Path:
    p = root / config / f"seed{seed}" / "val.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in records),
                 encoding="utf-8")
    return p


def rec(pair_id, config, seed=0, **kw):
    r = {"pair_id": pair_id, "config": config, "seed": seed,
         "final": F.CLEAN, "has_sigma": True}
    r.update(kw)
    return r


def test_matched_ids_intersects_across_configs(tmp_path):
    write(tmp_path, "C1", 0, [rec("a", "C1"), rec("b", "C1"), rec("c", "C1")])
    write(tmp_path, "C2", 0, [rec("a", "C2"), rec("b", "C2")])
    rows = T.load(tmp_path, "val")
    assert T.matched_ids(rows, [0]) == {"a", "b"}, \
        "c is missing from C2, so no row may be scored on it"


def test_an_error_record_removes_that_function_from_every_row(tmp_path):
    write(tmp_path, "C1", 0, [rec("a", "C1"), rec("b", "C1")])
    write(tmp_path, "C2", 0, [rec("a", "C2"),
                              {"pair_id": "b", "config": "C2", "seed": 0,
                               "error": "ollama: 404"}])
    rows = T.load(tmp_path, "val")
    assert T.matched_ids(rows, [0]) == {"a"}


def test_seeds_are_intersected_too_when_pooling(tmp_path):
    write(tmp_path, "C1", 0, [rec("a", "C1", 0), rec("b", "C1", 0)])
    write(tmp_path, "C1", 1, [rec("a", "C1", 1)])
    rows = T.load(tmp_path, "val")
    assert T.matched_ids(rows, None) == {"a"}, \
        "b is missing from seed 1, so pooling must drop it"
    assert T.matched_ids(rows, [0]) == {"a", "b"}


def test_selecting_one_seed_ignores_the_others(tmp_path):
    write(tmp_path, "C1", 0, [rec("a", "C1", 0)])
    write(tmp_path, "C1", 1, [rec("z", "C1", 1)])
    rows = T.load(tmp_path, "val")
    assert T.matched_ids(rows, [1]) == {"z"}


def test_the_average_is_the_unweighted_mean_of_the_fields_present():
    assert T._mean([0.1, 0.2, 0.3]) == pytest.approx(0.2)
    assert T._mean([0.1, None, 0.3]) == pytest.approx(0.2), \
        "a field the comment does not carry must not count as zero"
    assert T._mean([None, None]) is None


def test_a_missing_field_prints_as_a_dash_not_a_zero():
    assert T._n(None) == "—"
    assert T._n(0.0) == "0.000"


def test_the_table_carries_every_field_and_the_average():
    block = {"n": 5,
             "notice": {"bleu": 0.10, "rouge_l": 0.20},
             "param": {"bleu": 0.30, "rouge_l": 0.40}}
    headers, rows = T.rows_for("bleu", [("ComGen", "G1", "full", block)])
    row = dict(zip(headers, rows[0]))
    assert row["notice BLEU"] == "0.100"
    assert row["dev BLEU"] == "—", "this run produced no @dev at all"
    assert row["BLEU avg"] == "0.200", "mean of 0.10 and 0.30"
    assert row["n"] == 5


def test_a_config_that_produced_nothing_is_left_out_not_zeroed():
    headers, rows = T.rows_for("bleu", [("ComGen", "G9", "absent", None)])
    assert rows == []
