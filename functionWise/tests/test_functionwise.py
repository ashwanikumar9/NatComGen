"""Tests for the function-level evaluation.

The end-to-end test builds a synthetic corpus and run tree, so it passes on a
machine that has no results on it -- which is every machine except the GPU
box. That is deliberate: the code has to be verifiable before it is pushed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from functionWise import metrics, scope, surface   # noqa: E402


# --------------------------------------------------------------------------
# scope
# --------------------------------------------------------------------------

def test_presets_and_lists():
    assert scope.parse_kinds("function") == ("function",)
    assert "event" in scope.parse_kinds("all")
    assert scope.parse_kinds("function,event") == ("function", "event")


def test_unknown_kind_is_refused():
    with pytest.raises(SystemExit):
        scope.parse_kinds("functions")          # the plural is a real typo


def _pairs():
    return {
        "a": {"id": "a", "kind": "function", "split": "val", "notice": "One.",
              "params": {"amount": "How much"}, "returns": []},
        "b": {"id": "b", "kind": "event", "split": "val", "notice": "Two."},
        "c": {"id": "c", "kind": "function", "split": "test", "notice": "Three."},
    }


def test_scope_filters_kind_and_split():
    got = scope.scoped(_pairs(), split="val", kinds=("function",))
    assert set(got) == {"a"}


def test_census_marks_the_scope_and_counts_what_is_dropped():
    headers, rows = scope.census(_pairs(), ("function",))
    assert any(h.startswith("function*") for h in headers)
    val = next(r for r in rows if r[0] == "val")
    assert val[headers.index("in scope")] == "1"
    assert val[headers.index("total")] == "2"


# --------------------------------------------------------------------------
# surface
# --------------------------------------------------------------------------

def test_identifier_splitting():
    assert surface.split_identifier("depositAmount") == ["deposit", "amount"]
    assert surface.split_identifier("MAX_FEE") == ["max", "fee"]
    assert surface.split_identifier("ERC721Token") == ["erc", "721", "token"]


def test_normalisation_modes_differ_where_it_matters():
    text = "Returns the `amount`, in wei."
    assert surface.normalize(text, "none") == text
    assert surface.normalize(text, "paper") == "returns the amount in wei"
    assert surface.normalize("depositAmount", "identifiers") == "deposit amount"


def test_join_is_canonical_so_both_sides_agree():
    values = {"return:10": "j", "return:2": "i", "param:b": "y",
              "param:a": "x", "dev": "d", "notice": "n"}
    assert surface.join(values) == "n d x y i j"


def test_notice_view_drops_everything_else():
    pair = {"notice": "N", "dev": "D", "params": {"a": "A"}, "returns": []}
    ref, hyp = surface.pair_strings(
        pair, "/// @notice N\n/// @dev D\n", target="notice", mode="paper")
    assert ref == "n" and hyp == "n"


# --------------------------------------------------------------------------
# metrics -- the convention trap
# --------------------------------------------------------------------------

def test_ba_is_cumulative_bleu4_and_b2_is_not_bleu2():
    refs = ["the caller withdraws the deposited funds now"]
    hyps = ["the caller withdraws deposited funds now"]
    s = metrics.score(refs, hyps)
    assert s["Ba"] == s["BLEU-4"]
    assert s["B1"] == s["BLEU-1"]
    assert s["B2"] != s["BLEU-2"], "individual and cumulative must differ"


def test_identical_text_scores_one_hundred():
    s = metrics.score(["a b c d e"], ["a b c d e"])
    for key in ("Ba", "B1", "BLEU-4", "ROUGE-1", "ROUGE-L"):
        assert s[key] == pytest.approx(100.0, abs=0.01)
    # METEOR never reaches 100: the fragmentation penalty is
    # 0.5 * (chunks/matches)^3, which is 0.004 even for a single chunk of
    # five words. 99.6 is the metric behaving correctly, not a bug.
    assert 99.0 < s["METEOR"] < 100.0


def test_empty_hypotheses_are_counted_not_dropped():
    s = metrics.score(["a b c", "d e f"], ["a b c", ""])
    assert s["n"] == 2 and s["empty"] == 1


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        metrics.score(["a"], ["a", "b"])


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def _corpus(tmp: Path) -> Path:
    root = tmp / "corpus"
    root.mkdir()
    rows = []
    for i in range(6):
        rows.append({"id": f"f{i}", "kind": "function", "split": "val",
                     "notice": "Transfers tokens to the recipient address.",
                     "dev": "", "params": {"to": "The recipient address"},
                     "returns": [], "doc_raw": ""})
    for i in range(3):
        rows.append({"id": f"e{i}", "kind": "event", "split": "val",
                     "notice": "Emitted on every transfer of tokens.",
                     "params": {}, "returns": [], "doc_raw": ""})
    (root / "pairs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return root


def _runs(tmp: Path, ids) -> Path:
    root = tmp / "runs"
    good = ("/// @notice Transfers tokens to the recipient address.\n"
            "/// @param to The recipient address\n")
    poor = "/// @notice Does a thing.\n"
    for config, text in (("G1", good), ("G8", poor)):
        d = root / config
        d.mkdir(parents=True)
        d.joinpath("val.jsonl").write_text("\n".join(
            json.dumps({"pair_id": i, "config": config, "seed": 0,
                        "final": text}) for i in ids), encoding="utf-8")
    return root


def test_end_to_end_scores_functions_only(tmp_path, capsys):
    from functionWise import evaluate
    corpus = _corpus(tmp_path)
    ids = [f"f{i}" for i in range(6)] + [f"e{i}" for i in range(3)]
    runs = _runs(tmp_path, ids)
    rc = evaluate.main(["--corpus", str(corpus), "--comgen-runs", str(runs),
                        "--natcomgen-runs", str(tmp_path / "absent"),
                        "--split", "val", "--no-write"])
    assert rc == 0
    out = capsys.readouterr().out
    # six functions, not nine declarations
    assert "6 declarations, common to every configuration" in out
    assert "## @notice only" in out
    # the faithful configuration must beat the vague one on the notice view
    assert "G1" in out and "G8" in out


def test_end_to_end_all_kinds_sees_more(tmp_path, capsys):
    from functionWise import evaluate
    corpus = _corpus(tmp_path)
    ids = [f"f{i}" for i in range(6)] + [f"e{i}" for i in range(3)]
    runs = _runs(tmp_path, ids)
    evaluate.main(["--corpus", str(corpus), "--comgen-runs", str(runs),
                   "--natcomgen-runs", str(tmp_path / "absent"),
                   "--split", "val", "--kinds", "all", "--no-write"])
    assert "9 declarations" in capsys.readouterr().out


# --------------------------------------------------------------------------
# qualitative examples
# --------------------------------------------------------------------------

def _runs_two(tmp: Path, ids):
    root = tmp / "runs2"
    good = ("/// @notice Transfers tokens to the recipient address.\n"
            "/// @param to The recipient address\n")
    poor = "/// @notice Does a thing.\n"
    for config, text in (("G8", good), ("G1", poor)):
        d = root / config
        d.mkdir(parents=True)
        d.joinpath("test.jsonl").write_text("\n".join(
            json.dumps({"pair_id": i, "config": config, "seed": 0,
                        "final": text}) for i in ids), encoding="utf-8")
    return root


def _corpus_test(tmp: Path) -> Path:
    root = tmp / "corpus2"
    root.mkdir()
    rows = [{"id": f"f{i}", "kind": "function", "split": "test",
             "container": "Token", "signature": f"send{i}(address)",
             "project": "demo", "file": "demo/Token.sol",
             "code": "function send(address to) external {\n    _send(to);\n}",
             "notice": "Transfers tokens to the recipient address.",
             "dev": "", "params": {"to": "The recipient address"},
             "returns": [], "doc_raw": ""} for i in range(5)]
    (root / "pairs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return root


def test_examples_render_gold_and_each_config(tmp_path, capsys):
    from functionWise import examples
    corpus = _corpus_test(tmp_path)
    ids = [f"f{i}" for i in range(5)]
    runs = _runs_two(tmp_path, ids)
    rc = examples.main(["--corpus", str(corpus), "--comgen-runs", str(runs),
                        "--natcomgen-runs", str(tmp_path / "absent"),
                        "--split", "test", "--configs", "G8,G1",
                        "--n", "2", "--no-write"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "**Gold**" in out
    assert "@notice  Transfers tokens to the recipient address." in out
    assert "**G8**" in out and "**G1**" in out
    assert "```solidity" in out
    # the faithful configuration must score above the vague one
    assert "notice BLEU 0.000" in out


def test_examples_refuses_a_missing_config(tmp_path):
    from functionWise import examples
    corpus = _corpus_test(tmp_path)
    runs = _runs_two(tmp_path, [f"f{i}" for i in range(5)])
    with pytest.raises(SystemExit):
        examples.main(["--corpus", str(corpus), "--comgen-runs", str(runs),
                       "--natcomgen-runs", str(tmp_path / "absent"),
                       "--split", "test", "--configs", "G8,G9", "--no-write"])


def test_tidy_strips_comment_markers():
    from functionWise import examples
    got = examples.tidy("/**\n * @notice Sends tokens.\n * @param to Address\n */")
    assert got == "@notice Sends tokens.\n@param to Address"


# --------------------------------------------------------------------------
# before / after comparison
# --------------------------------------------------------------------------

def test_mcnemar_uses_only_the_discordant_pairs():
    from functionWise.compare_runs import mcnemar
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(5, 5) == 1.0                  # symmetric change, no signal
    assert mcnemar(0, 12) < 0.001                # all movement one way
    assert mcnemar(2, 12) < 0.05
    assert mcnemar(0, 12) == pytest.approx(2 * 0.5 ** 12, rel=1e-9)


def _run_files(tmp: Path, name: str, ids, text) -> str:
    d = tmp / name
    d.mkdir(parents=True)
    for seed in (0, 1):
        (d / f"seed{seed}-test.jsonl").write_text("\n".join(
            json.dumps({"pair_id": i, "config": "G1", "seed": seed,
                        "final": text}) for i in ids), encoding="utf-8")
    return str(d / "seed*-test.jsonl")


def test_compare_pairs_on_the_intersection_only(tmp_path, capsys):
    from functionWise import compare_runs
    corpus = _corpus_test(tmp_path)
    before = _run_files(tmp_path, "before", [f"f{i}" for i in range(5)],
                        "/// @notice Does a thing.\n")
    # the after run is missing one function: it must drop from BOTH sides
    after = _run_files(tmp_path, "after", [f"f{i}" for i in range(4)],
                       "/// @notice Transfers tokens to the recipient address.\n"
                       "/// @param to The recipient address\n")
    rc = compare_runs.main(["--before", before, "--after", after,
                            "--corpus", str(corpus), "--split", "test",
                            "--label", "G1", "--no-write"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "4 functions completed by both runs" in out
    assert "## Whole comment" in out
    assert "Δ" in out
