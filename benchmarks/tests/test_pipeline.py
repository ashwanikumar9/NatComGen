"""Indexing, matching, annotating and joining — on fixtures small enough to
read, and on the two encodings that broke it.
"""
import json
import sqlite3
from pathlib import Path

import pytest

from benchmarks.smartdoc import build_corpus as BC
from benchmarks.smartdoc import index as IX
from benchmarks.smartdoc import reground as RG
from benchmarks.smartdoc import score as SC
from benchmarks.smartdoc import tokens as T

CONTRACT = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Vault {
    mapping(address => uint256) public balance;

    /**
     * @notice The author's own sentence, which must not survive.
     * @param amount how much
     */
    function deposit(uint256 amount) public {
        require(amount > 0, "zero deposit");
        balance[msg.sender] += amount;
    }

    /// @notice Another author sentence.
    /// @param to where
    function send(address to, uint256 amount) public {
        balance[to] += amount;
    }

    function untouched() public view returns (uint256) {
        return balance[msg.sender];
    }
}
"""


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "Vault.sol").write_text(CONTRACT, encoding="utf-8")
    db = tmp_path / "i.db"
    IX.build(root, db)
    return root, db


def test_index_finds_every_function(corpus):
    _, db = corpus
    con = sqlite3.connect(str(db))
    names = {r[0] for r in con.execute("SELECT name FROM funcs")}
    assert names == {"deposit", "send", "untouched"}


def test_index_records_the_signature_not_just_the_arity(corpus):
    _, db = corpus
    con = sqlite3.connect(str(db))
    sigs = {r[0] for r in con.execute("SELECT sig FROM funcs")}
    assert "send(address,uint256)" in sigs


def test_index_is_resumable(corpus):
    root, db = corpus
    con = sqlite3.connect(str(db))
    before = con.execute("SELECT COUNT(*) FROM funcs").fetchone()[0]
    con.close()
    IX.build(root, db)                       # same content: nothing re-added
    con = sqlite3.connect(str(db))
    assert con.execute("SELECT COUNT(*) FROM funcs").fetchone()[0] == before


def test_read_source_does_not_translate_crlf(tmp_path):
    """`Path.read_text` shortens a CRLF file, and every offset then points
    two characters past where it should."""
    p = tmp_path / "a.sol"
    p.write_bytes(b"a\r\nb\r\n")
    assert IX.read_source(p) == "a\r\nb\r\n"
    assert p.read_text() != IX.read_source(p)


def test_offsets_survive_crlf(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "V.sol").write_bytes(CONTRACT.replace("\n", "\r\n").encode())
    db = tmp_path / "i.db"
    IX.build(root, db)
    con = sqlite3.connect(str(db))
    start, end, path = con.execute(
        "SELECT f.start, f.end, fl.path FROM funcs f "
        "JOIN files fl ON fl.id = f.file_id WHERE f.name='deposit'").fetchone()
    body = RG._slice(root, path, start, end)
    assert body.startswith("function deposit")
    assert body.rstrip().endswith("}")


def _fixture_set(tmp_path, root, db, names, refs):
    """A SmartDoc-shaped directory holding the named functions."""
    con = sqlite3.connect(str(db))
    codes = []
    for n in names:
        s, e, p = con.execute(
            "SELECT f.start, f.end, fl.path FROM funcs f "
            "JOIN files fl ON fl.id=f.file_id WHERE f.name=?", (n,)).fetchone()
        codes.append(" ".join(T.tokenise(RG._slice(root, p, s, e))))
    d = tmp_path / "sd"
    (d / "dataset/test").mkdir(parents=True)
    (d / "dataset/test/test.token.code").write_text("\n".join(codes) + "\n")
    (d / "ref.txt").write_text("\n".join(refs))
    return d


def test_reground_matches_exactly(corpus, tmp_path):
    root, db = corpus
    data = _fixture_set(tmp_path, root, db, ["deposit", "send"],
                        ["ref one", "ref two"])
    out = tmp_path / "matched.jsonl"
    stats = RG.reground(db, data, out)
    assert stats["exact"] == 2 and stats["coverage"] == 1.0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert {r["name"] for r in rows} == {"deposit", "send"}
    assert rows[0]["sig"].startswith("deposit(")


def test_annotate_replaces_a_block_comment(corpus, tmp_path):
    root, db = corpus
    data = _fixture_set(tmp_path, root, db, ["deposit"], ["the real reference"])
    out = tmp_path / "m.jsonl"
    RG.reground(db, data, out)
    item = json.loads(out.read_text().splitlines()[0])
    result = BC.annotate(IX.read_source(root / item["file"]), [item])
    assert "/// @notice the real reference" in result
    assert "The author's own sentence" not in result
    assert "Another author sentence" in result          # untouched neighbour


def test_annotate_replaces_a_line_comment_run(corpus, tmp_path):
    root, db = corpus
    data = _fixture_set(tmp_path, root, db, ["send"], ["replacement"])
    out = tmp_path / "m.jsonl"
    RG.reground(db, data, out)
    item = json.loads(out.read_text().splitlines()[0])
    result = BC.annotate(IX.read_source(root / item["file"]), [item])
    assert "/// @notice replacement" in result
    assert "Another author sentence" not in result
    assert "@param to where" not in result


def test_annotate_replaces_under_crlf(tmp_path):
    """The `doc_line` span runs past the last visible character on a CRLF
    file; an equality test on its end finds nothing and leaves the author's
    comment in place, which is invisible in every count."""
    root = tmp_path / "src"
    root.mkdir()
    (root / "V.sol").write_bytes(CONTRACT.replace("\n", "\r\n").encode())
    db = tmp_path / "i.db"
    IX.build(root, db)
    data = _fixture_set(tmp_path, root, db, ["send"], ["replacement"])
    out = tmp_path / "m.jsonl"
    RG.reground(db, data, out)
    item = json.loads(out.read_text().splitlines()[0])
    result = BC.annotate(IX.read_source(root / item["file"]), [item])
    assert "Another author sentence" not in result


def test_annotate_is_bottom_up_over_several_functions(corpus, tmp_path):
    root, db = corpus
    data = _fixture_set(tmp_path, root, db, ["deposit", "send"], ["one", "two"])
    out = tmp_path / "m.jsonl"
    RG.reground(db, data, out)
    items = [json.loads(l) for l in out.read_text().splitlines()]
    result = BC.annotate(IX.read_source(root / items[0]["file"]), items)
    assert "/// @notice one" in result and "/// @notice two" in result
    assert result.index("@notice one") < result.index("@notice two")


def test_register_mutates_the_split_sets_in_place():
    """`build.py` imported these by name, so rebinding would not reach it."""
    from natspec_corpus import projects
    before_val = set(projects.VAL_PROJECTS)
    val_obj = projects.VAL_PROJECTS
    try:
        BC.register(["smartdoc-test-00000"], split="val")
        assert projects.VAL_PROJECTS is val_obj
        assert "smartdoc-test-00000" in projects.VAL_PROJECTS
        assert projects.slug_for("smartdoc-test-00000") == "smartdoc-test-00000"
        BC.register(["smartdoc-test-00000"], split="train")
        assert "smartdoc-test-00000" not in projects.VAL_PROJECTS
    finally:
        projects.VAL_PROJECTS.clear()
        projects.VAL_PROJECTS.update(before_val)
        projects.SLUGS[:] = [t for t in projects.SLUGS
                             if not t[1].startswith("smartdoc-")]


def test_fixed_width_project_names_do_not_collide():
    """`slug_for` matches by substring: `smartdoc-test-1` would also match
    `smartdoc-test-12`, folding two contracts into one project."""
    from natspec_corpus import projects
    try:
        BC.register(["smartdoc-test-00001", "smartdoc-test-00012"],
                    split="val")
        assert projects.slug_for("smartdoc-test-00012") == "smartdoc-test-00012"
    finally:
        projects.VAL_PROJECTS.difference_update(
            {"smartdoc-test-00001", "smartdoc-test-00012"})
        projects.SLUGS[:] = [t for t in projects.SLUGS
                             if not t[1].startswith("smartdoc-")]


def test_nl_tokenise_matches_their_reference_style():
    assert SC.nl_tokenise("Checks the EIN, then returns `x`.") == \
        "Checks the EIN , then returns ` x ` ."


def test_notice_of_takes_only_the_notice():
    text = ("/// @notice Transfers the balance.\n"
            "/// @dev not for external use\n"
            "/// @param to recipient\n")
    assert SC.notice_of(text) == "Transfers the balance."


def test_notice_of_reads_untagged_prose():
    assert SC.notice_of("/// Transfers the balance.") == "Transfers the balance."


def test_notice_of_empty():
    assert SC.notice_of("") == ""


def test_score_subset_scores_only_what_was_produced():
    refs = ["alpha beta gamma delta", "one two three four"]
    assert SC.score_subset(refs, {0: "alpha beta gamma delta"})["n"] == 1


# --------------------------------------------------------------------------
# the whole chain, on three functions
# --------------------------------------------------------------------------

def test_end_to_end_stage_build_finalise_join(corpus, tmp_path):
    """Index, match, annotate, build, finalise, join — and the reference must
    come back out of `pairs.jsonl` byte for byte.

    This is the check that catches the failure no count reveals: the
    reference travels through a `///` line, the lexer, the extractor and the
    scorer, and if any of them alters it the benchmark scores NatComGen
    against the contract author's sentence while every total still looks
    right.
    """
    from natspec_corpus import projects

    root, db = corpus
    refs = ["first reference here", "second reference here"]
    data = _fixture_set(tmp_path, root, db, ["deposit", "send"], refs)
    matched = tmp_path / "m.jsonl"
    RG.reground(db, data, matched)

    out_src = tmp_path / "staged"
    meta = BC.stage(matched, root, out_src, split="test")
    assert meta["functions"] == 2
    assert all(p.startswith("smartdoc-test-") for p in meta["projects"])

    saved = (list(projects.SLUGS), set(projects.VAL_PROJECTS),
             set(projects.TEST_PROJECTS))
    try:
        out_corpus = tmp_path / "built"
        BC.build(out_src, out_corpus, meta["projects"], split="val")
        fin = BC.finalise(out_corpus, out_src.parent / "corpus_map_test.json",
                          split="val")
    finally:
        projects.SLUGS[:] = saved[0]
        projects.VAL_PROJECTS.clear(); projects.VAL_PROJECTS.update(saved[1])
        projects.TEST_PROJECTS.clear(); projects.TEST_PROJECTS.update(saved[2])

    assert fin["scored"] == 2

    # A one-sentence notice is "partial" to the corpus builder, which is the
    # right verdict about the comment and the wrong reason to drop the
    # function: SmartDoc's gold is notice-only by construction.
    assert fin["promoted_from_partial"] >= 1

    splits = json.loads((out_corpus / "splits.json").read_text())
    assert len(splits["val"]) == 2
    # An undocumented function never becomes a pair at all, so `untouched`
    # is simply absent. What matters is that nothing lands in `train`: a
    # non-matched pair from these same files would be retrievable as an
    # exemplar for the very functions being evaluated.
    assert splits["train"] == []
    scored_names = {json.loads(l)["name"]
                    for l in (out_corpus / "pairs.jsonl").read_text().splitlines()
                    if json.loads(l)["split"] == "val"}
    assert scored_names == {"deposit", "send"}

    pair_to_index = SC.load_map(out_corpus, out_src.parent / "corpus_map_test.json")
    assert len(pair_to_index) == 2
    seen = {}
    for line in (out_corpus / "pairs.jsonl").read_text().splitlines():
        p = json.loads(line)
        i = pair_to_index.get(p["id"])
        if i is not None:
            seen[i] = " ".join((p.get("notice") or "").split())
    assert seen == {0: refs[0], 1: refs[1]}


# --------------------------------------------------------------------------
# the driver's checkpoints
# --------------------------------------------------------------------------

import subprocess

BENCH = Path(__file__).resolve().parents[1] / "run_benchmark.sh"


def _fingerprint(stage: str, **env) -> str:
    """Ask the script itself what it would stamp for a stage."""
    script = (f'set -a; SELFTEST=0; CORPUS_SOURCE=self; CORPUS_FROM=""; '
              f'CORPUS_LIMIT=""; SPLIT=test; CONFIGS=C5; SEEDS=0; LIMIT=""; '
              f'MODELS=""; OLLAMA=""; '
              + "".join(f'{k}={v!r}; ' for k, v in env.items())
              + f'source <(sed -n "/^fingerprint()/,/^}}/p" {BENCH}); '
              f'fingerprint {stage}')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return out.stdout.strip()


def test_the_selftest_and_a_real_corpus_stamp_different_checkpoints():
    """The bug this guards against handed back a fixture score as SmartDoc.

    The self-test leaves a full set of `state/*.done` markers. If those
    satisfy a real `--corpus-source disl` invocation, every stage reports
    "done, skipping", nothing runs, and `smartdoc.md` still holds the
    200-function fixture — labelled as the benchmark.
    """
    for stage in ("corpus", "index", "reground", "build", "score"):
        a = _fingerprint(stage, SELFTEST=1)
        b = _fingerprint(stage, SELFTEST=0, CORPUS_SOURCE="disl")
        assert a and b and a != b, f"{stage}: {a!r} == {b!r}"


def test_changing_the_corpus_source_invalidates_the_index():
    assert (_fingerprint("index", CORPUS_SOURCE="disl")
            != _fingerprint("index", CORPUS_SOURCE="sanctuary"))


def test_changing_the_corpus_size_invalidates_the_index():
    assert (_fingerprint("index", CORPUS_LIMIT="50000")
            != _fingerprint("index", CORPUS_LIMIT="500000"))


def test_the_model_is_part_of_the_run_checkpoint():
    """Swap the model and the previous run's comments are not yours."""
    assert (_fingerprint("run", MODELS='{"G":"a"}')
            != _fingerprint("run", MODELS='{"G":"b"}'))


def test_fetching_the_release_does_not_depend_on_the_corpus():
    """SmartDoc's own files never change with the corpus you match against,
    so re-pointing the corpus must not re-download them."""
    assert (_fingerprint("fetch", CORPUS_SOURCE="disl")
            == _fingerprint("fetch", CORPUS_SOURCE="self"))
