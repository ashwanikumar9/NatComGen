"""Stage 4: reproducibility.

The point of these tests is that the manifest must *notice*. A hash record
that fails to change when the corpus or a prompt changes is worse than no
record at all, because it certifies something untrue.
"""
import json
from pathlib import Path

import pytest

from natspec_corpus.reproduce import (ReproduceError, artifact_hashes, check,
                                      environment, manifest, prompt_hashes,
                                      run, sha1_file, sha1_tree, verify)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "NatSpecGold"
    (root / "contracts" / "p").mkdir(parents=True)
    (root / "sigma").mkdir()
    (root / "contracts" / "p" / "A.sol").write_text("contract A {}")
    (root / "contracts" / "p" / "B.sol").write_text("contract B {}")
    (root / "pairs.jsonl").write_text('{"id": "1"}\n')
    (root / "splits.json").write_text('{"train": []}')
    (root / "index_allowlist.json").write_text('{"train": []}')
    (root / "sigma" / "sigma.jsonl").write_text('{"pair_id": "1"}\n')
    return root


# -- hashing ---------------------------------------------------------------

def test_a_file_hash_changes_with_its_contents(tmp_path):
    p = tmp_path / "f"
    p.write_text("a")
    first = sha1_file(p)
    p.write_text("b")
    assert sha1_file(p) != first


def test_a_tree_hash_covers_every_file(corpus):
    first = sha1_tree(corpus / "contracts", "**/*.sol")
    (corpus / "contracts" / "p" / "B.sol").write_text("contract B { }")
    assert sha1_tree(corpus / "contracts", "**/*.sol") != first


def test_a_tree_hash_changes_when_a_file_is_renamed(corpus):
    """Paths are part of the hash, so a rename is a different tree."""
    first = sha1_tree(corpus / "contracts", "**/*.sol")
    (corpus / "contracts" / "p" / "B.sol").rename(
        corpus / "contracts" / "p" / "C.sol")
    assert sha1_tree(corpus / "contracts", "**/*.sol") != first


def test_a_tree_hash_does_not_depend_on_filesystem_order(corpus):
    a = sha1_tree(corpus / "contracts", "**/*.sol")
    (corpus / "contracts" / "p" / "A.sol").touch()
    assert sha1_tree(corpus / "contracts", "**/*.sol") == a


# -- what the manifest records --------------------------------------------

def test_artifact_hashes_cover_every_stage_output(corpus):
    h = artifact_hashes(corpus)
    assert set(h) >= {"pairs.jsonl", "splits.json", "index_allowlist.json",
                      "contracts/", "sigma/sigma.jsonl"}


def test_a_missing_artifact_is_simply_absent(tmp_path):
    assert artifact_hashes(tmp_path) == {}


def test_prompt_hashes_cover_the_whole_pipeline():
    h = prompt_hashes()
    assert set(h) >= {"L7", "L1b", "L2", "L3", "L8", "retrieval_config"}


def test_a_changed_prompt_changes_its_hash(monkeypatch):
    """A prompt edit must invalidate every result derived from it."""
    import natspec_corpus.prompts_v3 as P
    before = prompt_hashes()["L2"]
    patched = P.Prompt(id=P.SEMANTIC_CRITIC.id, agent=P.SEMANTIC_CRITIC.agent,
                       model=P.SEMANTIC_CRITIC.model,
                       temp=P.SEMANTIC_CRITIC.temp,
                       system=P.SEMANTIC_CRITIC.system + " extra sentence.",
                       user=P.SEMANTIC_CRITIC.user,
                       schema=P.SEMANTIC_CRITIC.schema,
                       inputs=P.SEMANTIC_CRITIC.inputs)
    monkeypatch.setattr(P, "PIPELINE", tuple(
        patched if p.id == "L2" else p for p in P.PIPELINE))
    assert prompt_hashes()["L2"] != before


def test_the_environment_records_what_could_change_a_number():
    e = environment()
    assert set(e) >= {"python", "platform", "solc_versions", "slither",
                      "faiss"}


def test_a_manifest_carries_the_run_configuration(corpus):
    m = manifest(corpus, models={"GENERATOR_MODEL": "qwen"}, seeds=(0, 1),
                 configs=("C1", "C2"), split="test", encoder="jina")
    assert m["models"]["GENERATOR_MODEL"] == "qwen"
    assert m["seeds"] == [0, 1] and m["configs"] == ["C1", "C2"]
    assert m["split"] == "test" and m["encoder"] == "jina"


# -- checking --------------------------------------------------------------

def test_an_unchanged_corpus_reports_no_differences(corpus):
    assert check(corpus, manifest(corpus)) == []


def test_a_changed_contract_is_reported(corpus):
    m = manifest(corpus)
    (corpus / "contracts" / "p" / "A.sol").write_text("contract A { uint x; }")
    problems = check(corpus, m)
    assert any("contracts/" in p for p in problems)


def test_a_changed_pairs_file_is_reported(corpus):
    m = manifest(corpus)
    (corpus / "pairs.jsonl").write_text('{"id": "2"}\n')
    assert any("pairs.jsonl" in p for p in problems_of(corpus, m))


def problems_of(corpus, m):
    return check(corpus, m)


def test_a_deleted_artifact_is_reported_as_missing(corpus):
    m = manifest(corpus)
    (corpus / "splits.json").unlink()
    assert any("splits.json: missing now" in p for p in check(corpus, m))


def test_a_new_artifact_is_reported_too(corpus):
    m = manifest(corpus)
    (corpus / "pairs_partial.jsonl").write_text("{}\n")
    assert any("not in the manifest" in p for p in check(corpus, m))


def test_verify_raises_and_names_the_stage(corpus, tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(manifest(corpus)))
    (corpus / "pairs.jsonl").write_text('{"id": "changed"}\n')
    with pytest.raises(ReproduceError, match="pairs.jsonl"):
        verify(corpus, p)


def test_verify_passes_on_an_untouched_corpus(corpus, tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(manifest(corpus)))
    verify(corpus, p)


# -- the pipeline ----------------------------------------------------------

@pytest.mark.skipif(
    not __import__("natspec_corpus.compile", fromlist=["x"]).installed_versions(),
    reason="no solc")
def test_a_rebuild_reproduces_its_own_manifest(tmp_path):
    """Stage 1 is deterministic; this is the test that says so end to end."""
    src = tmp_path / "src" / "Trail_of_Bits-UniswapV3Core" / "v3" / "contracts"
    src.mkdir(parents=True)
    src.joinpath("M.sol").write_text(
        "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n"
        "library M {\n    /// @notice Doubles the given value.\n"
        "    /// @param x the value to double\n"
        "    /// @return the value times two\n"
        "    function double(uint256 x) internal pure returns (uint256) "
        "{ return x * 2; }\n}\n")
    out = tmp_path / "corpus"

    first = run(tmp_path / "src", out, with_sigma=False, progress=False)
    assert first.manifest["artifacts"]["pairs.jsonl"]

    second = run(tmp_path / "src", out, with_sigma=False,
                 expect=first.manifest, progress=False)
    assert second.manifest["reproduction"]["matches"]


@pytest.mark.skipif(
    not __import__("natspec_corpus.compile", fromlist=["x"]).installed_versions(),
    reason="no solc")
def test_a_changed_source_fails_the_reproduction(tmp_path):
    src = tmp_path / "src" / "Trail_of_Bits-UniswapV3Core" / "v3" / "contracts"
    src.mkdir(parents=True)
    f = src / "M.sol"
    # The notice must be three words or more, or the scorer labels it
    # `text_short`, the pair lands in pairs_partial.jsonl and the corpus has
    # no verified pairs at all — which the build correctly refuses to ship.
    f.write_text("// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n"
                 "library M {\n    /// @notice Doubles the given value.\n"
                 "    /// @param x the value to double\n"
                 "    /// @return the value times two\n"
                 "    function double(uint256 x) internal pure returns "
                 "(uint256) { return x * 2; }\n}\n")
    out = tmp_path / "corpus"
    first = run(tmp_path / "src", out, with_sigma=False, progress=False)

    f.write_text(f.read_text().replace("Doubles the given value.",
                                       "Doubles the supplied value."))
    with pytest.raises(ReproduceError, match="does not reproduce"):
        run(tmp_path / "src", out, with_sigma=False, expect=first.manifest,
            progress=False)
