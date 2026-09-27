"""The CLI's preflight and model resolution.

These exist because of a real failure: `python -m comgen run` was run in a
shell where $MODELS was empty, every slot went to the server as its own literal
name, and three functions came back as `HTTP Error 404: Not Found` — Ollama's
way of saying a model name is wrong, arriving once per function after
generating nothing. `run_pipeline.sh` had guarded against this since the
beginning; the CLI did not.
"""
from __future__ import annotations

import json

import pytest

from comgen import cli


HAVE = {"qwen2.5-coder:7b-instruct", "llama3.1:8b"}
BARE = {h.split(":", 1)[0] for h in HAVE}


# --------------------------------------------------------------------------
# resolving a model name
# --------------------------------------------------------------------------

def test_an_exact_tag_resolves():
    assert cli.resolves("qwen2.5-coder:7b-instruct", HAVE, BARE)


def test_a_bare_name_resolves_to_whatever_tag_the_server_has():
    assert cli.resolves("qwen2.5-coder", HAVE, BARE)


def test_a_wrong_tag_does_not_resolve_even_sharing_a_bare_name():
    """The bug this rule was written for. `qwen2.5-coder:7b` is not
    `qwen2.5-coder:7b-instruct`, and treating it as one is how a run spends
    hours on a model nobody chose."""
    assert not cli.resolves("qwen2.5-coder:7b", HAVE, BARE)


def test_an_unmapped_slot_never_resolves():
    assert not cli.resolves("GENERATOR_MODEL", HAVE, BARE)


# --------------------------------------------------------------------------
# the preflight
# --------------------------------------------------------------------------

def fake_tags(monkeypatch, names, *, fail=None):
    import urllib.request

    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"models": [{"name": n} for n in names]}).encode()

    def urlopen(url, timeout=0):
        if fail:
            raise OSError(fail)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_preflight_passes_when_every_slot_maps(monkeypatch, capsys):
    fake_tags(monkeypatch, HAVE)
    cli.preflight("http://localhost:11434", {
        s: "qwen2.5-coder:7b-instruct" for s in
        ("INTENT_REASONER_MODEL", "GENERATOR_MODEL", "SEMANTIC_CRITIC_MODEL",
         "VERIFIER_MODEL")})
    assert "GENERATOR_MODEL=qwen2.5-coder:7b-instruct" in capsys.readouterr().out


def test_preflight_names_the_unmapped_slots_and_says_why(monkeypatch):
    fake_tags(monkeypatch, HAVE)
    with pytest.raises(SystemExit) as e:
        cli.preflight("http://localhost:11434", {})
    msg = str(e.value)
    for slot in ("INTENT_REASONER_MODEL", "GENERATOR_MODEL",
                 "SEMANTIC_CRITIC_MODEL", "VERIFIER_MODEL"):
        assert slot in msg
    assert "qwen2.5-coder:7b-instruct" in msg, "say what the server does have"
    assert "MODELS was probably empty" in msg, "name the actual cause"


def test_preflight_reports_an_unreachable_server_as_that(monkeypatch):
    fake_tags(monkeypatch, HAVE, fail="connection refused")
    with pytest.raises(SystemExit) as e:
        cli.preflight("http://localhost:11434", {})
    assert "no model server" in str(e.value)
    assert "start ollama" in str(e.value)


def test_preflight_covers_every_slot_the_prompts_name():
    """If a new prompt introduces a model slot, the preflight must check it —
    otherwise the first run to use that prompt 404s hours in."""
    from comgen import prompts as P
    assert {p.model for p in P.PROMPTS} == {
        "INTENT_REASONER_MODEL", "GENERATOR_MODEL", "SEMANTIC_CRITIC_MODEL",
        "VERIFIER_MODEL"}, "ComGen introduced a model slot; update preflight"


# --------------------------------------------------------------------------
# where the mapping comes from
# --------------------------------------------------------------------------

class Args:
    def __init__(self, models=None):
        self.models = models


def test_the_flag_wins(monkeypatch):
    monkeypatch.setenv("MODELS", '{"GENERATOR_MODEL":"from-env"}')
    assert cli._models(Args('{"GENERATOR_MODEL":"from-flag"}')) == {
        "GENERATOR_MODEL": "from-flag"}


def test_the_environment_is_next(monkeypatch):
    monkeypatch.setenv("MODELS", '{"GENERATOR_MODEL":"from-env"}')
    assert cli._models(Args()) == {"GENERATOR_MODEL": "from-env"}


def test_an_empty_environment_falls_through_to_models_json(monkeypatch, tmp_path):
    """The exact shape of the failure: MODELS set but empty, which json.loads
    cannot read and which must not be mistaken for a mapping."""
    monkeypatch.setenv("MODELS", "")
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    (tmp_path / "models.json").write_text('{"GENERATOR_MODEL":"from-file"}')
    assert cli._models(Args()) == {"GENERATOR_MODEL": "from-file"}


def test_no_mapping_anywhere_is_empty_not_an_error(monkeypatch, tmp_path):
    monkeypatch.delenv("MODELS", raising=False)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    assert cli._models(Args()) == {}


def test_a_malformed_models_json_says_which_file(monkeypatch, tmp_path):
    monkeypatch.delenv("MODELS", raising=False)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    (tmp_path / "models.json").write_text("{not json")
    with pytest.raises(SystemExit) as e:
        cli._models(Args())
    assert "models.json" in str(e.value)


# --------------------------------------------------------------------------
# resuming past an error record
# --------------------------------------------------------------------------

def a_run_file(tmp_path, *records):
    p = tmp_path / "G1" / "seed0" / "val.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in records),
                 encoding="utf-8")
    return p


def test_error_records_are_dropped_so_they_are_re_attempted(tmp_path):
    """The trap: `run_config` resumes by skipping every pair_id already in the
    file. A run that 404s on every function leaves a complete-looking file of
    failures, and the next run does nothing and says `0 hits, 0 misses`."""
    p = a_run_file(tmp_path,
                   {"pair_id": "a", "final": "/// @notice ok"},
                   {"pair_id": "b", "error": "ollama: HTTP Error 404"},
                   {"pair_id": "c", "error": "ollama: HTTP Error 404"})
    assert cli.drop_errors(tmp_path, "val") == 2
    kept = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert [r["pair_id"] for r in kept] == ["a"], \
        "the comment survives, the failures do not"


def test_dropping_errors_leaves_a_clean_file_untouched(tmp_path):
    p = a_run_file(tmp_path, {"pair_id": "a", "final": "/// @notice ok"})
    before = p.read_text()
    assert cli.drop_errors(tmp_path, "val") == 0
    assert p.read_text() == before


def test_a_file_of_nothing_but_errors_empties_rather_than_breaking(tmp_path):
    p = a_run_file(tmp_path, {"pair_id": "a", "error": "x"},
                   {"pair_id": "b", "error": "y"})
    assert cli.drop_errors(tmp_path, "val") == 2
    assert p.read_text() == ""
    assert cli.count_errors(tmp_path, "val") == 0


def test_counting_errors_does_not_change_the_file(tmp_path):
    p = a_run_file(tmp_path, {"pair_id": "a", "error": "x"},
                   {"pair_id": "b", "final": "ok"})
    before = p.read_text()
    assert cli.count_errors(tmp_path, "val") == 1
    assert p.read_text() == before


def test_another_split_is_left_alone(tmp_path):
    a_run_file(tmp_path, {"pair_id": "a", "error": "x"})
    other = tmp_path / "G1" / "seed0" / "test.jsonl"
    other.write_text(json.dumps({"pair_id": "z", "error": "x"}) + "\n")
    assert cli.drop_errors(tmp_path, "val") == 1
    assert other.read_text().strip(), "the test split was not asked about"
