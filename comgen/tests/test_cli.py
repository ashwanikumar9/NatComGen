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
