"""Tests for the shipped bundle: the runner script and the offline stub.

`run_pipeline.sh` is the part of this project nobody unit-tests and everybody
depends on. Its failure mode is not a crash — it is a stage that reports
"done" while having produced nothing, or a checkpoint that skips work after an
input changed. Both are silent, and both are checked here.

The script's own shell functions are extracted and run directly rather than
re-implemented, so these tests fail when the script changes and the test does
not.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "run_pipeline.sh"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(),
                                reason="run_pipeline.sh not in this tree")


def _run(*args, **kw):
    """Invoke the script through `bash` rather than by path.

    The executable bit does not survive a zip round trip through Windows, nor
    a clone on a filesystem without one. Depending on it here made eleven
    tests fail with PermissionError on the GPU box — and since the `tests`
    stage gates the pipeline, that took the whole run down over a file mode.
    """
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True,
                          text=True, cwd=ROOT, **kw)


def _fn(name: str, call: str, env=None) -> str:
    """Run one of the script's functions in isolation."""
    body = subprocess.run(
        ["sed", "-n", f"/^{name}()/,/^}}/p", str(SCRIPT)],
        capture_output=True, text=True, check=True).stdout
    assert body.strip(), f"{name}() not found in run_pipeline.sh"
    out = subprocess.run(["bash", "-c", body + "\n" + call],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


# -- the script parses and documents itself --------------------------------

def test_the_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_every_stage_in_the_list_has_a_function():
    text = SCRIPT.read_text()
    line = next(l for l in text.splitlines() if l.startswith("STAGES=("))
    stages = line.split("(", 1)[1].split(")")[0].split()
    for s in stages:
        assert f"stage_{s}()" in text, f"stage {s} has no stage_{s}()"


def test_every_stage_has_an_output_to_check_for():
    """A stage with no declared output can never be caught having produced
    nothing, which is the failure this guard exists for."""
    text = SCRIPT.read_text()
    line = next(l for l in text.splitlines() if l.startswith("STAGES=("))
    stages = line.split("(", 1)[1].split(")")[0].split()
    declared = subprocess.run(["sed", "-n", "/^outputs()/,/^}/p", str(SCRIPT)],
                              capture_output=True, text=True).stdout
    for s in stages:
        if s == "tests":            # its output is its own exit status
            continue
        assert f"{s})" in declared, f"stage {s} declares no output"


def test_every_documented_option_is_accepted():
    """The header is the only documentation anyone reads."""
    head = SCRIPT.read_text().split("set -euo")[0]
    opts = {w.strip("'\",") for w in head.split() if w.startswith("--")}
    body = SCRIPT.read_text()
    for opt in opts:
        assert f"{opt})" in body, f"{opt} is documented but not parsed"


def test_no_test_here_invokes_the_script_by_path():
    """Invoking by path needs the executable bit, which a zip through Windows
    and some clones do not carry. Everything here goes through `bash`."""
    src = Path(__file__).read_text()
    needle = "subprocess.run(" + "[str(SCRIPT)"   # split, or this test is a hit
    assert needle not in src, \
        "invoke the script with _run(), which calls it through bash"


def test_an_unknown_option_is_rejected_rather_than_ignored():
    out = _run("--nonsense")
    assert out.returncode == 2 and "unknown option" in out.stderr


# -- fingerprints survive the tree moving ----------------------------------
# The whole point of the checkpoints is that unzipping this on the GPU box and
# rerunning does not redo nine minutes of work. `sha1sum somefile` prints the
# path next to the digest, so hashing that way makes every checkpoint stale
# the moment the tree moves — which is precisely when it must not.

def _tree(root: Path) -> Path:
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "x.sol").write_text("contract C {}")
    (root / "sub" / "y.jsonl").write_text('{"a": 1}\n')
    (root / "f.py").write_text("print(1)\n")
    return root


def test_a_file_hash_is_the_same_from_two_different_paths(tmp_path):
    a, b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    ha = _fn("_hash_files", f"_hash_files {a}/f.py")
    hb = _fn("_hash_files", f"_hash_files {b}/f.py")
    assert ha and ha == hb


def test_a_tree_hash_is_the_same_from_two_different_paths(tmp_path):
    a, b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    assert _fn("_hash_tree", f"_hash_tree {a}") == _fn("_hash_tree",
                                                       f"_hash_tree {b}")


def test_changed_content_still_changes_the_hash(tmp_path):
    """Path independence must not cost the detection it exists to serve."""
    a, b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    (b / "f.py").write_text("print(2)\n")
    assert _fn("_hash_files", f"_hash_files {a}/f.py") != \
        _fn("_hash_files", f"_hash_files {b}/f.py")


def test_a_renamed_file_inside_a_tree_still_changes_the_hash(tmp_path):
    a, b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    (b / "sub" / "x.sol").rename(b / "sub" / "renamed.sol")
    assert _fn("_hash_tree", f"_hash_tree {a}") != _fn("_hash_tree",
                                                       f"_hash_tree {b}")


def test_an_added_file_changes_the_tree_hash(tmp_path):
    a, b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    (b / "sub" / "extra.sol").write_text("contract D {}")
    assert _fn("_hash_tree", f"_hash_tree {a}") != _fn("_hash_tree",
                                                       f"_hash_tree {b}")


def test_the_tree_hash_pins_the_collation_locale():
    """`sort` collates by locale and the digest is order-dependent: under C,
    uppercase sorts first so `openzeppelin-*` comes last; under en_US.UTF-8
    case is folded and it comes first. Same files, different order, different
    hash — which surfaced as `corpus STALE` on a box whose only difference
    from the build machine was $LANG.

    Checked at the source, because a machine with one locale installed cannot
    demonstrate the disagreement no matter how the test is written."""
    body = subprocess.run(["sed", "-n", "/^_hash_tree()/,/^}/p", str(SCRIPT)],
                          capture_output=True, text=True).stdout
    assert "sort" in body, "the tree hash no longer sorts"
    assert "LC_ALL=C sort" in body, \
        "sort in _hash_tree must be locale-pinned or the hash is not portable"


def test_the_tree_hash_is_the_same_under_every_locale_available_here(tmp_path):
    """Weaker than the check above wherever only C locales exist, but it is
    the one that would catch a regression on a machine that has more."""
    root = _tree(tmp_path / "a")
    for extra in ("Zebra.sol", "apple.sol", "Mango.sol", "_under.sol"):
        (root / "sub" / extra).write_text(f"contract {extra[:3]} {{}}")
    locales = subprocess.run(["locale", "-a"], capture_output=True,
                             text=True).stdout.split() or ["C"]
    hashes = set()
    for loc in locales[:8]:
        import os
        env = dict(os.environ, LC_ALL=loc, LANG=loc)
        hashes.add(_fn("_hash_tree", f"_hash_tree {root}", env=env))
    assert len(hashes) == 1, f"the hash varies by locale: {hashes}"


def test_hashing_a_missing_file_is_not_an_error(tmp_path):
    """Stages fingerprint artifacts that do not exist yet; a non-zero exit
    under `set -e` would take the run down with it."""
    assert _fn("_hash_files", f"_hash_files {tmp_path}/absent.json") == ""
    assert _fn("_hash_tree", f"_hash_tree {tmp_path}/absent") == ""


# -- the outputs the checkpoints look for are the ones the code writes -----

def test_the_report_outputs_are_what_write_report_actually_writes(tmp_path):
    """The bug this catches: `outputs report` named a file the report stage
    has never written, so the stage re-ran on every invocation, forever.

    Checked by running the real writer and looking on disk, because the point
    is agreement between the script and the code, and a test that reads the
    script alone cannot see a disagreement."""
    from natspec_corpus.report import write_report
    write_report(tmp_path, results={"C1": {"n": 1}}, labels={"C1": "full"},
                 manifest={"artifacts": {}}, figure=False)
    declared = _fn("outputs",
                   f'RESULTS={tmp_path} CORPUS=/C outputs report').split()
    assert declared, "the report stage declares no output"
    for path in declared:
        assert Path(path).exists(), f"report never writes {path}"


def test_the_emit_output_is_what_the_emit_stage_writes():
    declared = _fn("outputs", 'RESULTS=/R CORPUS=/C outputs emit').split()
    assert declared == ["/R/emission_report.json"]
    assert "emission_report.json" in SCRIPT.read_text()


def test_corpus_outputs_match_the_builders_own_artifact_list():
    from natspec_corpus.reproduce import artifact_hashes
    import inspect
    declared = {Path(d).name for d in
                _fn("outputs", 'RESULTS=/R CORPUS=/C outputs corpus').split()}
    src = inspect.getsource(artifact_hashes)
    for name in declared - {"contracts"}:
        assert name in src, f"{name} is not a corpus artifact"


# -- the offline stub -------------------------------------------------------

def test_the_stub_answers_every_prompt_in_the_pipeline_with_valid_json():
    """If the stub cannot satisfy a prompt's schema, the offline wiring test
    it exists for silently stops covering that prompt."""
    sys.path.insert(0, str(ROOT / "tools"))
    import stub_ollama

    from natspec_corpus.llm import missing_fields
    from natspec_corpus.prompts_v3 import PIPELINE
    for p in PIPELINE:
        data = stub_ollama.reply(p.id, "function transfer(address to) F1 N2")
        assert data, f"the stub has no reply for {p.id}"
        assert missing_fields(data, p.schema) == [], \
            f"{p.id}: stub reply is missing {missing_fields(data, p.schema)}"


def test_the_stub_quotes_row_ids_it_was_given():
    """Claims have to cite something that exists, or the gate rejects every
    one of them and the wiring test passes for the wrong reason."""
    sys.path.insert(0, str(ROOT / "tools"))
    import stub_ollama
    data = stub_ollama.reply("L1b", "rows: F3 N9 C2 and prose")
    assert data["claims"][0]["ids"] == ["F3", "N9"]


def test_the_stub_falls_back_when_there_are_no_rows_to_cite():
    sys.path.insert(0, str(ROOT / "tools"))
    import stub_ollama
    assert stub_ollama.reply("L1b", "nothing here")["claims"][0]["ids"] == ["F1"]


# -- requirements -----------------------------------------------------------

def test_every_import_the_pipeline_needs_is_in_requirements():
    req = (ROOT / "requirements.txt").read_text().lower()
    for pkg in ("slither-analyzer", "numpy", "faiss", "scipy", "matplotlib",
                "pytest"):
        assert pkg in req, f"{pkg} is used but not required"


def test_setup_env_reports_a_machine_that_cannot_run_the_pipeline():
    """--check must exit non-zero when something is missing, or a broken box
    looks ready."""
    src = (ROOT / "setup_env.py").read_text()
    assert 'report["problems"]' in src and "return 1" in src


# -- the script runs ---------------------------------------------------------

def test_status_names_every_stage_and_exits_cleanly():
    out = _run("--status")
    assert out.returncode == 0, out.stderr
    line = SCRIPT.read_text().splitlines()
    stages = next(l for l in line if l.startswith("STAGES=(")
                  ).split("(", 1)[1].split(")")[0].split()
    for s in stages:
        assert s in out.stdout, f"--status does not mention {s}"


def test_status_reports_a_state_for_every_stage():
    """Every line has to say something; a blank state is how a stage gets
    quietly skipped."""
    out = _run("--status").stdout
    states = {"done", "pending", "STALE", "FAILED"}
    lines = [l for l in out.splitlines() if l.startswith("  ") and l.strip()]
    assert lines
    for l in lines:
        assert any(s in l for s in states), f"no state on: {l!r}"


def test_help_prints_the_header_without_running_anything():
    out = _run("--help")
    assert out.returncode == 0
    assert "--status" in out.stdout and "CHECKPOINT" in out.stdout.upper()


# -- the stub actually serves ----------------------------------------------

@pytest.fixture(scope="module")
def stub_server():
    """The stub over real HTTP, because the offline wiring check talks to it
    through a socket, and a handler that only works when called directly would
    pass every unit test and still serve nothing."""
    import threading
    from http.server import HTTPServer
    sys.path.insert(0, str(ROOT / "tools"))
    import stub_ollama
    srv = HTTPServer(("127.0.0.1", 0), stub_ollama.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_the_stub_advertises_a_model_so_the_script_finds_it(stub_server):
    """run_pipeline.sh decides whether to run the model stages by asking
    /api/tags, so an empty answer here silently skips half the pipeline."""
    import urllib.request
    with urllib.request.urlopen(stub_server + "/api/tags", timeout=5) as r:
        data = json.load(r)
    assert data["models"] and data["models"][0]["name"]


def _l1b_arguments() -> dict:
    """Whatever L1b's template asks for, filled with something plausible."""
    import re

    from natspec_corpus.prompts_v3 import BY_ID
    names = set(re.findall(r"\{(\w+)\}", BY_ID["L1b"].user))
    filled = {n: "" for n in names}
    for n in names:
        if "source" in n or "code" in n:
            filled[n] = "function transfer(address to) external {}"
        elif "sigma" in n or "fact" in n:
            filled[n] = "F1 name=transfer\nN2 EXPRESSION"
    return filled


def test_a_real_client_gets_a_parsable_answer_from_the_stub(stub_server):
    from natspec_corpus.llm import Client, OllamaBackend
    from natspec_corpus.prompts_v3 import BY_ID
    client = Client(OllamaBackend(stub_server), models={})
    call = client.call(BY_ID["L1b"], **_l1b_arguments())
    assert call.first_attempt_parsed
    assert call.data["natspec"].startswith("///")
    assert call.data["claims"][0]["ids"]


def test_an_unknown_path_is_a_404_not_a_silent_empty_answer(stub_server):
    import urllib.error
    import urllib.request
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(stub_server + "/api/nope", timeout=5)
    assert e.value.code == 404


# -- the model pre-flight ---------------------------------------------------
# Finding out that a model name is wrong four hours into a generation run is
# the expensive way to learn it, so the script checks every slot up front.

ALL_SLOTS = ("INTENT_REASONER_MODEL", "GENERATOR_MODEL",
             "SEMANTIC_CRITIC_MODEL", "VERIFIER_MODEL")


@pytest.fixture
def sandbox(tmp_path):
    """--corpus/--results/--state, so a test that gets past the pre-flight and
    actually runs a stage cannot overwrite the real checkpoints or reports.

    It also takes its own run lock. The lock lives in the state directory, and
    the `tests` stage runs the suite from inside a pipeline run that is
    holding the real one — so without this every test here that invokes the
    script fails, but only when run through the pipeline, which is the worst
    place to find out."""
    return ["--corpus", str(tmp_path / "corpus"),
            "--results", str(tmp_path / "results"),
            "--state", str(tmp_path / "state")]


def test_a_model_slot_naming_a_model_the_server_lacks_stops_the_run(stub_server,
                                                                    sandbox):
    out = _run("--only", "harness", "--ollama", stub_server,
               "--models", '{"GENERATOR_MODEL": "not-pulled:70b"}', *sandbox)
    assert out.returncode == 3, out.stdout + out.stderr
    assert "GENERATOR_MODEL" in out.stderr and "not-pulled:70b" in out.stderr


def test_the_message_names_every_unmapped_slot_not_just_the_first(stub_server,
                                                                  sandbox):
    """An unmapped slot is sent as its own literal name. Reporting one at a
    time turns a single fix into four round trips."""
    out = _run("--only", "experiments", "--ollama", stub_server,
               "--models", "{}", *sandbox)
    assert out.returncode == 3
    from natspec_corpus.prompts_v3 import PIPELINE
    for slot in {p.model for p in PIPELINE}:
        assert slot in out.stderr, f"{slot} was not reported"


def test_a_stage_that_needs_no_model_is_not_blocked_by_a_model_name(stub_server,
                                                                    sandbox):
    """Aborting a corpus rebuild over a model it never calls would be its own
    small betrayal, so the check is scoped to the stages that generate."""
    out = _run("--only", "retrieval", "--ollama", stub_server,
               "--models", '{"GENERATOR_MODEL": "not-pulled:70b"}', *sandbox)
    assert out.returncode != 3, out.stdout + out.stderr


def test_every_isolation_option_is_honoured(tmp_path):
    """--corpus, --results and --state exist so a run can be pointed
    somewhere else; a flag that parses but does nothing is worse than none."""
    out = _run("--status", "--corpus", str(tmp_path / "c"),
               "--results", str(tmp_path / "r"), "--state", str(tmp_path / "s"))
    assert out.returncode == 0, out.stderr
    assert "pending" in out.stdout      # nothing is done in an empty state dir
    assert "done" not in out.stdout


def test_slots_mapped_to_a_model_the_server_has_are_accepted(stub_server, sandbox):
    out = _run("--only", "harness", "--ollama", stub_server,
               "--models", json.dumps({s: "stub:latest" for s in ALL_SLOTS}),
               *sandbox)
    assert out.returncode != 3, out.stdout + out.stderr
    assert "does not have" not in out.stderr


def test_a_tag_free_model_name_matches_the_servers_tagged_one(stub_server,
                                                              sandbox):
    """Ollama reports `stub:latest`; people write `stub`. Refusing that would
    be pedantry with a four-hour cost attached."""
    out = _run("--only", "harness", "--ollama", stub_server,
               "--models", json.dumps({s: "stub" for s in ALL_SLOTS}),
               *sandbox)
    assert out.returncode != 3, out.stdout + out.stderr
    assert "does not have" not in out.stderr


def test_a_wrong_tag_is_refused_even_when_the_bare_name_matches(stub_server,
                                                               sandbox):
    """The regression that cost a GPU run.

    The server has `stub:latest`. Asking for `stub:7b` shares the bare name
    and nothing else — they would be different weights — and the earlier
    check compared only the part before the colon, so it accepted it. The
    run then failed on its first call and the harness reported "M1 not met:
    a prompt parses below 95%", because zero answers and bad answers look
    identical to that gate.
    """
    out = _run("--only", "harness", "--ollama", stub_server,
               "--models", json.dumps({s: "stub:7b" for s in ALL_SLOTS}),
               *sandbox)
    assert out.returncode == 3, out.stdout + out.stderr
    assert "does not have" in out.stderr


def test_the_refusal_names_what_the_server_actually_has(stub_server, sandbox):
    """The right tag is usually one character away from the wrong one."""
    out = _run("--only", "harness", "--ollama", stub_server,
               "--models", json.dumps({s: "stub:7b" for s in ALL_SLOTS}),
               *sandbox)
    assert "server has:" in out.stderr and "stub:latest" in out.stderr


def test_the_isolation_options_keep_the_real_state_untouched(stub_server,
                                                            tmp_path):
    """--state is what lets these tests run a stage at all. If it were
    ignored, every run here would stamp the real checkpoints."""
    before = sorted(p.name for p in (ROOT / "state").glob("*"))
    _run("--only", "harness", "--ollama", stub_server,
         "--models", json.dumps({s: "stub" for s in ALL_SLOTS}),
         "--corpus", str(tmp_path / "corpus"),
         "--results", str(tmp_path / "results"),
         "--state", str(tmp_path / "state"))
    assert sorted(p.name for p in (ROOT / "state").glob("*")) == before
    assert (tmp_path / "state").is_dir()


def test_an_unreachable_model_is_not_treated_as_a_bad_model_name(sandbox):
    """No server is a normal state: the model stages skip, the run succeeds."""
    out = _run("--only", "harness", "--ollama", "http://127.0.0.1:1", *sandbox)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "no model reachable" in out.stdout
