#!/usr/bin/env bash
#
# NatComGen — the whole pipeline, start to finish.
#
#   ./run_pipeline.sh                    run everything that can run
#   ./run_pipeline.sh --status           what is done, what is not
#   ./run_pipeline.sh --from sigma       start again from a stage
#   ./run_pipeline.sh --only experiments run one stage
#   ./run_pipeline.sh --force            ignore checkpoints, redo everything
#   ./run_pipeline.sh --seeds "0 1 2"    which seeds the experiments use
#   ./run_pipeline.sh --limit 10         a few functions first, to check a model
#   ./run_pipeline.sh --background       detach; survives a lost connection
#   ./run_pipeline.sh --ollama URL       where the model is
#   ./run_pipeline.sh --corpus DIR       a corpus somewhere other than data/
#   ./run_pipeline.sh --results DIR      write the tables and figures elsewhere
#   ./run_pipeline.sh --state DIR        keep the checkpoints elsewhere
#   ./run_pipeline.sh --models '{"GENERATOR_MODEL":"qwen2.5-coder:7b", ...}'
#
# There are four model slots — INTENT_REASONER_MODEL, GENERATOR_MODEL,
# SEMANTIC_CRITIC_MODEL and VERIFIER_MODEL — and they may all name the same
# model. Every one is checked against what the server actually has before any
# generation starts, because finding out a model name is wrong four hours into
# a run is the expensive way to learn it. So on a GPU box, start small:
#
#   ./run_pipeline.sh --limit 5 --seeds 0 --models '{...}'
#   ./run_pipeline.sh --background --models '{...}'    # then the real thing
#
# CHECKPOINTS. Every stage writes state/<stage>.done holding a fingerprint of
# its inputs. Re-running skips a stage whose fingerprint is unchanged, so an
# interrupted run resumes where it stopped rather than at the beginning. The
# fingerprint matters as much as the marker: if you edit a prompt or a
# contract, the stages downstream of it re-run by themselves, because a
# checkpoint that ignored its inputs would cheerfully serve results from the
# previous experiment.
#
# Long stages are ALSO resumable inside themselves — compilation is cached by
# content, and every generation run appends one JSON line per function and
# skips the ids already written — so even a stage killed halfway loses only
# the function it was working on.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

STATE="$HERE/state"
LOGS="$HERE/logs"
DATA="$HERE/data"
SOURCES="$DATA/sources"
RESULTS="$HERE/results"
CORPUS=""            # --corpus, or $DATA/NatSpecGold; resolved after parsing
PY="${PYTHON:-python3}"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOGS"

# --- options --------------------------------------------------------------
FORCE=0; FROM=""; ONLY=""; STATUS=0; BACKGROUND=0; SPLIT="val"; SEEDS="0 1 2"
LIMIT=""; OLLAMA="${OLLAMA:-http://localhost:11434}"; MODELS="${MODELS:-{\}}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)      FORCE=1 ;;
    --from)       FROM="${2:-}"; shift ;;
    --only)       ONLY="${2:-}"; shift ;;
    --status)     STATUS=1 ;;
    --background) BACKGROUND=1 ;;
    --split)      SPLIT="${2:-val}"; shift ;;
    --seeds)      SEEDS="${2:-0 1 2}"; shift ;;
    --limit)      LIMIT="${2:-}"; shift ;;
    --ollama)     OLLAMA="${2:-}"; shift ;;
    --models)     MODELS="${2:-}"; shift ;;
    --corpus)     CORPUS="${2:-}"; shift ;;
    --results)    RESULTS="${2:-}"; shift ;;
    --state)      STATE="${2:-}"; shift ;;
    -h|--help)    sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

CORPUS="${CORPUS:-$DATA/NatSpecGold}"
mkdir -p "$STATE" "$LOGS" "$RESULTS"

STAGES=(env tests corpus sigma retrieval harness experiments report emit)

# --- checkpoint machinery -------------------------------------------------
# A fingerprint is a hash of everything a stage reads. Change an input and the
# stage re-runs; change nothing and it is skipped.
fingerprint() {
  local stage="$1"
  case "$stage" in
    env)         _hash_files requirements.txt setup_env.py ;;
    tests)       _hash_files natspec_corpus/*.py tests/*.py ;;
    corpus)      _hash_tree "$SOURCES" ; _hash_files natspec_corpus/build.py \
                             natspec_corpus/extract.py natspec_corpus/masking.py \
                             natspec_corpus/solidity.py natspec_corpus/natspec.py \
                             natspec_corpus/inherit.py natspec_corpus/score.py \
                             natspec_corpus/closure.py natspec_corpus/projects.py ;;
    sigma)       _hash_files "$CORPUS/pairs.jsonl" "$CORPUS/manifest.json" \
                             natspec_corpus/sigma.py natspec_corpus/compile.py \
                             natspec_corpus/sigma_build.py ;;
    retrieval)   _hash_files "$CORPUS/sigma/sigma.jsonl" \
                             "$CORPUS/index_allowlist.json" \
                             natspec_corpus/retrieve.py natspec_corpus/views.py ;;
    harness|experiments)
                 _hash_files natspec_corpus/prompts_v3.py \
                             natspec_corpus/experiment.py \
                             natspec_corpus/runner.py natspec_corpus/gate.py
                 # The model and the sample are as much an input as the code:
                 # swap the model and the previous run's results are not yours.
                 echo "$SPLIT $SEEDS $LIMIT $MODELS" ;;
    report)      _hash_tree "$CORPUS/runs" ; _hash_files natspec_corpus/report.py \
                             natspec_corpus/stats.py natspec_corpus/evaluate.py ;;
    emit)        _hash_tree "$CORPUS/runs" ; _hash_files natspec_corpus/emit.py \
                             natspec_corpus/assemble.py \
                             natspec_corpus/verify_file.py ;;
    *)           echo "$stage" ;;
  esac
}

# Both hash CONTENT, never absolute paths. `sha1sum somefile` prints the path
# alongside the digest, so a fingerprint built that way changes the moment the
# tree is moved — unzip this on another machine and every checkpoint reads as
# stale, which is the one thing the checkpoints exist to prevent.
_hash_files() {
  local f
  for f in "$@"; do
    if [[ -f "$f" ]]; then
      printf '%s ' "${f##*/}"
      sha1sum < "$f"
    fi
  done
  return 0
}

_hash_tree() {
  [[ -d "$1" ]] || return 0
  # Hashed from inside the directory, so the names that go into the digest are
  # relative to it — a renamed file still changes the answer, a moved tree
  # does not.
  ( cd "$1" && find . -type f \( -name '*.sol' -o -name '*.jsonl' \) 2>/dev/null \
      | sort | xargs -r sha1sum 2>/dev/null | sha1sum )
}

marker()  { echo "$STATE/$1.done"; }
current() { fingerprint "$1" | sha1sum | cut -c1-16; }

# A marker on its own is not proof: a deleted or half-written output directory
# would otherwise be skipped forever on the strength of a file in state/. So a
# stage counts as done only when the things it was supposed to produce are
# still there.
outputs() {
  case "$1" in
    env)        echo "$STATE/env.json" ;;
    corpus)     echo "$CORPUS/pairs.jsonl" "$CORPUS/splits.json" \
                     "$CORPUS/manifest.json" "$CORPUS/contracts" ;;
    sigma)      echo "$CORPUS/sigma/sigma.jsonl" ;;
    retrieval)  echo "$CORPUS/retrieval_report.json" ;;
    harness)    echo "$CORPUS/harness_report.json" ;;
    experiments) echo "$CORPUS/runs" ;;
    report)     echo "$RESULTS/tables/main.md" "$RESULTS/manifest.json" ;;
    emit)       echo "$RESULTS/emission_report.json" ;;
    *)          echo "" ;;
  esac
}

have_outputs() {
  local o
  for o in $(outputs "$1"); do
    [[ -e "$o" ]] || return 1
  done
  return 0
}

is_done() {
  local stage="$1" m; m="$(marker "$stage")"
  [[ $FORCE -eq 1 ]] && return 1
  [[ -f "$m" ]] || return 1
  have_outputs "$stage" || return 1
  local want have
  want="$(cut -d' ' -f1 < "$m")"
  have="$(current "$stage")"
  [[ "$want" == "$have" ]]
}

mark_done() {
  local stage="$1"
  echo "$(current "$stage") $(date -u +%FT%TZ)" > "$(marker "$stage")"
}

# --- running a stage ------------------------------------------------------
# Each stage logs to its own file and records a failure marker, so a crashed
# stage is visible in --status instead of looking merely unfinished.
run_stage() {
  local stage="$1"; shift
  if is_done "$stage"; then
    printf '  %-12s skipped (done %s)\n' "$stage" \
      "$(cut -d' ' -f2 < "$(marker "$stage")")"
    return 0
  fi
  printf '  %-12s running…\n' "$stage"
  rm -f "$STATE/$stage.failed"
  local log="$LOGS/$stage.log" code=0
  # Backgrounded and waited on rather than run in the foreground, so that a
  # Ctrl-C or a SIGTERM reaches the trap immediately instead of queueing
  # behind a solc invocation that has minutes left to run.
  "$@" >>"$log" 2>&1 &
  CHILD=$!
  wait "$CHILD" || code=$?
  CHILD=""
  if [[ $code -eq 0 ]]; then
    mark_done "$stage"
    printf '  %-12s done\n' "$stage"
  else
    echo "$(date -u +%FT%TZ) exit $code" > "$STATE/$stage.failed"
    printf '  %-12s FAILED (exit %s) — see %s\n' "$stage" "$code" "$log"
    tail -n 12 "$log" | sed 's/^/      | /'
    return "$code"
  fi
}

skip_stage() { printf '  %-12s skipped (%s)\n' "$1" "$2"; }

# --- the stages -----------------------------------------------------------
stage_env()   { "$PY" setup_env.py --check --json "$STATE/env.json"; }
stage_tests() { "$PY" -m pytest tests -q; }

stage_corpus() {
  "$PY" -m natspec_corpus.build "$SOURCES" "$CORPUS"
}

stage_sigma() {
  # The long one, and the one most likely to be interrupted. It is resumable
  # inside itself: compilation is cached by content and every analysed file
  # leaves a shard under sigma/parts/, so a killed run picks up at the file it
  # was on. --force clears that too, because "redo everything" should mean it.
  FORCE="$FORCE" "$PY" - "$CORPUS" <<'SIGMAEOF'
import json, os, sys
from pathlib import Path
from natspec_corpus import sigma_build
rep = sigma_build.build(Path(sys.argv[1]),
                        resume=os.environ.get("FORCE") != "1")
rep.pop("failures", None)
print(json.dumps(rep, indent=1))
SIGMAEOF
}

stage_retrieval() {
  "$PY" - <<'PYEOF'
import json, os
from pathlib import Path
from natspec_corpus.retrieve import build_index, evaluate_loo, coverage, load_units
root = Path(os.environ["CORPUS"])
idx = build_index(root)
out = {"units": len(idx.units),
       "view_coverage": coverage(idx.units),
       "leave_one_out": evaluate_loo(idx),
       "val_units": len(load_units(root, "val"))}
(root / "retrieval_report.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
PYEOF
}

stage_harness() {
  SPLIT="$SPLIT" "$PY" - <<'PYEOF'
import json, os
from pathlib import Path
from natspec_corpus.harness import critic_calibration, parse_rates
from natspec_corpus.llm import Client, OllamaBackend
root = Path(os.environ["CORPUS"])
models = json.loads(os.environ.get("MODELS", "{}"))
client = Client(OllamaBackend(os.environ.get("OLLAMA", "http://localhost:11434")),
                models=models)
m1 = parse_rates(root, client, n=int(os.environ.get("HARNESS_N", "20")),
                 split=os.environ.get("SPLIT", "val"))
print(json.dumps(m1, indent=1))
m2 = critic_calibration(root, client)
print(json.dumps({k: v for k, v in m2.items() if k != "rows"}, indent=1))
(root / "harness_report.json").write_text(
    json.dumps({"M1": m1, "M2": {k: v for k, v in m2.items() if k != "rows"}},
               indent=1))
if not m1["gate_m1_met"]:
    raise SystemExit("M1 not met: a prompt parses below 95% on first attempt")
PYEOF
}

stage_experiments() {
  SPLIT="$SPLIT" SEEDS="$SEEDS" LIMIT="$LIMIT" "$PY" - <<'PYEOF'
import json, os
from pathlib import Path
from natspec_corpus.cache import CallCache
from natspec_corpus.experiment import BY_NAME, RUN_ORDER, run_config
from natspec_corpus.llm import Client, OllamaBackend
from natspec_corpus.retrieve import build_index
from natspec_corpus.runner import load_contexts
root = Path(os.environ["CORPUS"]); split = os.environ.get("SPLIT", "val")
seeds = [int(s) for s in os.environ.get("SEEDS", "0 1 2").split()]
models = json.loads(os.environ.get("MODELS", "{}"))
limit = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
cache = CallCache(root / ".call-cache")
idx = build_index(root)
ctxs = load_contexts(root, split)
# Full system first, so every later configuration reuses its cached calls.
for name in RUN_ORDER:
    for seed in seeds:
        client = Client(OllamaBackend(os.environ.get("OLLAMA",
                                                     "http://localhost:11434")),
                        models=models, cache=cache, seed=seed)
        print(f"== {name} seed {seed}", flush=True)
        run_config(root, BY_NAME[name], client, split=split, seed=seed,
                   index=idx, contexts=ctxs, limit=limit, progress=True)
PYEOF
}

stage_report() {
  SPLIT="$SPLIT" RESULTS="$RESULTS" "$PY" - <<'PYEOF'
import json, os
from collections import defaultdict
from pathlib import Path
from natspec_corpus import reproduce
from natspec_corpus.evaluate import aggregate, score_record
from natspec_corpus.experiment import BY_NAME, RUN_ORDER, load_results
from natspec_corpus.report import write_report
from natspec_corpus.stats import ablation_table
root = Path(os.environ["CORPUS"]); split = os.environ.get("SPLIT", "val")
out = Path(os.environ["RESULTS"])
pairs = {json.loads(l)["id"]: json.loads(l)
         for l in (root / "pairs.jsonl").read_text().splitlines() if l.strip()}
rows = load_results(root / "runs", split)
if not rows:
    raise SystemExit("no runs to report on")
per_seed, results = defaultdict(dict), {}
for name in RUN_ORDER:
    seeds = sorted({r["seed"] for r in rows if r["config"] == name})
    if not seeds:
        continue
    for seed in seeds:
        scored = [score_record(r, pairs[r["pair_id"]])
                  for r in rows
                  if r["config"] == name and r["seed"] == seed
                  and not r.get("error") and r["pair_id"] in pairs]
        per_seed[name][seed] = {
            s["pair_id"]: (s["support_rate"] if s["support_rate"] is not None
                           else (s["by_kind"].get("notice") or {}).get("bleu", 0.0))
            for s in scored}
        if seed == seeds[0]:
            results[name] = aggregate(scored)
abl = None
if "C1" in per_seed and len(per_seed) > 1:
    abl = ablation_table(per_seed["C1"],
                         {n: v for n, v in per_seed.items() if n != "C1"})
man = reproduce.manifest(root, models=json.loads(os.environ.get("MODELS", "{}")),
                         seeds=sorted({r["seed"] for r in rows}),
                         configs=sorted(results), split=split)
written = write_report(out, results=results,
                       labels={n: BY_NAME[n].label for n in results},
                       ablations=abl, manifest=man)
print(json.dumps({"files": sorted(written), "configs": sorted(results)},
                 indent=1))
PYEOF
}

stage_emit() {
  RESULTS="$RESULTS" SPLIT="$SPLIT" "$PY" - <<'PYEOF'
import json, os
from pathlib import Path
from natspec_corpus.assemble import document_file
from natspec_corpus.compile import compile_unit, unit_for
from natspec_corpus.emit import Comment
from natspec_corpus.evaluate import fields
from natspec_corpus.experiment import load_results
from natspec_corpus.extract import build_file
from natspec_corpus.verify_file import strip_doc_comments, verify
root = Path(os.environ["CORPUS"]); out = Path(os.environ["RESULTS"]) / "documented"
out.mkdir(parents=True, exist_ok=True)
pairs = {json.loads(l)["id"]: json.loads(l)
         for l in (root / "pairs.jsonl").read_text().splitlines() if l.strip()}
rows = [r for r in load_results(root / "runs", os.environ.get("SPLIT", "val"))
        if r.get("config") == "C1" and not r.get("error")]
contracts = root / "contracts"
read = lambda x: (contracts / x).read_text() if (contracts / x).is_file() else None
by_file = {}
for r in rows:
    by_file.setdefault(r["file"], []).append(r)
report = []
for rel, group in sorted(by_file.items()):
    u = unit_for(rel, read)
    res = compile_unit(u, cache_dir=root / ".compile-cache")
    if not res.ok:
        continue
    src = u.sources[rel]
    want = {}
    for r in group:
        p = pairs.get(r["pair_id"])
        if not p:
            continue
        f = fields(r["final"])
        want[(p["container"], p["signature"])] = Comment(
            notice=f.get("notice", ""), dev=f.get("dev", ""),
            params={k.split(":", 1)[1]: v for k, v in f.items()
                    if k.startswith("param:")},
            returns=[v for k, v in sorted(f.items()) if k.startswith("return:")])
    stripped = strip_doc_comments(src, rel=rel)
    m = build_file(rel, stripped)
    by = {}
    for d in m.decls:
        by.setdefault((d.container, d.sig), d.header_start)
    cm = {by[k]: v for k, v in want.items() if k in by and not v.empty}
    if not cm:
        continue
    emitted = document_file(stripped, cm, mode="fill_gaps")
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(emitted, encoding="utf-8")
    report.append(verify(rel, src, emitted, u.sources, res.version).to_dict())
(Path(os.environ["RESULTS"]) / "emission_report.json").write_text(
    json.dumps(report, indent=1))
ok = sum(1 for r in report if r["ok"])
print(f"{len(report)} files emitted, {ok} fully verified")
PYEOF
}

# --- status ---------------------------------------------------------------
if [[ $STATUS -eq 1 ]]; then
  echo "NatComGen pipeline — $HERE"
  echo
  for s in "${STAGES[@]}"; do
    if [[ -f "$STATE/$s.failed" ]]; then
      printf '  %-12s FAILED   %s\n' "$s" "$(cat "$STATE/$s.failed")"
    elif is_done "$s"; then
      printf '  %-12s done     %s\n' "$s" \
        "$(cut -d' ' -f2 < "$(marker "$s")")"
    elif [[ -f "$(marker "$s")" ]] && ! have_outputs "$s"; then
      printf '  %-12s STALE    its output is gone\n' "$s"
    elif [[ -f "$(marker "$s")" ]]; then
      printf '  %-12s STALE    inputs changed since it ran\n' "$s"
    else
      printf '  %-12s pending\n' "$s"
    fi
  done
  exit 0
fi

# --- detach ---------------------------------------------------------------
if [[ $BACKGROUND -eq 1 ]]; then
  shift_args=()
  [[ -n "$FROM" ]] && shift_args+=(--from "$FROM")
  [[ -n "$ONLY" ]] && shift_args+=(--only "$ONLY")
  [[ $FORCE -eq 1 ]] && shift_args+=(--force)
  shift_args+=(--split "$SPLIT" --seeds "$SEEDS")
  shift_args+=(--ollama "$OLLAMA" --models "$MODELS")
  [[ -n "$LIMIT" ]] && shift_args+=(--limit "$LIMIT")
  nohup setsid "$0" "${shift_args[@]}" >>"$LOGS/pipeline.log" 2>&1 < /dev/null &
  echo "detached as pid $! — log: $LOGS/pipeline.log"
  echo "check with: $0 --status"
  exit 0
fi

# --- one run at a time ----------------------------------------------------
# Two concurrent runs would race on the same checkpoints and the same append
# only run files, so the second one waits rather than corrupting the first.
exec 9>"$STATE/.lock"
if ! flock -n 9; then
  echo "another run is already going (state/.lock). Use --status to watch it." >&2
  exit 1
fi

# Interrupting must stop the work, not just the script: a stage left running
# after its parent exits would keep writing to a corpus the next run believes
# it owns. The stage's whole process tree goes down with it.
CHILD=""
_kill_tree() {
  local pid="$1" kid
  for kid in $(pgrep -P "$pid" 2>/dev/null || true); do _kill_tree "$kid"; done
  kill -TERM "$pid" 2>/dev/null || true
}
on_signal() {
  trap - INT TERM
  [[ -n "$CHILD" ]] && _kill_tree "$CHILD"
  echo
  echo "interrupted — progress is checkpointed; rerun to resume where it stopped"
  exit 130
}
trap on_signal INT TERM

echo "NatComGen pipeline"
echo "  root    $HERE"
echo "  split   $SPLIT   seeds: $SEEDS"
echo

selected=("${STAGES[@]}")
if [[ -n "$ONLY" ]]; then
  selected=("$ONLY")
elif [[ -n "$FROM" ]]; then
  keep=0; selected=()
  for s in "${STAGES[@]}"; do
    [[ "$s" == "$FROM" ]] && keep=1
    [[ $keep -eq 1 ]] && selected+=("$s")
  done
  [[ ${#selected[@]} -eq 0 ]] && { echo "unknown stage: $FROM" >&2; exit 2; }
fi

# Is a model reachable, and is it the model the prompts ask for? Stages 3 and
# 4 are skipped rather than failed when nothing answers: an unreachable model
# is a normal state here, not an error. A model that answers but does not have
# the tags the prompts name is different — that fails now, loudly, rather than
# four hours into a run.
HAVE_MODEL=0
export OLLAMA MODELS
MODEL_CHECK="$("$PY" - <<'PYEOF' 2>/dev/null
import json, os, sys, urllib.request

host = os.environ.get("OLLAMA", "http://localhost:11434").rstrip("/")
try:
    with urllib.request.urlopen(host + "/api/tags", timeout=4) as r:
        have = {m.get("name", "") for m in json.loads(r.read()).get("models", [])}
except Exception:
    sys.exit(1)

# An unmapped slot is sent as its own literal name, which no server has. Say
# which slot, because "model not found" alone sends you hunting.
try:
    mapping = json.loads(os.environ.get("MODELS") or "{}")
except ValueError:
    print("MODELS is not valid JSON")
    sys.exit(0)
sys.path.insert(0, os.getcwd())          # the script has already cd'd to HERE
try:
    from natspec_corpus.prompts_v3 import PIPELINE
    slots = sorted({p.model for p in PIPELINE})
except Exception:
    slots = []
bare = {n.split(":")[0] for n in have}
missing = [f"{s} -> {mapping.get(s, s)}" for s in slots
           if mapping.get(s, s) not in have
           and mapping.get(s, s).split(":")[0] not in bare]
print("; ".join(missing))
PYEOF
)" && HAVE_MODEL=1

# Only when a stage that will actually call a model is in this run. Aborting a
# corpus rebuild over a model name it never uses would be its own small
# betrayal.
NEEDS_MODEL=0
for s in "${selected[@]}"; do
  [[ "$s" == "harness" || "$s" == "experiments" ]] && NEEDS_MODEL=1
done

if [[ $NEEDS_MODEL -eq 1 && $HAVE_MODEL -eq 1 && -n "$MODEL_CHECK" ]]; then
  echo "  the model at $OLLAMA does not have: $MODEL_CHECK" >&2
  echo "  map every slot with --models '{\"SLOT\":\"model:tag\"}', or pull it" >&2
  exit 3
fi

export CORPUS SOURCES RESULTS OLLAMA MODELS
for s in "${selected[@]}"; do
  case "$s" in
    harness|experiments)
      if [[ $HAVE_MODEL -eq 0 ]]; then
        skip_stage "$s" "no model reachable — start ollama and rerun"
        continue
      fi ;;
    report|emit)
      if [[ ! -d "$CORPUS/runs" ]]; then
        skip_stage "$s" "nothing generated yet"
        continue
      fi ;;
  esac
  run_stage "$s" "stage_$s"
done

echo
echo "done. results in $RESULTS, logs in $LOGS"
echo "status: $0 --status"
