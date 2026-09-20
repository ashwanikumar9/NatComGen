#!/usr/bin/env bash
#
# One command to get NatComGen running on a shared GPU box with no sudo.
#
#   bash tools/gpu_setup.sh              set up, smoke test, then detach a run
#   bash tools/gpu_setup.sh --setup-only stop after the smoke test
#
# Everything it writes stays inside this project directory:
#
#   .venv/    a virtualenv with --system-site-packages, so packages are added
#             here and never into a conda base other people share
#   .solc/    compilers, via tools/install_solc.py — not ~/.solc-select
#   state/    checkpoints        logs/  one log per stage
#
# It never runs `ollama pull`: models live in ~/.ollama, outside this folder,
# and on a shared machine that is not ours to fill. If nothing suitable is
# loaded it says which models it looked for and stops.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
SETUP_ONLY=0
[[ "${1:-}" == "--setup-only" ]] && SETUP_ONLY=1

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
OLLAMA="${OLLAMA:-http://localhost:11434}"

step "1/6  where are we"
echo "  host      $(hostname)"
echo "  dir       $HERE"
echo "  python    $(python3 -V 2>&1)"
command -v nvidia-smi >/dev/null && \
  nvidia-smi --query-gpu=index,name,memory.used,memory.total \
             --format=csv,noheader | sed 's/^/  gpu       /' \
  || echo "  gpu       nvidia-smi not found"

step "2/6  project-local virtualenv"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv --system-site-packages .venv
  echo "  created .venv (inherits whatever conda already provides)"
else
  echo "  .venv already present"
fi
PY="$HERE/.venv/bin/python"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt
echo "  dependencies installed into .venv only"

step "3/6  compilers, inside the project"
"$PY" tools/install_solc.py || true
"$PY" tools/install_solc.py --check || \
  echo "  WARNING: no compiler installed — the emit stage cannot verify output"

step "4/6  environment report"
export SOLC_SHIM_ROOT="$HERE/.solc"
"$PY" setup_env.py --check --json state/env.json || true

step "5/6  which models are loaded"
TAGS="$(curl -fsS "$OLLAMA/api/tags" 2>/dev/null || true)"
if [[ -z "$TAGS" ]]; then
  echo "  no Ollama at $OLLAMA."
  echo "  start it, or pass OLLAMA=http://host:port, then rerun."
  exit 1
fi
MODELS_JSON="$("$PY" - <<'PYEOF2'
import json, os, sys, urllib.request
host = os.environ.get("OLLAMA", "http://localhost:11434").rstrip("/")
with urllib.request.urlopen(host + "/api/tags", timeout=10) as r:
    have = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
if not have:
    print("", end=""); sys.exit(0)
# Prefer a code-trained model, and the largest one for the generator, since
# that is the slot doing the writing. Everything else can be smaller.
def rank(n):
    s = 0
    if "coder" in n or "code" in n: s += 100
    for tag, w in (("32b", 40), ("14b", 30), ("13b", 28), ("8b", 20),
                   ("7b", 18), ("3b", 8), ("1.5b", 4)):
        if tag in n.lower(): s += w
    return s
best = max(have, key=rank)
small = min(have, key=rank) if len(have) > 1 else best
print(json.dumps({"GENERATOR_MODEL": best,
                  "INTENT_REASONER_MODEL": small,
                  "SEMANTIC_CRITIC_MODEL": small,
                  "VERIFIER_MODEL": small}))
PYEOF2
)"
if [[ -z "$MODELS_JSON" ]]; then
  echo "  Ollama is up but has no models. Pull one, then rerun:"
  echo "    ollama pull qwen2.5-coder:7b"
  exit 1
fi
echo "  using: $MODELS_JSON"
echo "$MODELS_JSON" > state/models.json

step "6/6  smoke test — 5 functions, one seed"
# Small on purpose. If a model name or a prompt is wrong, this says so in
# minutes rather than after the whole matrix has run.
./run_pipeline.sh --ollama "$OLLAMA" --models "$MODELS_JSON" \
                  --limit 5 --seeds 0

echo
echo "smoke test passed."
if [[ $SETUP_ONLY -eq 1 ]]; then
  echo "stopping here (--setup-only)."
  exit 0
fi

step "starting the full run, detached"
./run_pipeline.sh --background --ollama "$OLLAMA" --models "$MODELS_JSON"
cat <<EOF

  Check on it later, from any shell:

      cd $HERE
      ./run_pipeline.sh --status          # per-stage: done / STALE / FAILED
      tail -f logs/pipeline.log           # live output
      tail -f logs/experiments.log        # the long stage, function by function

  It survives your SSH session dropping. If it does die, rerunning
  ./run_pipeline.sh resumes where it stopped.

EOF
