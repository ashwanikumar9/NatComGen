#!/usr/bin/env bash
# The SmartDoc re-grounded benchmark, end to end.
#
#   ./benchmarks/run_benchmark.sh --corpus-source disl        # the real thing
#   ./benchmarks/run_benchmark.sh --selftest                  # no download
#   ./benchmarks/run_benchmark.sh --status
#
# Every stage is checkpointed the same way the main pipeline is: a stage is
# skipped when its marker exists and its outputs are still on disk. Delete a
# marker, or pass --force, to redo one.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DATA="$HERE/data"; RESULTS="$HERE/results"; STATE="$HERE/state"; LOGS="$HERE/logs"
mkdir -p "$DATA" "$RESULTS" "$STATE" "$LOGS"

CORPUS_SOURCE="self"; CORPUS_FROM=""; CORPUS_LIMIT=""
SPLIT="test"; FORCE=0; STATUS=0; SELFTEST=0; ONLY=""; FROM=""
CONFIGS="C5"; SEEDS="0"; LIMIT=""; OLLAMA=""; MODELS=""; INDEX_ROOT=""

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --corpus-source) CORPUS_SOURCE="$2"; shift 2;;
    --corpus-from)   CORPUS_FROM="$2";   shift 2;;
    --corpus-limit)  CORPUS_LIMIT="$2";  shift 2;;
    --split)         SPLIT="$2";         shift 2;;
    --configs)       CONFIGS="$2";       shift 2;;
    --seeds)         SEEDS="$2";         shift 2;;
    --limit)         LIMIT="$2";         shift 2;;
    --ollama)        OLLAMA="$2";        shift 2;;
    --models)        MODELS="$2";        shift 2;;
    --only)          ONLY="$2";          shift 2;;
    --from)          FROM="$2";          shift 2;;
    --force)         FORCE=1;            shift;;
    --status)        STATUS=1;           shift;;
    --selftest)      SELFTEST=1;         shift;;
    -h|--help)       usage;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

STAGES=(fetch corpus index reground build run score)

outputs() {
  case "$1" in
    fetch)    echo "$DATA/smartdoc/ref.txt $DATA/smartdoc/fetch_report.json";;
    corpus)   echo "$DATA/corpus";;
    index)    echo "$DATA/corpus.db";;
    reground) echo "$DATA/matched.jsonl $DATA/coverage_${SPLIT}.json";;
    build)    echo "$DATA/smartdoc_corpus/pairs.jsonl $DATA/corpus_map_${SPLIT}.json";;
    run)      echo "$DATA/smartdoc_corpus/runs";;
    score)    echo "$RESULTS/smartdoc.json $RESULTS/smartdoc.md";;
  esac
}

have_outputs() { local o; for o in $(outputs "$1"); do [[ -e "$o" ]] || return 1; done; return 0; }
done_marker()  { echo "$STATE/$1.done"; }
is_done()      { [[ -f "$(done_marker "$1")" ]] && have_outputs "$1"; }

if [[ $STATUS -eq 1 ]]; then
  printf '%-10s %s\n' stage state
  for s in "${STAGES[@]}"; do
    if is_done "$s"; then st="done"
    elif [[ -f "$(done_marker "$s")" ]]; then st="STALE its output is gone"
    else st="pending"; fi
    printf '%-10s %s\n' "$s" "$st"
  done
  [[ -f "$DATA/coverage_${SPLIT}.json" ]] && \
    python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(f\"\ncoverage: {d['matched']} of {d['total']} ({d['coverage']*100:.1f}%)  exact {d['exact']}  fuzzy {d['fuzzy']}\")" "$DATA/coverage_${SPLIT}.json"
  exit 0
fi

run_stage() {
  local name="$1"; shift
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then return 0; fi
  if [[ $FORCE -eq 0 ]] && is_done "$name"; then
    echo "== $name: done, skipping"; return 0
  fi
  echo "== $name"
  if "$@" 2>&1 | tee "$LOGS/$name.log"; then
    : > "$(done_marker "$name")"
  else
    echo "!! $name failed — see $LOGS/$name.log" >&2; exit 1
  fi
}

started=0
for s in "${STAGES[@]}"; do
  [[ -n "$FROM" && $started -eq 0 && "$s" != "$FROM" ]] && continue
  started=1
  case "$s" in
    fetch) run_stage fetch python3 -m benchmarks.smartdoc.fetch --dest "$DATA/smartdoc" ;;
    corpus)
      if [[ $SELFTEST -eq 1 ]]; then
        run_stage corpus python3 -m benchmarks.smartdoc.fetch_corpus --source self --dest "$DATA/corpus"
      else
        args=(--source "$CORPUS_SOURCE" --dest "$DATA/corpus")
        [[ -n "$CORPUS_FROM"  ]] && args+=(--from "$CORPUS_FROM")
        [[ -n "$CORPUS_LIMIT" ]] && args+=(--limit "$CORPUS_LIMIT")
        run_stage corpus python3 -m benchmarks.smartdoc.fetch_corpus "${args[@]}"
      fi ;;
    index)    run_stage index python3 -m benchmarks.smartdoc.index "$DATA/corpus" --db "$DATA/corpus.db" ;;
    reground)
      if [[ $SELFTEST -eq 1 ]]; then
        run_stage reground python3 -m benchmarks.smartdoc.selftest --stage reground --data "$DATA"
      else
        run_stage reground python3 -m benchmarks.smartdoc.reground \
          --db "$DATA/corpus.db" --data "$DATA/smartdoc" \
          --out "$DATA/matched.jsonl" --split "$SPLIT"
      fi ;;
    build)
      run_stage build python3 -m benchmarks.smartdoc.build_corpus \
        --matched "$DATA/matched.jsonl" --index-root "$DATA/corpus" \
        --out-src "$DATA/smartdoc_sources" --out-corpus "$DATA/smartdoc_corpus" \
        --split "$SPLIT"
      if [[ $SELFTEST -eq 1 && -z "$ONLY" ]]; then
        # The reference has to survive the trip into a `///` line, through
        # the corpus builder and back out of pairs.jsonl byte for byte. If it
        # does not, every count downstream still looks right and the
        # benchmark quietly scores against the wrong sentence.
        python3 -m benchmarks.smartdoc.selftest --stage check --data "$DATA" \
          | tee "$LOGS/selftest-check.log" || exit 1
      fi ;;
    run)
      args=(--corpus "$DATA/smartdoc_corpus" --configs "$CONFIGS" --seeds "$SEEDS")
      [[ -n "$LIMIT"  ]] && args+=(--limit "$LIMIT")
      [[ -n "$OLLAMA" ]] && args+=(--ollama "$OLLAMA")
      [[ -n "$MODELS" ]] && args+=(--models "$MODELS")
      [[ $SELFTEST -eq 1 ]] && args+=(--stub)
      run_stage run python3 -m benchmarks.smartdoc.run "${args[@]}" ;;
    score)
      data_dir="$DATA/smartdoc"
      [[ $SELFTEST -eq 1 ]] && data_dir="$DATA/selftest"
      run_stage score python3 -m benchmarks.smartdoc.score \
        --corpus "$DATA/smartdoc_corpus" \
        --corpus-map "$DATA/corpus_map_${SPLIT}.json" \
        --data "$data_dir" --out "$RESULTS/smartdoc.json" ;;
  esac
done

echo
echo "results: $RESULTS/smartdoc.md"
