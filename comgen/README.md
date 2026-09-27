# ComGen

The revised architecture, living inside the NatComGen repository rather than
beside it. NatComGen is 32 modules and 7,414 lines; this architecture changes
about five of them, so a fork would mean maintaining two copies of a corpus
builder, a Σ(f) extractor, a retrieval index and an evaluation harness in order
to change an orchestration loop.

```
  contract intent (once per file) ─┐
  function intent                 ─┴→ Generator ──→ draft
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  │                 CRITIC GROUP                    │
                  │   deterministic (solc + score)   semantic (LLM)  │
                  └───────────────────────┬───────────────────────┘
                                          │  merged feedback
                          Generator (revising) ──→ next candidate
                                          │   (loop, bounded)
                                        Judge
                                          │
                                     Aggregator ──→ Linter Agent
```

## What changed from NatComGen

| | NatComGen (C1) | ComGen (G1) |
|---|---|---|
| critics | one, then a refiner | two, in one group, both reading the draft |
| deterministic check | a gate *after* the refiner; could only veto | a critic *inside* the group; its findings are feedback |
| refinement | a separate Refiner agent, once | the generator itself, up to 3 rounds |
| which candidate ships | the refined one unless the gate vetoed it | the round with the fewest defects |
| contract context | none | one call per file, including `do_not_claim` |
| last word | ClaimVerifier | the same prompt, renamed Judge |
| file output | a pipeline stage | the Aggregator and Linter Agent |
| calls per function | exactly 5 | 4–9, recorded per function |

## The three decisions that are not in a prompt

**Merge precedence** (`critics.py`). Deterministic findings are MUST FIX and
survive the merge intact. A semantic finding that would undo one — "delete that
@param, the evidence doesn't describe it", against "solc dropped that @param" —
is suppressed, and the suppression is recorded. This is the only place the two
critics can genuinely contradict each other, and it is resolved in code rather
than left to a 7B model's judgement inside a prompt.

**Stopping** (`orchestrator.py`). The loop exits the moment the group is clean.
The deterministic critic is the exit guard because it costs nothing: a compile
and a score answer "should we stop?" for free, so there is no reason to spend a
model call on it.

**Best round, not last round** (`orchestrator.select`). Every round's candidate
is kept and the one with the fewest defects wins — blocking first, then total,
ties to the later round. Taking the last round on faith is how v2's refiner
managed to make things worse about as often as better on the functions where it
changed anything.

## Configurations

| | what it removes | what it answers |
|---|---|---|
| G1 | — | the architecture as designed |
| G3 | the loop (1 round) | does looping earn its variable cost |
| G5 | the deterministic critic | |
| G4 | the semantic critic | G4 vs G5: does *a model's* criticism help, or just criticism |
| G6 | contract intent | what one call per file buys |
| G7 | retrieval | comparable to C5 |
| G2 | Σ(f) | comparable to C2 |
| G0 | everything | comparable to C0 |

Run order is `G1 G3 G5 G4 G6 G7 G2 G0` — cache-warm, so the configurations that
share G1's round-0 calls pay nothing for them.

## Results live here

`comgen/results/runs/<config>/seed<n>/<split>.jsonl` — one line per function,
resumable: a restart skips the ids already written.
`comgen/results/tables/` — the six NatComGen tables, plus `rounds` and `cost`.
`comgen/results/documented/` — the emitted source files.

Nothing is written into `data/NatSpecGold/runs`, so a ComGen run cannot disturb
the C0–C7 records. Result versioning applies as everywhere else: the second run
writes `main_2.md`, and `RUNS.md` says what each number was.

The call cache **is** shared, at `<corpus>/.call-cache`. It is keyed on the whole
rendered request, so where ComGen issues a request C1 already issued the answer
is free and — being the same request — identical. A cache is not a result.

## Running it

```bash
# through the pipeline, with its checkpoints
./run_pipeline.sh --only comgen --limit 5 --seeds 0 --models '{...}'
./run_pipeline.sh --from comgen --models '{...}'

# or directly, which is the same code
python -m comgen run    --corpus data/NatSpecGold --split val --seeds "0 1 2"
python -m comgen report --corpus data/NatSpecGold
python -m comgen emit   --corpus data/NatSpecGold --config G1
```

## Tests

`python -m pytest comgen/tests -q` — 50 tests, no model required. They cover the
merge precedence, the stopping rule, the tie-break, the record's compatibility
with the existing evaluation, and that a ComGen run writes nowhere but
`comgen/`. `comgen/tests/fixtures.py` has the scripted backend.

## What is reused unmodified

The corpus, Σ(f), the retrieval index, the compile cache, the call cache, the
checkpoint machinery, and 27 of NatComGen's 32 modules — including
`evaluate.py`, `report.py`, `stats.py` and `versioning.py`, which read a ComGen
record without modification because the record carries the same keys. That is
what makes a ComGen row comparable to a C1 row: the same function computed both.

One line of NatComGen changed: `experiment.run_config` and `run_matrix` now take
`generate_fn`, defaulting to `runner.generate`. `natspec_corpus/gate.py` was not
touched — `solc_emits` and `judge_text` were already single-candidate, and they
*are* the deterministic critic.
