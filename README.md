# NatComGen

Agentic NatSpec generation for Solidity smart contracts. Everything in one
tree: the source contracts, the built corpus, the code, the tests and a single
script that runs the whole thing.

```
chmod +x run_pipeline.sh setup_env.py tools/stub_ollama.py   # after unzipping
./setup_env.py --check        # is this machine able to run it?
./run_pipeline.sh --status    # what is done and what is not
./run_pipeline.sh             # run everything that can run
```

The `chmod` is needed because a zip carries no Unix executable bit, so a
transfer through Windows arrives without it. `setup_env.py --check` warns when
it is missing and prints that line; nothing else in the pipeline depends on
it.

---

## The short version

```bash
pip install -r requirements.txt     # or: python setup_env.py
python setup_env.py --solc          # the 13 compilers the corpus pins
./run_pipeline.sh --background      # detach; survives a lost connection
./run_pipeline.sh --status          # check on it from anywhere
```

With no model reachable the run stops cleanly after retrieval and says so.
Point it at one and the rest follows:

```bash
./run_pipeline.sh --ollama http://localhost:11434 \
                  --models '{"GENERATOR_MODEL":"qwen2.5-coder:7b",
                             "INTENT_REASONER_MODEL":"qwen2.5-coder:7b",
                             "SEMANTIC_CRITIC_MODEL":"qwen2.5-coder:7b",
                             "VERIFIER_MODEL":"qwen2.5-coder:7b"}' \
                  --limit 5 --seeds 0        # smoke test first — see below
./run_pipeline.sh --background               # then the real matrix
```

There are four model slots — `INTENT_REASONER_MODEL`, `GENERATOR_MODEL`,
`SEMANTIC_CRITIC_MODEL`, `VERIFIER_MODEL` — and they may all point at the same
model. A slot you leave unmapped is sent to Ollama as its own literal name,
which fails, so the script checks every slot against what Ollama actually has
before it starts generating and names any that are missing.

---

## What is in here

```
NatComGen/
  run_pipeline.sh        the whole pipeline, checkpointed
  setup_env.py           install and verify; names what is missing
  requirements.txt       the Python dependencies
  README.md              this file
  PACKAGE.md             the design notes: why each stage is built as it is

  natspec_corpus/        the 22 modules, stage 1 through stage 4
  tests/                 399 tests, 92% line coverage
  tools/stub_ollama.py   a fake model, for testing the wiring offline

  data/sources/          352 .sol files from 13 audited projects
  data/NatSpecGold/      the built corpus: pairs, splits, Σ(f) fact tables
  results/               tables, figures, manifest, documented contracts
                         one numbered set per run; RUNS.md is the index
  benchmarks/            SmartDoc re-grounded benchmark; see its own README
  state/                 the checkpoints
  logs/                  one log per stage
```

`data/NatSpecGold/` is a build product. It ships built so you can start at any
stage, and `./run_pipeline.sh --force` rebuilds it from `data/sources/` alone.

---

## The nine stages

| stage | what it does | needs a model | roughly |
|---|---|---|---|
| `env` | verifies Python packages, solc, Slither, the model | no | 3s |
| `tests` | the full suite | no | 40s |
| `corpus` | 352 sources → pairs, splits, index allowlist | no | 30s |
| `sigma` | compile + Slither → fact tables Σ(f) | no | 9 min cold, 2s resumed |
| `retrieval` | the three-view index and its leave-one-out recall | no | 5s |
| `harness` | M1/M2/M5 — are the prompts good enough to run? | **yes** | minutes |
| `experiments` | the ablation matrix: 8 configurations × N seeds | **yes** | hours |
| `report` | tables in Markdown and LaTeX, the figure, the manifest | no | 5s |
| `emit` | writes documented .sol files and verifies each one | no | 1 min |

Stages that need a model are **skipped, not failed**, when none is reachable.
An absent GPU is a normal state for this pipeline, not an error, and a run that
reported failure would train you to ignore its failures.

Current numbers on the shipped corpus: 208 scored files and 125 dependency
files; 860 verified pairs and 566 partial; 98 files compile from the corpus
alone giving 546 functions with fact tables, 401 of which join to a pair, with
zero compile and zero Slither failures; 519 training units in the retrieval
index, leave-one-out Recall@1 0.59 and Recall@5 0.83; 148 functions in the
validation split.

The remaining 110 files need npm packages. Run `npm i` in those project
folders and they light up with no code change — worth doing before the
ablations, since it roughly doubles the functions that have fact tables.

---

## Results are never overwritten

The first run writes `results/tables/main.md`. The second writes
`main_2.md`, the third `main_3.md`, and so on — for every table, both its
Markdown and its LaTeX, the figure, the manifest, the retrieval and harness
reports, the emission report and the folder of documented contracts.

Two things make the numbering worth having rather than merely safe.

**The number is per run, not per file.** `main_3.md`, `conditions_3.md`,
`ablations_3.png` and `manifest_3.json` are one run and can be read together.
Numbering each file on its own would hand you a `main_4.md` next to a
`conditions_2.md` the first time a configuration was missing, and nothing on
disk would say they disagreed.

**The number is read off the directory, not from a counter.** Delete
`main_2.md` and the next run is 2 again. There is no hidden state to drift
out of step with what you can see.

`results/RUNS.md` is the index — one row per run, with its date, split,
seeds, configurations and the files it wrote:

```
| run | when                | split | seeds | configs      | files |
| 1   | 2026-09-21 18:40:02 | val   | 0     | C1, C5       | 9: `main.md`, `main.tex`, … |
| 2   | 2026-09-22 09:12:55 | val   | 0,1,2 | C0 … C7      | 13: `main_2.md`, … |
```

That file is the one thing here that *is* rewritten, because it is a view of
the ledger beside it rather than a result. `NATCOMGEN_RESULT_VERSIONING=off`
restores plain overwriting.

Note this interacts with the checkpoints deliberately: a stage that is still
`done` does not re-run, so you get a new numbered set when something actually
changed or when you asked for one with `--force`, not on every invocation.

---

## Checkpoints

This is the part you asked for, so it is worth being precise about what it
does and does not promise.

**Every stage records a fingerprint of its own inputs.** `state/<stage>.done`
holds a hash of everything that stage reads — its source files, the artifacts
of the stages before it, and for the model stages the split, the seeds and the
model name. A stage is skipped only when that fingerprint still matches *and*
the files it was supposed to produce are still on disk.

That second condition matters. A marker on its own is not proof: delete
`data/NatSpecGold/sigma/` and the run must notice, which it does —
`--status` says `STALE its output is gone`.

The first condition is what makes the checkpoints trustworthy rather than
merely fast. Edit a prompt and the harness, experiments and report stages go
stale by themselves. Edit a contract and the corpus rebuilds. Change the model
and the previous run's results are correctly no longer yours. A checkpoint
that ignored its inputs would cheerfully serve you results from last week's
experiment, which is worse than no checkpoint at all.

**Long stages also resume inside themselves.** This is what makes a dropped
connection cheap rather than merely survivable:

* `sigma` writes one shard per analysed file under `sigma/parts/`, keyed by
  the file's content, its compilation unit, its compiler and the analysis code
  itself. Killed 40% of the way through a 9-minute run, the rerun redoes only
  the 40 files it had not finished — measured at 59 seconds — and produces a
  byte-identical `sigma.jsonl`. Finished and rerun, it takes 1.8 seconds.
* compilation is cached by content, shared between the batch driver and the
  single-contract path.
* `experiments` appends one JSON line per function and skips the ids already
  written, and every model call is cached by the content of its request — so
  configuration C6 costs no model calls at all once C1 has run.

So a killed run loses, at most, the one function or the one file it was
working on.

**Interrupting stops the work.** Ctrl-C or a `SIGTERM` takes down the stage's
whole process tree and exits immediately, rather than queueing behind a solc
invocation with minutes left on it. A stage left running after its parent
exited would keep writing into a corpus the next run believes it owns.

**The checkpoints survive being moved.** Fingerprints hash file *contents*,
never absolute paths, so unzipping this tree somewhere else — which is the
first thing that will happen to it — does not invalidate a single one. Copy
the folder to the GPU box, run `--status`, and the five stages that are
already done still say so.

**One run at a time.** `state/.lock` means a second invocation is refused
rather than racing the first over the same checkpoints and the same
append-only run files.

```bash
./run_pipeline.sh --status          # done / pending / STALE / FAILED per stage
./run_pipeline.sh --from sigma      # start again from a stage
./run_pipeline.sh --only report     # just one
./run_pipeline.sh --force           # ignore every checkpoint, redo everything
./run_pipeline.sh --corpus DIR --results DIR --state DIR   # run somewhere else
```

`--background` detaches with `nohup setsid`, so the run outlives the SSH
session that started it. Its output goes to `logs/pipeline.log`, and
`--status` works from any other shell.

---

## On a GPU box

Do the cheap thing first. Finding out that a model name was wrong after four
hours of generation is the expensive way to learn it:

```bash
python setup_env.py                 # install, then verify
python setup_env.py --solc          # the compilers, ~13 versions
./run_pipeline.sh --limit 5 --seeds 0 --models '{"GENERATOR_MODEL":"..."}'
```

Five functions, one seed. That exercises every stage end to end in a few
minutes. When it comes out clean:

```bash
./run_pipeline.sh --background --models '{...}'
```

The `--limit` run's checkpoint does not satisfy the full run, by design: the
sample is part of the experiments fingerprint, so the full matrix re-runs
rather than reporting on five functions. The call cache still makes the
overlap free.

### Testing the wiring with no model at all

`tools/stub_ollama.py` answers like Ollama and returns the smallest
schema-valid reply for each prompt. It measures nothing — the numbers it
produces are meaningless by construction — but it exercises every stage,
including the four that need a model:

```bash
python tools/stub_ollama.py --port 11500 &
./run_pipeline.sh --from harness --ollama http://127.0.0.1:11500 \
                  --limit 4 --seeds 0
```

This is how the `report` stage was caught declaring an output file it had
never written, which had made it re-run on every single invocation.

---

## Verification

```bash
python -m pytest tests -q                                          # 399 tests
python -m pytest tests benchmarks/tests -q                         # 442 with the benchmark
python -m coverage run --source=natspec_corpus,tools -m pytest tests
python -m coverage report                                          # 92%
```

Four layers, described in full in `PACKAGE.md`:

1. **399 unit tests** over inline Solidity fixtures, including the shell
   script itself and the offline stub.
2. **17 build invariants**, re-derived from the *written* artifacts rather
   than the in-memory objects, so a bug in the writer is caught too. Every
   invariant has a test that makes it fail, by name — an invariant exercised
   only on data that passes proves it does not fire wrongly, not that it
   fires at all.
3. **Determinism**: two consecutive corpus builds are byte-identical, and
   Σ(f) is byte-identical across `PYTHONHASHSEED` values.
4. **Emission verification**: every documented file is recompiled and its
   devdoc compared against the input, so a comment can never change the code.

Tests skip only for reasons they name. On a machine with no compiler the
Σ(f) tests skip and say so; with no built corpus, three retrieval tests skip
and print the command that fixes it. No test silently does nothing.

---

## Security

`E:\MTP\GPU Access.txt` holds an SSH host, a username and a plaintext
password. Replace it with an SSH key and delete the file before this folder is
backed up, zipped or shared. No credential from it has been used.

---

## What is still open

* Stage 2 gates **M1, M2 and M5** have been exercised against a scripted
  model, never a real one. They are the first thing the GPU box should run.
* Stage 3 **L10** (contract-level synthesis) and a single-file `document.py`
  driver are designed but not built.
* Stage 4 ablations on the validation split, and the held-out test run, are
  waiting on a model.
* `npm i` in the 110 dependency-blocked projects, before the ablations.
