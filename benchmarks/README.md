# benchmarks/ — putting NatComGen on somebody else's test set

Nothing in this directory is imported by the pipeline. It reads
`natspec_corpus`, never modifies it, and everything it writes lives under
`benchmarks/data/`, `benchmarks/results/`, `benchmarks/state/` and
`benchmarks/logs/`. Deleting the whole directory leaves the project exactly as
it was.

```bash
./benchmarks/run_benchmark.sh --selftest      # proves it works, no download
./benchmarks/run_benchmark.sh --status        # what is done, and coverage
```

---

## The problem this solves

SmartDoc (ASE'21) reports **47.39 BLEU** on 1,000 Solidity functions. Our own
number on our own corpus is not comparable to it, and the gap is not mostly
about model quality:

* their split is random, so a function's near-twin from the same contract can
  sit in training. Measured here on our corpus with a *model-free* 1-NN
  baseline, the same data scores **27.56 ± 0.66** under a random split and
  **0.64** under a project-disjoint one.
* they score `@notice` alone; we generate `@notice`, `@dev`, `@param` and
  `@return`, and the extra tags cannot earn marks against a one-sentence
  reference.
* their references are whitespace-tokenised, so punctuation is its own token.

Measured directly on their release: **434 of the 1,000 test references appear
verbatim in their own training file**. Scored on that split, `smartdoc.out`
gets **81.68** BLEU on the memorisable 434 and **17.53** on the other 566 —
and 17.53 is an ordinary code-summarisation score. The headline 47.39 is
mostly the first column.

| system | all 1000 | seen in train (434) | unseen (566) |
|---|---|---|---|
| smartdoc | 47.39 | 81.68 | 17.53 |
| re2com | 29.37 | 51.18 | 10.41 |
| attendgru | 29.01 | 53.64 | 7.38 |
| ast-attendgru | 26.01 | 48.52 | 6.31 |

None of this is a criticism of their system, which is a good one, and the
duplication is a property of deployed Solidity rather than of anything they
did. It is a statement about what a single random-split BLEU number can tell
you. Every table this benchmark prints carries those three columns, and the
figures above are pinned by a test.

The only way to know how much of the gap is method is to run on **their** test
set, with **their** metric. And that is blocked by how the set was published:
all 1,000 functions are tokenised bodies with no contract around them. They do
not compile. Σ(f) — the fact tables the whole architecture is built on — cannot
be computed for a function that will not compile, and **0 of the 1,000 contain
a `pragma`**.

## Why not just rebuild a contract around each function

That was tried. A compiler-driven scaffolding probe over 40 of them compiled
**0**, and the honest reading is that the test was handicapped: those functions
are Solidity 0.4/0.5-era and were being fed to a `^0.8.0` skeleton. But fixing
the pragma would not fix the real problem.

To make `tiers[_tierId].startDate = _start` compile you must declare `tiers`.
Guess `mapping(uint256 => uint256)` when the original was `Tier[]` and the
contract compiles, Slither runs, and Σ(f) reports facts about a data structure
that never existed. The fact table would be fluent and wrong, and the resulting
BLEU would be a number about our scaffolding.

## Re-grounding instead

SmartDoc crawled verified Etherscan contracts, so most of those 1,000 functions
still exist, verbatim, inside the bulk verified-source corpora. So: index every
function in such a corpus, match each tokenised body back to it, and recover
the **real** file. Real compilation unit, real Σ(f), and a function whose
published reference already exists.

The price is coverage. Only the matched subset can be scored, so that number is
reported everywhere as **"n of 1000"**, and the four published systems are
re-scored on exactly those indices. A system's score on the subset is not its
published score, and comparing our subset number against their full-set number
would be comparing two different test sets that share a name.

### How the matching works

**Exact first.** Whitespace is stripped from both sides and the result is
hashed. SmartDoc's tokeniser only inserts, removes or normalises whitespace —
it does not rename identifiers or drop punctuation — so this is not a
similarity score that happens to be 1.0; the two token streams are identical.
No threshold, nothing to tune.

**Fuzzy second**, only for what exact missed, and only inside a `name/arity`
bucket. Jaccard over token trigrams, with both a floor (0.85) and a margin over
the runner-up (0.05). Deployed Solidity is mostly copies of the same few
contracts — a `transfer/2` bucket holds thousands of near-identical bodies — so
a high score alone proves nothing. A bucket that cannot separate its candidates
yields **no match rather than a guess**: a confident wrong match would attach a
real fact table to someone else's reference, and that is the one error this
whole exercise cannot survive.

### How the corpus is built

The recovered contract is copied, and the SmartDoc reference is written *into*
the copy as a `/// @notice` line above its function, replacing whatever comment
was there. The ordinary corpus builder then extracts it as a gold pair through
exactly the same code path as every other pair in the project, and Σ(f) still
runs, because a comment cannot change what a compiler sees.

That is deliberate. Writing `pairs.jsonl` by hand would couple this benchmark
to a schema that is free to change, and would drift from it silently.

One recovered file becomes one project, `smartdoc-test-NNNNN`, registered in
`natspec_corpus.projects` **in this process only** — the module's list is
appended to and its split sets mutated in place. Nothing on disk is edited; run
the ordinary build in another process and it behaves as it always has.

---

## Running it

### 1. Prove the machinery first

```bash
./benchmarks/run_benchmark.sh --selftest --limit 25
python -m pytest benchmarks/tests -q            # 42 tests
```

The self-test builds a fixture with the same shape out of NatComGen's own 352
audited sources — real functions, tokenised exactly the way SmartDoc's release
is — and runs every stage for real against the offline stub. Two things must
hold: coverage is **1.0** (everything came out of the indexed corpus, so less
is a matcher bug), and **every scored pair's gold notice is byte-equal to its
reference**. The BLEU numbers it prints are meaningless by construction; the
mechanism is what is being tested.

### 2. Get a bulk corpus

This step decides the coverage, and it is the only slow one.

```bash
# best ratio of coverage to disk: 514,506 deduplicated verified sources, MIT
pip install --user datasets pyarrow
python -m benchmarks.smartdoc.fetch_corpus --source disl \
       --dest benchmarks/data/corpus

# broader crawl, much larger on disk; needs github.com
python -m benchmarks.smartdoc.fetch_corpus --source sanctuary \
       --dest benchmarks/data/corpus

# a directory you already have
python -m benchmarks.smartdoc.fetch_corpus --source local \
       --from /path/to/contracts --dest benchmarks/data/corpus
```

Start with `--limit 50000` to see what coverage a tenth of the corpus buys
before committing the disk; the index is resumable, so enlarging the corpus
later re-indexes only what is new.

### 3. The benchmark

```bash
./benchmarks/run_benchmark.sh \
    --corpus-source disl \
    --ollama http://localhost:11434 \
    --models '{"GENERATOR_MODEL":"qwen2.5-coder:7b",
               "INTENT_REASONER_MODEL":"qwen2.5-coder:7b",
               "SEMANTIC_CRITIC_MODEL":"qwen2.5-coder:7b",
               "REFINER_MODEL":"qwen2.5-coder:7b",
               "VERIFIER_MODEL":"qwen2.5-coder:7b"}' \
    --configs C5 --seeds 0
```

Under tmux, since the run is long:

```bash
tmux new -s bench
./benchmarks/run_benchmark.sh --corpus-source disl --ollama ... --models '...'
# detach with ctrl-b d;  reattach with: tmux attach -t bench
./benchmarks/run_benchmark.sh --status      # from any other shell
```

Every stage is checkpointed the way the main pipeline's are: skipped when its
marker exists *and* its outputs are still on disk. `--force` redoes everything,
`--from reground` restarts at a stage, `--only score` runs one.

### 4. Results

`benchmarks/results/smartdoc.md` — one table, every system scored on the same
indices, with the published full-1000 numbers underneath for reference.

Nothing is overwritten. The second run writes `smartdoc_2.md` and
`smartdoc_2.json`, the third `smartdoc_3.*`, and `benchmarks/results/RUNS.md`
records what each run was — corpus, split, coverage, configurations — so a
number in the thesis can be traced back to the run that produced it. The same
scheme applies to the main pipeline's tables; `natspec_corpus/versioning.py`
explains it, and `NATCOMGEN_RESULT_VERSIONING=off` turns it off.

---

## Choices you should know about before reading the table

**C5 (no retrieval) is the default configuration.** SmartDoc's train and test
sets come from one random split of one crawl, so a retrieval pool built from
their training functions would be doing on our side exactly what inflates their
published number. `--configs C1,C5` runs both and shows the difference, which
is the honest way to report it. To build an in-domain pool deliberately,
re-ground their training split too:

```bash
python -m benchmarks.smartdoc.reground --db benchmarks/data/corpus.db \
       --split train --out benchmarks/data/matched_train.jsonl
```

**`@notice` only.** SmartDoc generates one sentence, so the `@dev`, `@param`
and `@return` tags NatComGen produces are dropped before scoring. They have no
counterpart in the references and would only dilute n-gram precision.

**Predictions are tokenised like the references.** Their references put
punctuation in its own token (`EIN .`), so a model writing ordinary English
loses marks purely for attaching full stops. The raw-`.split()` score is
reported alongside, so the size of that adjustment is visible rather than
buried.

**The metric is theirs, and it is checked.** `bleu.py` reimplements what their
`evaluate.py` computes and reproduces **47.39** for `smartdoc.out` on the full
1,000 — the number in the paper — and agrees with nltk to 1e-9 where nltk is
installed. Both are tests, not claims.

---

## What will go wrong

**Coverage will not be 100%.** Some of those functions came from contracts that
are not in whichever corpus you indexed, some were edited between crawl and
publication, and some sit in buckets too crowded to disambiguate safely.
`--status` reports the split between exact and fuzzy matches; read it before
reading the scores.

**Old compilers.** These are 2017–2019 Etherscan contracts, mostly Solidity
0.4.x and 0.5.x. `compile.py` tries installed versions newest-first and keeps
the first that produces an AST, so it simply will not find one unless the old
compilers are installed:

```bash
python setup_env.py --solc          # the versions the main corpus pins
solc-select install 0.4.24 0.4.25 0.4.26 0.5.0 0.5.8 0.5.12 0.5.16 0.5.17
```

A function whose file will not compile still runs — the record carries
`has_sigma: false` and the evidence block says so — but it is measuring the
model without the fact tables, which is the thing the benchmark exists to test.
Check the Σ(f) coverage before drawing conclusions.

**An empty `pairs.jsonl` after the build is expected here.** A SmartDoc
reference is one sentence; the corpus builder scores a comment as *verified*
only when it documents every parameter and return. So almost every injected
pair lands in `pairs_partial.jsonl`, the `artifacts_exist` invariant fires, and
the build step tolerates that one named check and promotes the matched pairs
afterwards. Any other invariant failure is real and is re-raised.

**Non-matched pairs are excluded, not moved to `train`.** The recovered
contracts carry their own documentation, which SmartDoc has no reference for.
Those pairs go to a split named `excluded` that no stage reads — putting them
in `train` would make them retrievable as exemplars for evaluation functions
sitting in the same file.

---

## Files

```
benchmarks/
  run_benchmark.sh        the seven stages, checkpointed
  smartdoc/
    fetch.py              the SmartDoc release, verified by line count
    fetch_corpus.py       a bulk full-source corpus: disl | sanctuary | local
    tokens.py             match keys, buckets, the fuzzy score
    index.py              sqlite index of every function in the corpus
    reground.py           the two matching passes
    build_corpus.py       annotate, stage, build, promote, exclude
    run.py                NatComGen over the result
    bleu.py               their metric, reproduced and tested
    score.py              the comparison table
    selftest.py           the fixture and the gold round-trip check
  tests/                  43 tests
```

No new dependencies for the core path. `datasets` and `pyarrow` only for the
Hugging Face corpus route; `nltk` only for the optional cross-check.

## Still open

* **MMTrans** (Zenodo 4587089) ships a `contracts.zip` of full sources. If it
  contains SmartDoc's contracts, re-grounding against it would be far cheaper
  than crawling. `zenodo.org` was unreachable from the machine this was written
  on; worth five minutes somewhere it is not:
  `curl -sIL "https://zenodo.org/records/4587089/files/contracts.zip"`.
* Coverage on a real corpus is unmeasured — the number the self-test reports is
  1.0 by construction and says nothing about SmartDoc.
* SmartDoc's `final_results/RQ2` and `cross_project` folds are not wired up;
  only RQ1 is.
