# natspec_corpus

Pre-processing for the NatComGen corpus. Rebuilt from scratch because the v1
builder ran regexes over raw Solidity source and two of them were wrong in
ways that corrupted the data silently.

```
python -m natspec_corpus.build [SRC_ROOT] [OUT_ROOT]      # stage 1: extraction
python -m natspec_corpus.sigma_build [CORPUS_ROOT]        # stage 1b: Σ(f)
python -m natspec_corpus.reproduce [SRC] [OUT]            # rebuild + verify
python -m pytest tests -q                                 # 374 tests
python -m coverage run --source=natspec_corpus -m pytest tests  # 92%
```

## Modules

| module | responsibility |
|---|---|
| `errors.py` | typed failures: `LexError`, `ParseError`, `InvariantError`, `ResolutionError` |
| `masking.py` | one single-pass lexer; everything downstream reads its spans |
| `solidity.py` | balanced-delimiter parsing of declarations and parameter lists |
| `natspec.py` | comment text → tags, with solc's implicit-`@notice` rule |
| `extract.py` | attaches doc units to declarations positionally |
| `inherit.py` | resolves `@inheritdoc` across files, following chains |
| `score.py` | verified / partial with a labelled defect taxonomy |
| `closure.py` | file roles and the relative-import closure |
| `projects.py` | audit folder → project slug, and the split assignment |
| `checks.py` | 11 post-build invariants; raises rather than shipping |
| `build.py` | the ten stages, end to end |
| `compile.py` | compilation units, compiler selection, content-addressed cache |
| `sigma.py` | Σ(f): `F*` facts, `N*` CFG, `P*` paths, `C*` calls, `D*` deps |
| `sigma_build.py` | batch Σ(f) over a built corpus |
| `views.py` | the three retrieval views of a function |
| `retrieve.py` | encoders, three FAISS indices, z-normalised fusion |
| `prompts_v3.py` | the five stage-2 prompts + Σ(f) serialisation |
| `llm.py` | one model call: retry, JSON recovery, per-prompt accounting |
| `cache.py` | content-addressed call cache |
| `gate.py` | the deterministic check between L3 and L8 |
| `runner.py` | the five calls per function, resumable |
| `evaluate.py` | per-field scoring against the gold references |
| `harness.py` | prompt validation before any full run |
| `emit.py` | one comment as text: tag order, wrapping, style |
| `assemble.py` | many comments into one file, bottom-up, idempotent |
| `verify_file.py` | whole-file checks: compiles, devdoc complete, code untouched |
| `experiment.py` | the ablation matrix, run in cache-warm order |
| `stats.py` | paired bootstrap, Wilcoxon, Holm, Cliff's delta |
| `report.py` | tables in Markdown and LaTeX, and the ablation figure |
| `reproduce.py` | hashes every stage boundary; rebuilds and checks |

## The design rule

No regex ever runs against raw source. `masking.scan()` classifies every
character once — code, line comment, doc comment, block comment, doc block,
string — and every later stage works on those spans or on the *code view*,
which blanks comments and string interiors while preserving length and
newlines so every offset stays valid in the original file.

That rule exists because the v1 builder did not have it:

* an apostrophe in `/// We don't check this` opened a phantom string literal
  that swallowed the `*/` closing the comment. Two doc blocks merged and
  `ILensNFTBase.burn` was documented with its neighbour's `@param` tags.
* a non-greedy `\(([\s\S]*?)\)` truncated
  `mapping(int16 => uint256) storage self` at the first `)`.

Both are now regression tests, and both have a post-build invariant that would
catch a reintroduction independently of the tests.

## Stage 2 — Σ(f)

`compile.py` resolves a file into its compilation unit (the relative-import
closure, every source inlined by content), then finds a compiler empirically:
try the installed versions newest-first, keep the first that produces an AST.
The pragma is a hint, not an answer — v1 picked from the entry file's pragma
and 64 files failed. Results are cached by content hash, and the batch driver
and the single-contract inference path share one cache.

`sigma.py` runs Slither over that unit and emits, per function, rows with
stable citable ids: `F*` facts (visibility, mutability, parameters, returns,
modifiers, state read/written), `N*` CFG nodes with labelled edges, `P*`
acyclic paths (capped at 16, truncation recorded on the row), `C*` call sites
tagged library / builtin / external / internal / low-level, and `D*`
data-dependency rows.

Slither rather than hand-rolled analysis: modifier splicing, `require` as a
branch, loop back-edges, low-level calls and inline assembly each carry a long
tail of bugs, and Slither already has all of them plus variable-level data
dependency.

Three things this stage gets right that a naive version would not:

* **Byte versus character offsets.** Slither reports byte offsets; the corpus
  records character offsets. Uniswap's `Tick.sol` contains one em-dash at
  character 834, which shifts every later byte offset by two — enough to
  attach a function's control flow to its neighbour, silently. Everything
  crossing that boundary goes through `OffsetMap`.
* **Duplicate call rows.** A library call is also a high-level call, and the
  high-level view stringifies to a whole SlithIR line. The specific kind wins.
* **Reproducibility.** Several of Slither's collections are sets, and its
  dependency table is not a fixpoint, so two runs disagreed. Name lists are
  sorted, call rows are ordered before ids are assigned, and the dependency
  relation is transitively closed here — which also recovers chains that pass
  through a SlithIR temporary.

A fourth: **compilers are identified by what they report, never by their
directory name.** Two solcjs installs here were in the wrong folders, so every
file pinned to `=0.7.6` — four Uniswap v3-core contracts including the pool —
was skipped as uncompilable, and ten others were mislabelled in the report.
`probe()` asks each binary its version once; the same failure is easy to
produce with a hand-managed solc-select tree.

Coverage on the current corpus: 98 of 208 scored files compile from the corpus
alone, giving **546 functions** with fact tables, **401** of which join to a
corpus pair; 0 compile failures and 0 Slither failures. The other 110 files
need npm packages. Every one of the 546 functions lands exactly on a
declaration the extractor found — zero offset mismatches.

## Verification layers

1. **374 unit tests** (92% line coverage; what is left is named in the README) over inline Solidity fixtures — the lexer's ambiguous
   cases, parameter shapes, attachment rules, tag parsing, scoring.
2. **17 build invariants** re-derived from the *written* artifacts, not from
   the in-memory objects, so a bug in the writer is caught too. Twelve cover
   extraction; five cover Σ(f), including the offset join, dangling edge and
   call references, acyclic paths, and no SlithIR leakage.
3. **Determinism**: two consecutive extraction builds are byte-identical, and
   Σ(f) is byte-identical across different `PYTHONHASHSEED` values — the test
   that actually catches set-iteration order.
4. **Compile coverage**: 176 / 333 files produce a solc AST from the corpus
   alone; the rest need npm packages, which is a property of the upstream
   repos, not of this build.

## What is not covered

92% of lines. The uncovered 8% is deliberate and consists of four things:

* **`OllamaBackend` and `HFBackend`** (`llm.py`) and
  **`SentenceTransformerEncoder`** (`retrieve.py`) — every line that talks to
  a real model or downloads one. They cannot run without a GPU box, and a mock
  of them would test the mock.
* **`main()` argument parsing** in `sigma_build.py` and `reproduce.py`.
* **A few compiler-failure branches** in `compile.py` that need solc itself to
  crash rather than return an error.
* **`retrieve.evaluate`** — the cross-split Recall@k function, kept because it
  computes the ceiling that shows why that metric is unusable here, but not
  part of any pipeline.

Every invariant in `checks.py` has a test that makes it **fail**, by name. An
invariant exercised only on data that passes proves it does not fire wrongly;
it does not prove it fires at all.

## A note on the compiler in this container

`binaries.soliditylang.org` is blocked here, so native `solc` binaries cannot
be downloaded and solcjs is shimmed to look like one (`/opt/solcshim`).
`compile.py` reads its compiler root from `SOLC_SHIM_ROOT`, so on a normal
machine `solc-select install` and a real binary work with no code change. The
shim is test scaffolding, not part of the pipeline.
