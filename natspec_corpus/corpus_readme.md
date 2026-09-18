# NatSpecGold v2

A corpus of Solidity functions paired with the NatSpec comments their authors
wrote, built from 13 audited protocols in DAppSCAN. Built by
`natspec_corpus/`; every number below is reproduced by `python -m
natspec_corpus.build` and is checked by `natspec_corpus/checks.py` before the
build is allowed to write.

## Contents

| file | what it is |
|---|---|
| `contracts/<project>/…` | {{NFILES}} `.sol` files, laid out as in the upstream repo |
| `pairs.jsonl` | {{NVER}} **verified** pairs — the gold references |
| `pairs_partial.jsonl` | {{NPAR}} **partial** pairs, each with defect labels |
| `manifest.json` | per file: role, sha1, pragma, unresolved imports |
| `splits.json` | verified pair ids per split |
| `index_allowlist.json` | which files retrieval may ever index |
| `build_report.json` | the counts this build produced |

## Counts

{{COUNTS}}

Splits are by **project**, not by file: `lens-protocol` and `yield-fydai` are
the test set, `primitive-rmm`, `perp-v2-oracle` and `seaport` the validation
set. Holding out whole projects is what stops a model from being scored on a
contract whose sibling it trained on.

## File roles

`scored` files are the ones whose comments become pairs. `dependency` files
are pulled in only by the relative-import closure, so that a scored file is a
compilable unit. A dependency file is never scored, never retrieved as an
exemplar and never indexed — `index_allowlist.json` lists all 125 under
`never_index`, and the allowlist is default-deny: only files listed under
`train` may enter a retrieval index.

## What "verified" means

A pair is verified when the comment is complete NatSpec for that declaration:

* it has prose — `@notice`, `@dev`, or untagged text, which solc treats as the
  notice;
* every **named** parameter has a non-empty `@param` (unnamed parameters need
  none);
* the number of `@return` tags equals the number of return values;
* no placeholder (`TODO`, `TBD`, `FIXME`, `???`);
* the prose is more than two words and is not just the function name restated.

Everything else is partial, with a labelled defect. The defect histogram for
this build:

{{DEFECTS}}

`param_missing` dominates because the OpenZeppelin house style describes
parameters in prose with backticks instead of `@param` tags. That is real,
widely copied, and it is exactly the defect a comment generator should not
reproduce — which is why those pairs are kept and labelled rather than thrown
away. `pairs_partial.jsonl` is the labelled set the semantic critic is
evaluated against.

## Pair record

```jsonc
{
  "id": "…",                 // stable: sha1(file:qualified-name:offset)
  "project": "uniswap-v3-core",
  "file": "uniswap-v3-core/contracts/libraries/Tick.sol",
  "container": "Tick", "container_kind": "library",
  "kind": "function", "name": "cross",
  "signature": "cross(mapping(int24=>Tick.Info),int24,uint256,uint256)",
  "arity_key": "cross/4",
  "group_id": "…",           // project + signature: interface/impl twins
  "visibility": "internal", "mutability": null,
  "doc_start": 0, "doc_end": 0,
  "doc_spans": [[s,e], …],   // the doc lines only, excluding `//` lines
  "doc_raw": "…",
  "inheritdoc": null, "resolved_from": null,
  "notice": "…", "dev": "…", "params": {…}, "returns": […],
  "code_start": 0, "code_end": 0, "code": "…",
  "verified": true, "defects": [], "split": "train"
}
```

Offsets index the file as shipped in `contracts/`, and the build refuses to
write unless every one of them slices back to the recorded text.

## Compilation

176 of the 333 files produce a solc AST from the corpus alone (solcjs
0.5.16–0.8.13, newest compatible version per unit). The other 157 import npm
packages that are not vendored here — `@openzeppelin` mostly, plus `@uma`,
`@eth-optimism`, `@uniswap`, `@ribbon-finance`. Running `npm i` with the right
remappings in `contracts/` resolves them. `notional` cannot be made to compile
at all: DAppSCAN snapshotted only its 7 audited files.

## Invariants

The build calls `checks.run_all` before it reports success, and raises rather
than shipping a corpus that fails any of:

1. all five artifacts exist and are non-empty
2. every manifest sha1 matches the file on disk, and no file on disk is
   missing from the manifest
3. every pair's `code_start`/`code_end` and `doc_spans` slice back to exactly
   the recorded text
4. no two pairs in a file have overlapping doc ranges
5. no declaration is paired twice; pair ids are unique
6. no dependency file produced a pair
7. every verified pair has prose and carries no defects
8. re-parsing each file from disk yields the same parameters and signature the
   pair claims
9. no `group_id` appears in two splits
10. every import either resolves inside the corpus or is declared unresolved
11. the allowlist is default-deny and no dependency file is indexable

Check 4 is the v1 apostrophe bug; check 8 is the v1 truncated-`mapping` bug.
The build is deterministic — two runs produce byte-identical output.
