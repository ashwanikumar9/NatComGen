"""Does the comment name what the function DOES? A reference-free metric.

BLEU cannot tell a near-miss from an inversion. "Enables an interface"
against a reference of "Disables an interface" scores almost perfectly on
every n-gram order while asserting the opposite of the truth, and a reader
who trusts it mis-audits the contract. So this module asks a question BLEU
structurally cannot:

  verb agreement  of functions whose name begins with a verb, the share whose
                  comment uses that verb (or a synonym of it)
  inversion rate  of those, the share whose comment uses the verb's ANTONYM
                  and not the verb -- `disableInterface` described as
                  "Enables", `setInfo` as "Retrieves", `getDeed` as "Set"

Neither uses the reference, so the reference itself is scored in the same
table and a system may exceed it. The second number is the one that matters:
a missing verb is an incomplete comment, an inverted verb is a WRONG one, and
the two failures carry very different costs downstream.

Two corpora, one metric:

    python3 -m functionWise.verbs --pkg ~/Ashwani/MTP/smartdoc-pkg \
        --comgen-runs comgen/results/smartdoc_runs
    python3 -m functionWise.verbs --corpus data/NatSpecGold --split test

CAVEAT, STATED UP FRONT. The verb in a function's name is strong evidence of
its intent but not proof: a setter may legitimately be described as
"Updates", which the synonym table covers, or as "Records the new value",
which it does not. So verb agreement UNDERSTATES every system, including the
reference, by roughly the same amount -- read it as a comparison between
rows, never as an absolute. The inversion count does not have this problem:
asserting the antonym is wrong however the sentence is phrased.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from natspec_corpus.report import markdown_table               # noqa: E402
from natspec_corpus.versioning import next_path                # noqa: E402

from functionWise import scope, surface                        # noqa: E402

SYSTEMS = ("attendgru", "ast-attendgru", "re2com", "smartdoc")

# Pairs that genuinely invert intent. Deliberately conservative: a pair earns
# a place here only when naming one in place of the other makes the comment
# false, not merely imprecise. Read both ways at import.
_OPPOSITE: Dict[str, str] = {
    "enable": "disable", "add": "remove", "mint": "burn", "set": "get",
    "deposit": "withdraw", "lock": "unlock", "freeze": "unfreeze",
    "pause": "unpause", "approve": "revoke", "grant": "revoke",
    "open": "close", "start": "stop", "increase": "decrease",
    "increment": "decrement", "buy": "sell", "create": "destroy",
    "register": "deregister", "whitelist": "blacklist", "allow": "deny",
    "activate": "deactivate", "attach": "detach", "stake": "unstake",
    "wrap": "unwrap", "fund": "defund", "credit": "debit",
}
ANTONYMS: Dict[str, Tuple[str, ...]] = {}
for _a, _b in _OPPOSITE.items():
    ANTONYMS.setdefault(_a, ()); ANTONYMS.setdefault(_b, ())
    ANTONYMS[_a] = tuple(set(ANTONYMS[_a] + (_b,)))
    ANTONYMS[_b] = tuple(set(ANTONYMS[_b] + (_a,)))

# Ways a correct comment may say the same verb. Keep these TIGHT: a loose
# synonym inflates every row and hides the gap the table exists to show.
SYN: Dict[str, Tuple[str, ...]] = {
    "get": ("return", "retrieve", "read", "query", "fetch", "give"),
    "set": ("update", "assign", "change", "store", "write", "modify"),
    "add": ("insert", "append", "push", "register"),
    "remove": ("delete", "erase", "drop", "clear"),
    "mint": ("issue", "create"),
    "burn": ("destroy",),
    "transfer": ("send", "move"),
    "withdraw": ("claim", "redeem"),
    "deposit": ("fund",),
    "check": ("verify", "validate", "ensure", "test"),
    "is": ("return", "whether", "check"),
    "calculate": ("compute", "determine"),
    "approve": ("allow", "permit", "authorise", "authorize"),
    "disable": ("turn off", "switch off"),
    "enable": ("turn on", "switch on", "allow"),
    "init": ("initialise", "initialize", "construct"),
    "execute": ("run", "perform", "carry out", "call"),
}

# Name prefixes that carry no intent of their own; look at the next piece.
_SKIP = ("_", "do", "safe", "try", "internal", "external", "public")
_NAME = re.compile(r"\bfunction\s+([A-Za-z_$][\w$]*)")


def function_name(code: str) -> Optional[str]:
    """`function mintToken ( address ... ) ...` -> `mintToken`."""
    m = _NAME.search(code or "")
    return m.group(1) if m else None


def leading_verb(name: str) -> Optional[str]:
    """The intent-carrying first word of a function name, or None."""
    pieces = [p for p in surface.split_identifier(name or "") if p]
    while pieces and pieces[0] in _SKIP:
        pieces = pieces[1:]
    if not pieces:
        return None
    head = pieces[0]
    # a name that begins with a noun tells us nothing; require a known verb
    # or a plausible verb shape (letters only, not a type or qualifier word)
    if head in SYN or head in ANTONYMS or any(head in v for v in SYN.values()):
        return head
    return head if head.isalpha() and len(head) > 2 else None


def _stems(word: str) -> Tuple[str, ...]:
    """Crude inflections: set/sets/setting, disable/disables/disabled."""
    base = word.rstrip("e") if len(word) > 3 and word.endswith("e") else word
    return tuple({word, word + "s", word + "es", word + "d", word + "ed",
                  base + "ing", base + "ed", base + "es"})


def _said(text: str, word: str) -> bool:
    low = " " + re.sub(r"[^a-z ]+", " ", (text or "").lower()) + " "
    if " " in word:
        return f" {word} " in low
    return any(f" {s} " in low for s in _stems(word))


def says(text: str, verb: str) -> bool:
    """Does the comment use this verb, or a synonym this table allows?"""
    if _said(text, verb):
        return True
    return any(_said(text, s) for s in SYN.get(verb, ()))


# Synonyms too generic to PROVE an inversion. They are fine evidence that a
# comment names its own verb, but seeing one is not evidence that it named the
# opposite: `getWithdrawableDates` described as "Does n't change state" is not
# a comment claiming to set anything, it just contains a common word the table
# maps to `set`.
_WEAK = frozenset({"change", "modify", "store", "write", "allow", "permit",
                   "create", "run", "call", "perform", "give", "test"})


def inverts(text: str, verb: str) -> bool:
    """Does it use the verb's antonym INSTEAD of the verb?

    Asymmetric on purpose: an antonym's own inflections count, and so do its
    specific synonyms ("Retrieves" for `set`, whose antonym is `get`), but its
    weak ones do not. Routing the full synonym table in both directions
    manufactures false positives.
    """
    if says(text, verb):
        return False
    for a in ANTONYMS.get(verb, ()):
        if _said(text, a):
            return True
        if any(_said(text, s) for s in SYN.get(a, ()) if s not in _WEAK):
            return True
    return False


def measure(items: Sequence[Tuple[str, str]]) -> dict:
    """items: (verb, comment). Comments that are None are skipped."""
    named = wrong = scored = 0
    bad: List[Tuple[str, str]] = []
    for verb, comment in items:
        if comment is None:
            continue
        scored += 1
        if says(comment, verb):
            named += 1
        elif inverts(comment, verb):
            wrong += 1
            bad.append((verb, comment))
    return {"n": scored, "named": named, "inverted": wrong, "examples": bad,
            "agreement": named / scored if scored else None,
            "inversion_rate": wrong / scored if scored else None}


def _pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def _row(label: str, m: dict) -> List[str]:
    return [label, str(m["n"]), str(m["named"]), _pct(m["agreement"]),
            str(m["inverted"]), _pct(m["inversion_rate"])]


HEAD = ["comment", "n scored", "names the verb", "verb agreement",
        "states its antonym", "inversion rate"]

NOTE = ("\n`verb agreement` is the share of verb-named functions whose "
        "comment uses that verb or an allowed synonym; `inversion rate` is "
        "the share whose comment uses the verb's **antonym** instead. "
        "Agreement understates every row equally — a correct comment may "
        "phrase the action a way the synonym table misses — so compare rows, "
        "not absolutes. An inversion is unambiguous: the comment asserts the "
        "opposite of what the code does, and BLEU rewards it for the "
        "n-grams it shares with the truth.\n")


# ---------------------------------------------------------------------------
# SmartDoc's released artefact
# ---------------------------------------------------------------------------
def _lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _comgen_by_line(args) -> Dict[int, str]:
    idx = args.eval_corpus.expanduser() / "index.jsonl"
    if not idx.exists():
        return {}
    line_of = {}
    for l in idx.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            line_of[r["id"]] = r["line"]
    got: Dict[int, str] = {}
    root = args.comgen_runs.expanduser() if args.comgen_runs else None
    if root is None or not root.exists():
        return {}
    for path in sorted(root.rglob("test.jsonl")):
        for l in path.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            rec = json.loads(l)
            if (rec.get("config") != args.comgen_config
                    or rec.get("seed") != args.comgen_seed
                    or rec.get("error")):
                continue
            ln = line_of.get(rec.get("pair_id"))
            if ln is not None:
                got[ln] = rec.get("final") or ""
    return got


def on_package(args) -> str:
    pkg = args.pkg.expanduser()
    split = args.split_dir
    code = _lines(pkg / "final_results" / split / "code.txt") \
        if (pkg / "final_results" / split / "code.txt").exists() \
        else _lines(pkg / "dataset" / "test" / "test.token.code")
    ref = _lines(pkg / "final_results" / split / "ref.txt")
    outs = {s: _lines(pkg / "final_results" / split / f"{s}.out")
            for s in SYSTEMS
            if (pkg / "final_results" / split / f"{s}.out").exists()}
    n = min([len(code), len(ref)] + [len(v) for v in outs.values()])

    verbs: Dict[int, str] = {}
    for i in range(n):
        name = function_name(code[i])
        v = leading_verb(name) if name else None
        if v:
            verbs[i] = v

    reproduced = {i for i in verbs
                  if outs.get("smartdoc") and
                  surface.normalize(outs["smartdoc"][i], "paper")
                  == surface.normalize(ref[i], "paper")}

    comgen = _comgen_by_line(args)
    shared = sorted(i for i in verbs if i in comgen) if comgen else []

    doc = [f"# Verb agreement — SmartDoc {split}\n",
           f"{n} functions aligned; {len(verbs)} ({len(verbs)/n*100:.0f}%) "
           f"have a name that begins with a verb. Those are the ones scored.\n"]

    rows = [_row("their reference", measure([(verbs[i], ref[i])
                                            for i in verbs]))]
    for s, hyp in outs.items():
        rows.append(_row(s, measure([(verbs[i], hyp[i]) for i in verbs])))
        if s == "smartdoc" and reproduced:
            keep = [i for i in verbs if i not in reproduced]
            rows.append(_row(f"{s} — excl. verbatim reproductions",
                             measure([(verbs[i], hyp[i]) for i in keep])))
    doc.append("## Every system on their test set\n\n"
               + markdown_table(HEAD, rows) + NOTE)

    if shared:
        sub = [(verbs[i], ref[i]) for i in shared]
        rows = [_row("their reference", measure(sub))]
        if "smartdoc" in outs:
            rows.append(_row("SmartDoc", measure(
                [(verbs[i], outs["smartdoc"][i]) for i in shared])))
        rows.append(_row(f"ComGen {args.comgen_config}", measure(
            [(verbs[i], surface.join(surface.hypothesis_view(
                comgen[i], "notice"))) for i in shared])))
        doc.append(f"\n## Both systems, the same {len(shared)} functions\n\n"
                   + markdown_table(HEAD, rows))
        doc.append("\nOne denominator, no reference used. ComGen runs here "
                   "without a fact table — their release does not compile — "
                   "so this is its ungrounded configuration.\n")

    if "smartdoc" in outs and args.list_inversions:
        own = [i for i in verbs if inverts(outs["smartdoc"][i], verbs[i])]
        copied = [i for i in own if i in reproduced]
        own = [i for i in own if i not in reproduced]
        doc.append(f"\n## SmartDoc's inversions\n")
        if copied:
            doc.append(f"{len(copied)} of the flagged cases are verbatim "
                       f"copies of the reference, where the reference itself "
                       f"omits the verb — those are the corpus's problem, not "
                       f"the model's, and are left out below.\n")
        for i in own[:args.list_inversions]:
            doc.append(f"- `{function_name(code[i])}` (verb `{verbs[i]}`) → "
                       f"\"{outs['smartdoc'][i].strip()}\" "
                       f"— reference: \"{ref[i].strip()}\"")
        doc.append("")
    return "\n".join(doc)


# ---------------------------------------------------------------------------
# NatSpecGold
# ---------------------------------------------------------------------------
def on_corpus(args) -> str:
    from tools.bleu_table import load

    every = scope.load_pairs(args.corpus)
    kinds = scope.parse_kinds(args.kinds)
    pairs = scope.scoped(every, split=args.split, kinds=kinds)

    verbs: Dict[str, str] = {}
    for pid, pair in pairs.items():
        name = pair.get("name") or function_name(pair.get("code") or "")
        v = leading_verb(name) if name else None
        if v:
            verbs[pid] = v

    def gold(pid):
        p = pairs[pid]
        return ". ".join(x for x in (p.get("notice") or "",
                                     p.get("dev") or "") if x)

    doc = [f"# Verb agreement — {args.corpus.name} {args.split}, "
           f"{'+'.join(kinds)}\n",
           f"{len(pairs)} functions; {len(verbs)} "
           f"({len(verbs)/max(len(pairs),1)*100:.0f}%) are verb-named.\n"]

    by: Dict[str, Dict[str, str]] = {}
    rows_in = []
    for root in (args.comgen_runs or (HERE / "comgen/results/runs"),
                 args.corpus / "runs"):
        if Path(root).exists():
            rows_in += load(Path(root), args.split)
    for r in rows_in:
        if r.get("error") or r.get("seed") != args.comgen_seed:
            continue
        by.setdefault(r.get("config"), {})[r.get("pair_id")] = r.get("final") or ""

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    common = set(verbs)
    for name in names:
        if by.get(name):
            common &= set(by[name])
    common = sorted(common)
    if not common:
        common = sorted(verbs)

    rows = [_row("gold (reference)",
                 measure([(verbs[p], gold(p)) for p in common]))]
    for name in names:
        got = by.get(name)
        if not got:
            print(f"note: no {args.split} rows for {name} at seed "
                  f"{args.comgen_seed}", file=sys.stderr)
            continue
        rows.append(_row(name, measure(
            [(verbs[p], surface.join(surface.hypothesis_view(
                got.get(p, ""), "notice"))) for p in common])))
    doc.append(markdown_table(HEAD, rows) + NOTE)
    return "\n".join(doc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", type=Path, default=None,
                    help="the cloned xing-hu/SmartDoc repository")
    ap.add_argument("--split-dir", default="RQ1",
                    help="which final_results/ subdirectory to score")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="score a NatSpecGold-shaped corpus instead")
    ap.add_argument("--split", default="test")
    ap.add_argument("--kinds", default="function")
    ap.add_argument("--configs", default="G1,G8,C1")
    ap.add_argument("--comgen-runs", type=Path, default=None)
    ap.add_argument("--eval-corpus", type=Path,
                    default=HERE / "data/SmartDocEval")
    ap.add_argument("--comgen-config", default="G1")
    ap.add_argument("--comgen-seed", type=int, default=0)
    ap.add_argument("--list-inversions", type=int, default=12)
    ap.add_argument("--out", type=Path, default=HERE / "results/functionwise")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    if not args.pkg and not args.corpus:
        raise SystemExit("pass --pkg (SmartDoc's artefact) or --corpus")

    text = on_package(args) if args.pkg else on_corpus(args)
    print(text)
    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        tag = "smartdoc" if args.pkg else f"{args.corpus.name}_{args.split}"
        path = next_path(args.out / f"verbs_{tag}.md")
        path.write_text(text, encoding="utf-8")
        print(f"\nwritten to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
