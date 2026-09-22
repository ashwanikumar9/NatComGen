"""Re-grounding the SmartDoc (ASE'21) test set.

SmartDoc published 1,000 test functions as whitespace-tokenised bodies with
no contract around them. They do not compile, so Σ(f) cannot be computed and
NatComGen cannot run on them as shipped.

This package recovers the missing context instead of inventing it: every one
of those functions was scraped from a verified Etherscan contract, so most of
them still exist, verbatim, inside the bulk verified-source corpora. Match the
tokenised body against such a corpus, and each hit hands back the *real* file
— hence a real compilation unit, a real Σ(f), and a function whose published
reference comment already exists.

The price is coverage: only the matched subset can be scored. That number is
reported everywhere, always as "n of 1000", and the published systems are
re-scored on the identical subset so the comparison is between like and like.
"""

__all__ = ["tokens", "index", "reground", "build_corpus", "bleu", "score"]
