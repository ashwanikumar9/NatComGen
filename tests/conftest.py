"""Shared fixtures — chiefly, finding or building a corpus to test against.

Two test modules used to look for a corpus at a hard coded relative path and
skip themselves when it was not there. That is the worst way for a test to
fail: the suite stays green while whole modules go unexercised, and it depends
on which directory pytest happened to be started from. This resolves a corpus
properly, and builds a small one where a small one will do.
"""
from __future__ import annotations

import functools
import os
import tempfile
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

#: Where a built corpus might be, most specific first. The environment
#: variable exists so a run can be pointed at a corpus elsewhere.
CORPUS_CANDIDATES = (
    ROOT / "data" / "NatSpecGold",
    ROOT / "out" / "NatSpecGold",
    Path("out/NatSpecGold"),
)


def find_corpus() -> Optional[Path]:
    """A real, fully built corpus, or None. Never builds one."""
    env = os.environ.get("NATCOMGEN_CORPUS")
    for c in ((Path(env),) if env else ()) + CORPUS_CANDIDATES:
        if (c / "pairs.jsonl").exists():
            return c
    return None


def contract(name: str, *, full: int, partial: int) -> str:
    """A contract with a known mix of complete and incomplete NatSpec.

    The counts matter. M1 samples verified pairs and M2 needs both families,
    so a fixture holding two documented functions would fail those tests for
    reasons that have nothing to do with the code under test.
    """
    body = [f"contract {name} {{", "    uint256 public total;", ""]
    for i in range(full):
        body += [f"    /// @notice Adds {i} to the amount and returns it.",
                 "    /// @param amount The amount to add to.",
                 "    /// @return The amount plus the constant.",
                 f"    function full{i}(uint256 amount) external pure"
                 " returns (uint256) {",
                 f"        return amount + {i};", "    }", ""]
    for i in range(partial):
        body += [f"    /// @notice Multiplies the amount by {i + 1}.",
                 f"    function part{i}(uint256 amount) external pure"
                 " returns (uint256) {",
                 f"        return amount * {i + 1};", "    }", ""]
    body += ["    function undocumented() external {", "        total = 0;",
             "    }", "}"]
    return ("// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n\n"
            + "\n".join(body) + "\n")


@functools.lru_cache(maxsize=1)
def mini_corpus() -> Path:
    """A built two-project corpus, with Σ(f) when a compiler is available.

    The folder names are not decoration: the splitter assigns projects, so
    these two names are what put one contract in train and the other in val.
    """
    from natspec_corpus import build as corpus_build
    from natspec_corpus.compile import installed_versions

    tmp = Path(tempfile.mkdtemp(prefix="natcomgen-fixture-"))
    src = tmp / "src"
    train = src / "Trail_of_Bits-UniswapV3Core" / "v3-core-abc" / "contracts"
    val = src / "Trail_of_Bits-Primitive" / "rmm-core-def" / "contracts"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    (train / "Vault.sol").write_text(contract("Vault", full=8, partial=14),
                                     encoding="utf-8")
    (val / "Engine.sol").write_text(contract("Engine", full=8, partial=14),
                                    encoding="utf-8")
    out = tmp / "NatSpecGold"
    corpus_build.build(src, out)
    if installed_versions():
        from natspec_corpus import sigma_build
        sigma_build.build(out)
    return out


def corpus_or_mini() -> Path:
    """The real corpus where there is one, a small built one otherwise."""
    return find_corpus() or mini_corpus()
