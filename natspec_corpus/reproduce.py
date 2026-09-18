"""One command from sources to results, with hashes checked at each boundary.

This is the difference between a thesis that says what it did and one that can
be checked. The pieces are already deterministic — stage 1 rebuilds
byte-identically, Σ(f) is stable across `PYTHONHASHSEED` values, and every
model call is cached by the content of its request — but determinism nobody
records is not reproducibility. This module records it.

What it does NOT do is hide a failure. If a stage's output hash differs from
the manifest, the run stops and says which stage and which file, because a
pipeline that silently continues from changed inputs produces numbers nobody
can defend.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .errors import CorpusError


class ReproduceError(CorpusError):
    """A stage's output did not match the manifest."""


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_tree(root: Path, pattern: str = "**/*") -> str:
    """One hash over a directory: paths and contents, in sorted order.

    Sorted, so the filesystem's iteration order cannot change the answer, and
    paths included, so a renamed file is a different tree.
    """
    h = hashlib.sha1()
    for p in sorted(x for x in root.glob(pattern) if x.is_file()):
        h.update(str(p.relative_to(root)).encode())
        h.update(sha1_file(p).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# what a run is made of
# --------------------------------------------------------------------------

def environment() -> dict:
    """Everything about the machine that could change a number."""
    from .compile import installed_versions
    try:
        import slither
        slither_v = getattr(slither, "__version__", "unknown")
    except Exception:                                   # noqa: BLE001
        slither_v = "not installed"
    try:
        import faiss
        faiss_v = getattr(faiss, "__version__", "unknown")
    except Exception:                                   # noqa: BLE001
        faiss_v = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "solc_versions": installed_versions(),
        "slither": slither_v,
        "faiss": faiss_v,
        "git_commit": _git_commit(),
    }


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:                                   # noqa: BLE001
        return None


def artifact_hashes(corpus_root: Path) -> Dict[str, str]:
    """The hash of every stage output that a later stage depends on."""
    out: Dict[str, str] = {}
    for name in ("pairs.jsonl", "pairs_partial.jsonl", "manifest.json",
                 "splits.json", "index_allowlist.json"):
        p = corpus_root / name
        if p.exists():
            out[name] = sha1_file(p)
    contracts = corpus_root / "contracts"
    if contracts.is_dir():
        out["contracts/"] = sha1_tree(contracts, "**/*.sol")
    sigma = corpus_root / "sigma" / "sigma.jsonl"
    if sigma.exists():
        out["sigma/sigma.jsonl"] = sha1_file(sigma)
    return out


def prompt_hashes() -> Dict[str, str]:
    """A prompt change must invalidate every result derived from it."""
    from . import prompts_v3 as P
    out = {}
    for p in P.PIPELINE:
        blob = json.dumps({"system": p.system, "user": p.user,
                           "schema": p.schema, "temp": p.temp,
                           "model": p.model}, sort_keys=True)
        out[p.id] = hashlib.sha1(blob.encode()).hexdigest()[:16]
    out["retrieval_config"] = hashlib.sha1(
        json.dumps(P.RETRIEVAL_CONFIG, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return out


def manifest(corpus_root: Path, *, models: Optional[Dict[str, str]] = None,
             seeds: Sequence[int] = (), configs: Sequence[str] = (),
             split: str = "val", encoder: Optional[str] = None) -> dict:
    return {
        "artifacts": artifact_hashes(corpus_root),
        "prompts": prompt_hashes(),
        "environment": environment(),
        "models": dict(models or {}),
        "encoder": encoder,
        "seeds": list(seeds),
        "configs": list(configs),
        "split": split,
    }


def check(corpus_root: Path, expected: dict, *,
          sections: Sequence[str] = ("artifacts", "prompts")) -> List[str]:
    """Differences between now and a recorded manifest, as readable lines."""
    now = {"artifacts": artifact_hashes(corpus_root),
           "prompts": prompt_hashes()}
    problems: List[str] = []
    for section in sections:
        want, got = expected.get(section, {}), now.get(section, {})
        for key in sorted(set(want) | set(got)):
            if key not in got:
                problems.append(f"{section}.{key}: missing now")
            elif key not in want:
                problems.append(f"{section}.{key}: not in the manifest")
            elif want[key] != got[key]:
                problems.append(
                    f"{section}.{key}: {want[key][:12]} → {got[key][:12]}")
    return problems


def verify(corpus_root: Path, manifest_path: Path,
           sections: Sequence[str] = ("artifacts", "prompts")) -> None:
    """Raise unless everything matches. Used before a run reuses a cache."""
    expected = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    problems = check(corpus_root, expected, sections=sections)
    if problems:
        raise ReproduceError(
            "the corpus no longer matches its manifest:\n  "
            + "\n  ".join(problems))


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------

@dataclass
class Stage:
    name: str
    run: Callable[[], object]
    description: str = ""


@dataclass
class Trace:
    stages: List[dict] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"stages": self.stages, "manifest": self.manifest}


def pipeline(src_root: Path, corpus_root: Path, *,
             with_sigma: bool = True) -> List[Stage]:
    """Stages 1 and 1b, the two that need no model.

    Stage 2 onwards is not here on purpose: those need a client, and a
    reproduction script that quietly instantiates a model connection is a
    script that fails on someone else's machine for reasons it cannot explain.
    """
    from . import build as corpus_build
    from . import sigma_build

    stages = [Stage("corpus", lambda: corpus_build.build(src_root, corpus_root),
                    "sources → pairs, splits, allowlist")]
    if with_sigma:
        stages.append(Stage("sigma", lambda: sigma_build.build(corpus_root),
                            "compile → CFG, data dependency, fact tables"))
    return stages


def run(src_root: Path, corpus_root: Path, *, with_sigma: bool = True,
        expect: Optional[dict] = None, progress: bool = True) -> Trace:
    """Rebuild from sources and record what came out.

    With `expect`, the rebuild is checked against a previous manifest and
    raises on any difference. Without it, the manifest produced becomes the
    record for next time.
    """
    trace = Trace()
    for stage in pipeline(src_root, corpus_root, with_sigma=with_sigma):
        if progress:
            print(f"[{stage.name}] {stage.description}", flush=True)
        report = stage.run()
        trace.stages.append({"name": stage.name,
                             "report": report if isinstance(report, dict) else {}})
    trace.manifest = manifest(corpus_root)
    if expect is not None:
        problems = check(corpus_root, expect)
        trace.manifest["reproduction"] = {
            "matches": not problems, "differences": problems}
        if problems:
            raise ReproduceError(
                "rebuild does not reproduce the recorded manifest:\n  "
                + "\n  ".join(problems))
    return trace


def main(argv=None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    src = Path(argv[0]) if argv else Path("/mnt/user-data/uploads/contracts")
    out = Path(argv[1]) if len(argv) > 1 else Path("out/NatSpecGold")
    expect_path = Path(argv[2]) if len(argv) > 2 else None
    expect = json.loads(expect_path.read_text()) if expect_path else None
    trace = run(src, out, expect=expect)
    (out / "reproduce_manifest.json").write_text(
        json.dumps(trace.manifest, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"stages": [s["name"] for s in trace.stages],
                      "artifacts": trace.manifest["artifacts"]}, indent=1))


if __name__ == "__main__":
    main()
