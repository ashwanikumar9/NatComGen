"""Multi-view retrieval over the train split.

One index per view — code, AST, CFG — searched separately, then fused. The
protocol follows SAGE: take the union of each index's top-K, z-normalise each
modality's scores *over that union*, and sum them with fixed equal weights.

Two decisions are deliberate and should stay that way:

**The weights are not tuned.** A third each, fixed. SAGE reports (its Table
16) that tuning them gains almost nothing, and tuning three weights on a
validation split of 148 items would fit noise and quietly inflate every number
downstream.

**The retrieval unit is a pair, not a file.** A 500-line contract embedded as
one vector retrieves nothing useful, and the exemplar block wants a function
with its comment.

The allowlist is enforced at build time, not at query time. A file that must
never be indexed cannot be filtered out later if it is already in the index —
the vector is there, and one bug in the filter leaks a test contract into a
prompt. `build_index` raises instead.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np

from .errors import CorpusError
from .views import VIEW_NAMES, views_for

WEIGHTS = {"code": 1 / 3, "ast": 1 / 3, "cfg": 1 / 3}
TOP_K = 5
THRESHOLD_TAU = 0.5
ABSENT_MODALITY_SCORE = 0.0


class RetrievalError(CorpusError):
    """The index could not be built, or a query violated the allowlist."""


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------

class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalised, one row per text."""


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]")


class HashingEncoder:
    """Deterministic, offline, no model download.

    This exists so the whole retrieval pipeline — index construction, fusion,
    allowlist enforcement, Recall@k measurement — can be built and tested
    without network access, and so the tests do not depend on a model's
    weights. It is a real encoder (hashed token and character n-grams, sublinear
    term weighting) but a weak one. Recall figures measured with it are a floor
    and a smoke test, never the numbers to report.
    """

    name = "hashing"

    def __init__(self, dim: int = 1024, ngram: int = 4) -> None:
        self.dim = dim
        self.ngram = ngram

    def _features(self, text: str) -> Dict[int, float]:
        counts: Dict[int, float] = {}
        for tok in _TOKEN.findall(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4)
                               .digest(), "little") % self.dim
            counts[h] = counts.get(h, 0.0) + 1.0
        squeezed = re.sub(r"\s+", " ", text)
        for i in range(len(squeezed) - self.ngram + 1):
            g = squeezed[i:i + self.ngram]
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4)
                               .digest(), "little") % self.dim
            counts[h] = counts.get(h, 0.0) + 0.5
        return counts

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for h, c in self._features(t or "").items():
                out[r, h] = 1.0 + np.log(c)          # sublinear tf
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEncoder:
    """A real code encoder. Used on the machine that has one.

    Recommended, in order:
      jinaai/jina-embeddings-v2-base-code   161M, 8k context, code-trained
      microsoft/unixcoder-base              125M, pretrained on code AND AST,
                                            which suits the φ_AST view
      Salesforce/codet5p-110m-embedding     110M, code-trained

    Ollama's `nomic-embed-text` also works and is the least setup, but it is a
    general-purpose encoder — on the AST and CFG views, which are sequences of
    node-type tokens rather than prose, a code-trained model should do better.
    Whichever is used, the same one encodes the index and the queries.
    """

    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code",
                 device: Optional[str] = None, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.batch_size = batch_size
        self._m = SentenceTransformer(model_name, device=device,
                                      trust_remote_code=True)
        self.dim = self._m.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        v = self._m.encode(list(texts), batch_size=self.batch_size,
                           convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False)
        return v.astype(np.float32)


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

@dataclass
class Unit:
    """One indexed training example."""
    pair_id: str
    file: str
    project: str
    group_id: str
    signature: str
    code: str
    natspec: str
    views: Dict[str, str] = field(default_factory=dict)


@dataclass
class Hit:
    unit: Unit
    score: float
    scores: Dict[str, float]

    def as_exemplar(self) -> dict:
        return {"code": self.unit.code, "natspec": self.unit.natspec,
                "scores": self.scores, "pair_id": self.unit.pair_id}


class MultiViewIndex:
    def __init__(self, units: List[Unit], encoder: Encoder,
                 weights: Optional[Dict[str, float]] = None) -> None:
        import faiss
        self.units = units
        self.encoder = encoder
        self.weights = dict(weights or WEIGHTS)
        self._faiss = faiss
        self.indices: Dict[str, "faiss.Index"] = {}
        for view in VIEW_NAMES:
            texts = [u.views.get(view, "") for u in units]
            vecs = encoder.encode(texts)
            idx = faiss.IndexFlatIP(encoder.dim)   # exact; 519 units is small
            idx.add(vecs)
            self.indices[view] = idx
        self.present = {v: np.array([bool(u.views.get(v, "").strip())
                                     for u in units]) for v in VIEW_NAMES}

    def search(self, query_views: Dict[str, str], *, top_k: int = TOP_K,
               tau: float = THRESHOLD_TAU,
               pool_k: Optional[int] = None) -> List[Hit]:
        pool_k = pool_k or max(top_k * 4, 20)
        per_view: Dict[str, Dict[int, float]] = {}
        union: set = set()
        for view in VIEW_NAMES:
            q = query_views.get(view, "")
            if not q.strip():
                per_view[view] = {}
                continue
            vec = self.encoder.encode([q])
            k = min(pool_k, len(self.units))
            sims, ids = self.indices[view].search(vec, k)
            per_view[view] = {int(i): float(s)
                              for i, s in zip(ids[0], sims[0]) if i >= 0}
            union |= set(per_view[view])

        if not union:
            return []

        members = sorted(union)
        fused = np.zeros(len(members), dtype=np.float64)
        detail: Dict[int, Dict[str, float]] = {m: {} for m in members}
        for view in VIEW_NAMES:
            raw = np.array([per_view[view].get(m, np.nan) for m in members])
            seen = ~np.isnan(raw)
            z = np.full(len(members), ABSENT_MODALITY_SCORE)
            if seen.sum() >= 2:
                mu, sd = raw[seen].mean(), raw[seen].std()
                # z-scored over the UNION, not over each index's own top-K:
                # a document one view ranked highly and another never returned
                # must be comparable with one both returned.
                z[seen] = (raw[seen] - mu) / (sd if sd > 1e-9 else 1.0)
            elif seen.sum() == 1:
                z[seen] = 0.0
            fused += self.weights[view] * z
            for i, m in enumerate(members):
                detail[m][view] = float(z[i]) if seen[i] else 0.0

        order = np.argsort(-fused)
        hits = [Hit(unit=self.units[members[i]], score=float(fused[i]),
                    scores=detail[members[i]]) for i in order[:top_k]]
        return [h for h in hits if h.score >= tau] if tau is not None else hits


# --------------------------------------------------------------------------
# construction from the corpus
# --------------------------------------------------------------------------

def load_units(corpus_root: Path, split: str = "train") -> List[Unit]:
    """Training units, with the allowlist enforced here and not later."""
    allow = json.loads((corpus_root / "index_allowlist.json").read_text())
    permitted = set(allow.get(split, []))
    never = set(allow.get("never_index", []))

    tables: Dict[str, dict] = {}
    sigma = corpus_root / "sigma" / "sigma.jsonl"
    if sigma.exists():
        for line in sigma.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                if t.get("pair_id"):
                    tables[t["pair_id"]] = t

    units: List[Unit] = []
    for line in (corpus_root / "pairs.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        if p["split"] != split:
            continue
        if p["file"] in never:
            raise RetrievalError(
                f"{p['file']} is on never_index but produced a {split} pair")
        if p["file"] not in permitted:
            raise RetrievalError(
                f"{p['file']} is not in the {split} allowlist")
        t = tables.get(p["id"])
        units.append(Unit(
            pair_id=p["id"], file=p["file"], project=p["project"],
            group_id=p["group_id"], signature=p["signature"],
            code=p["code"], natspec=p["doc_raw"],
            views=views_for(p, t)))
    return units


def build_index(corpus_root: Path, encoder: Optional[Encoder] = None,
                split: str = "train") -> MultiViewIndex:
    units = load_units(corpus_root, split)
    if not units:
        raise RetrievalError(f"no units in split {split!r}")
    return MultiViewIndex(units, encoder or HashingEncoder())


def coverage(units: Iterable[Unit]) -> Dict[str, float]:
    """Share of units with a non-empty view. A low AST or CFG figure means
    most units have no fact table, which caps what fusion can do."""
    units = list(units)
    return {v: sum(1 for u in units if u.views.get(v, "").strip()) / len(units)
            for v in VIEW_NAMES} if units else {}


# --------------------------------------------------------------------------
# measuring it
# --------------------------------------------------------------------------

def _fname(sig: str) -> str:
    return sig.split("(", 1)[0]


def evaluate(index: MultiViewIndex, queries: List[Unit], *,
             top_k: int = TOP_K, views: Optional[Sequence[str]] = None
             ) -> Dict[str, float]:
    """Recall@1 and Recall@k over held-out queries.

    Relevance label: a hit counts when it shares the query's **signature**, or
    more loosely its function name. `group_id` cannot be used here — it is
    project-scoped, so a validation unit's group never appears in training by
    construction. Both labels are proxies: two functions named `deposit` need
    not do the same thing. Read these numbers as a sanity check on the
    retriever, not as a measure of exemplar usefulness; whether retrieval
    helps at all is settled by the end-to-end ablation, not here.
    """
    saved = dict(index.weights)
    if views is not None:
        index.weights = {v: (1.0 / len(views) if v in views else 0.0)
                         for v in VIEW_NAMES}
    hit1 = hitk = name1 = namek = 0
    empty = 0
    try:
        for q in queries:
            hits = index.search(q.views, top_k=top_k, tau=None)
            if not hits:
                empty += 1
                continue
            sigs = [h.unit.signature for h in hits]
            if sigs[0] == q.signature:
                hit1 += 1
            if q.signature in sigs:
                hitk += 1
            names = [_fname(s) for s in sigs]
            if names[0] == _fname(q.signature):
                name1 += 1
            if _fname(q.signature) in names:
                namek += 1
    finally:
        index.weights = saved
    n = max(len(queries), 1)
    return {"queries": len(queries), "empty": empty,
            "recall@1_signature": hit1 / n, f"recall@{top_k}_signature": hitk / n,
            "recall@1_name": name1 / n, f"recall@{top_k}_name": namek / n}


def evaluate_loo(index: MultiViewIndex, *, top_k: int = 5,
                 views: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Leave-one-out retrieval quality *within* the indexed split.

    Cross-split Recall@k is not measurable on this corpus. The split is by
    project, so only 2.0% of validation signatures occur in training at all —
    a ceiling low enough that the metric measures the overlap of two protocols'
    vocabularies rather than the retriever.

    Here the relevance label is `group_id`: the same signature inside the same
    project, which is how an interface declaration and its implementation are
    linked. Query with each unit that has such a twin, exclude itself, and ask
    whether the twin comes back. The retriever should find it; if it cannot,
    fusion or the views are broken, and that is worth knowing before any
    end-to-end run.

    Whether retrieval *helps the generated comments* is a different question,
    and only the end-to-end ablation answers it.
    """
    from collections import Counter
    saved = dict(index.weights)
    if views is not None:
        index.weights = {v: (1.0 / len(views) if v in views else 0.0)
                         for v in VIEW_NAMES}
    counts = Counter(u.group_id for u in index.units)
    probes = [u for u in index.units if counts[u.group_id] > 1]
    hit1 = hitk = 0
    try:
        for u in probes:
            hits = index.search(u.views, top_k=top_k + 1, tau=None)
            ranked = [h.unit for h in hits if h.unit.pair_id != u.pair_id][:top_k]
            same = [r.group_id == u.group_id for r in ranked]
            if same and same[0]:
                hit1 += 1
            if any(same):
                hitk += 1
    finally:
        index.weights = saved
    n = max(len(probes), 1)
    return {"probes": len(probes), "recall@1": hit1 / n,
            f"recall@{top_k}": hitk / n}


def label_ceiling(index_units: List[Unit], queries: List[Unit]) -> Dict[str, float]:
    """How often a query's signature or name exists in the index at all.

    Always compute this before reporting a cross-split recall figure. On this
    corpus it is 2.0% by signature, which makes the recall number beneath it
    uninterpretable rather than bad.
    """
    sigs = {u.signature for u in index_units}
    names = {_fname(u.signature) for u in index_units}
    n = max(len(queries), 1)
    return {
        "queries": len(queries),
        "signature_ceiling": sum(1 for q in queries if q.signature in sigs) / n,
        "name_ceiling": sum(1 for q in queries
                            if _fname(q.signature) in names) / n,
    }
