"""Content-addressed cache for model calls.

Keyed by the whole rendered request, so a changed prompt, a changed model or a
changed fact table all miss. That is the point: a cache that survives an edit
to the prompt would silently serve results from the previous experiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class CallCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(prompt_id: str, request: Dict[str, Any]) -> str:
        blob = json.dumps(request, sort_keys=True, ensure_ascii=False)
        return f"{prompt_id}-{hashlib.sha1(blob.encode()).hexdigest()}"

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, prompt_id: str, request: Dict[str, Any]) -> Optional[dict]:
        p = self._path(self.key(prompt_id, request))
        if not p.exists():
            self.misses += 1
            return None
        try:
            v = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.misses += 1
            return None
        self.hits += 1
        return v

    def put(self, prompt_id: str, request: Dict[str, Any],
            value: Dict[str, Any]) -> None:
        p = self._path(self.key(prompt_id, request))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)                 # atomic: a killed run leaves no half file
