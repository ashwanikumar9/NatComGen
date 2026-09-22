#!/usr/bin/env python3
"""A fake Ollama, so the pipeline's model stages can be wired-tested offline.

This is NOT a model and it measures nothing. It answers `/api/chat` with the
smallest reply that satisfies each prompt's schema, so that `run_pipeline.sh`
can be run start to finish — harness, experiments, report, emission — on a
machine with no GPU and no Ollama. What that proves is the wiring: that every
stage's arguments line up, that the run files, the statistics and the report
agree on their key names, and that the emitted Solidity still compiles. The
numbers it produces are meaningless by construction.

    python tools/stub_ollama.py --port 11500 &
    ./run_pipeline.sh --ollama http://localhost:11500 --limit 5 --seeds 0

Use it before a long run on the GPU box too: it exercises every stage in
seconds, so a wiring mistake surfaces before four hours of generation do.
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

#: Pulled out of the rendered prompt so the reply at least mentions the
#: function it is supposed to be about — enough for the scoring code to have
#: something non-empty to work on.
_SIG = re.compile(r"function\s+(\w+)|(?:^|\n)\s*signature:\s*(\w+)", re.I)
_ROW = re.compile(r"\b([FNPCDRE]\d+)\b")


def _name(text: str) -> str:
    m = _SIG.search(text or "")
    return (m.group(1) or m.group(2)) if m else "f"


def _rows(text: str, n: int = 2):
    seen, out = set(), []
    for m in _ROW.finditer(text or ""):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
        if len(out) >= n:
            break
    return out or ["F1"]


def reply(prompt_id: str, user: str) -> dict:
    """The smallest schema-valid object for each prompt in the pipeline."""
    name, rows = _name(user), _rows(user)
    claim = f"Returns the value {name} computes."
    if prompt_id == "L7":
        return {"purpose": f"{name} performs its stated operation.",
                "purpose_ids": rows, "caller": None, "caller_ids": [],
                "preconditions": [], "effects": [], "returns_meaning": [],
                "unknowns": []}
    if prompt_id in ("L1b", "L3"):
        return {"natspec": f"/// @notice {claim}",
                "claims": [{"text": claim, "ids": rows}],
                "used_examples": False, "changed": False, "removed": [],
                "added": []}
    if prompt_id == "L2":
        return {"defects": [], "missing_high_value_facts": []}
    if prompt_id == "L8":
        return {"verdicts": [{"claim": claim, "proposition": claim,
                              "rows": rows, "verdict": "SUPPORTED",
                              "citation_correct": True, "note": ""}],
                "gate": "PASS"}
    return {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                   # noqa: N802
        if self.path.startswith("/api/tags"):
            self._send({"models": [{"name": "stub:latest"}]})
        else:
            self.send_error(404)

    def do_POST(self):                                  # noqa: N802
        if not self.path.startswith("/api/chat"):
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        pid = req.get("_prompt_id", "?")
        user = (req.get("messages") or [{}])[-1].get("content", "")
        self._send({"message": {"role": "assistant",
                                "content": json.dumps(reply(pid, user))},
                    "prompt_eval_count": len(user) // 4,
                    "eval_count": 32, "done": True})

    def log_message(self, *a):                          # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=11500)
    args = ap.parse_args()
    srv = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"stub ollama on http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
