"""One model call, with the accounting the evaluation needs.

Three clients behind one interface. `OllamaClient` and `HFClient` talk to a
real model; `MockClient` returns scripted, schema-shaped responses so the
runner, the gate and the evaluation can be built and tested where no model is
reachable. The mock is not a stand-in for measurement — it exists so that a
bug in the orchestration is found before a GPU is involved.

Every call records whether the **first** attempt parsed. That number is the
one that matters: a prompt whose first-attempt parse rate sits below about
95% is a prompt problem, and no amount of retrying makes the resulting
distribution of outputs trustworthy.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .errors import CorpusError
from .prompts_v3 import Prompt, render


class LLMError(CorpusError):
    """The model could not be reached, or never produced parsable output."""


@dataclass
class Call:
    """One completed call, whatever the backend."""
    prompt_id: str
    model: str
    data: Dict[str, Any]
    raw: str
    attempts: int
    first_attempt_parsed: bool
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    #: A bookkeeping field was absent and filled with its empty value; the
    #: substantive answer came from the model. See `repair`.
    repaired: bool = False

    def summary(self) -> dict:
        return {"prompt_id": self.prompt_id, "model": self.model,
                "attempts": self.attempts,
                "first_attempt_parsed": self.first_attempt_parsed,
                "seconds": round(self.seconds, 3), "repaired": self.repaired,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cached": self.cached}


class Backend(Protocol):
    name: str

    def complete(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Takes a rendered request; returns {"text", "prompt_tokens",
        "completion_tokens"}."""


# --------------------------------------------------------------------------
# JSON recovery
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Optional[dict]:
    """Recover an object from a response that is nearly JSON.

    Small models wrap JSON in prose or a code fence perhaps a tenth of the
    time. Recovering it is not the same as accepting bad output: the call
    still records that the first attempt did not parse cleanly, so the
    harness can see how often this happens per prompt.
    """
    if not text:
        return None
    for candidate in (text, *(m.group(1) for m in _FENCE.finditer(text))):
        candidate = candidate.strip()
        try:
            v = json.loads(candidate)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(text[start:i + 1])
                        if isinstance(v, dict):
                            return v
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def missing_fields(data: dict, schema: dict) -> List[str]:
    return [k for k in schema.get("required", []) if k not in data]


#: The one field whose absence cannot be repaired. Deliberately just this:
#: the observed failure is a written comment discarded for want of a flag, and
#: `natspec` is the only field that is unambiguously *the answer*. Widening
#: this to the critic's `defects` or the intent reader's `purpose` would turn
#: "the model failed" into "the model found nothing", which is a worse error
#: than losing the record — an empty critique reads as a clean comment.
PRIMARY = ("natspec",)

#: What an absent bookkeeping field becomes. Only fields with an unambiguous
#: empty value, so a repaired reply is indistinguishable downstream from one
#: that said "nothing to add".
DEFAULTS = {
    "claims": [], "changed": False, "removed": [], "added": [],
    "used_examples": False, "conflicts_with_intent": [],
    "addressed": [], "declined": [],
}


def repair(data: Optional[dict], schema: dict) -> Optional[dict]:
    """Fill absent bookkeeping fields when the substantive answer is present.

    Measured on the C-series: 8% of records were lost to
    `no schema-valid JSON after 3 attempts` on the refiner, and the replies
    were not malformed — they carried a perfectly good `natspec` and omitted
    `changed` or `claims`. Discarding a written comment because the model did
    not also report whether it had changed anything is a bookkeeping failure
    being charged to the experiment.

    A field is only ever filled when it has an unambiguous empty value and the
    prompt's substantive field is present. A reply with no `natspec` is still
    a failure, and `repaired` is recorded on the call so the rate is visible
    rather than hidden.
    """
    if not isinstance(data, dict):
        return None
    required = schema.get("required", [])
    primary = [k for k in required if k in PRIMARY]
    if not primary or any(k not in data for k in primary):
        return None
    absent = [k for k in required if k not in data]
    if not absent or any(k not in DEFAULTS for k in absent):
        return None
    return {**{k: DEFAULTS[k] for k in absent}, **data}


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

class OllamaBackend:
    """Ollama's /api/chat. The rendered request already carries `format` as
    the JSON schema, which Ollama enforces server-side."""

    def __init__(self, host: str = "http://localhost:11434",
                 timeout: int = 300) -> None:
        self.name = host
        self.host = host.rstrip("/")
        self.timeout = timeout

    def complete(self, request: Dict[str, Any]) -> Dict[str, Any]:
        import urllib.error
        import urllib.request
        body = json.dumps({**request, "stream": False}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read())
        except urllib.error.URLError as e:
            raise LLMError(f"ollama at {self.host}: {e}") from e
        return {"text": (payload.get("message") or {}).get("content", ""),
                "prompt_tokens": payload.get("prompt_eval_count", 0),
                "completion_tokens": payload.get("eval_count", 0)}


class HFBackend:
    """A local transformers model. Slower to set up than Ollama and without
    server-side schema enforcement, so `extract_json` does more work here."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device or "auto")

    def complete(self, request: Dict[str, Any]) -> Dict[str, Any]:
        msgs = request["messages"]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        opts = request.get("options", {})
        out = self.model.generate(
            **ids, max_new_tokens=opts.get("num_predict", 1024),
            temperature=max(opts.get("temperature", 0.0), 1e-5),
            do_sample=opts.get("temperature", 0.0) > 0)
        gen = out[0][ids["input_ids"].shape[1]:]
        return {"text": self.tok.decode(gen, skip_special_tokens=True),
                "prompt_tokens": int(ids["input_ids"].shape[1]),
                "completion_tokens": int(gen.shape[0])}


class MockBackend:
    """Scripted responses, for building and testing the orchestration.

    `handler(prompt_id, request) -> dict | str` decides what comes back. A
    dict is serialised as the model's JSON; a string is returned verbatim, so
    a test can reproduce malformed output deliberately.
    """

    name = "mock"

    def __init__(self, handler: Callable[[str, dict], Any]) -> None:
        self.handler = handler
        self.calls: List[dict] = []

    def complete(self, request: Dict[str, Any]) -> Dict[str, Any]:
        pid = request.get("_prompt_id", "?")
        self.calls.append(request)
        v = self.handler(pid, request)
        text = v if isinstance(v, str) else json.dumps(v)
        return {"text": text, "prompt_tokens": 0, "completion_tokens": 0}


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------

REPAIR_SUFFIX = (
    "\n\nYour previous reply was not valid JSON matching the required schema. "
    "Reply with the JSON object only — no prose, no code fence.")


class Client:
    def __init__(self, backend: Backend, models: Optional[Dict[str, str]] = None,
                 cache=None, max_attempts: int = 3,
                 seed: Optional[int] = None) -> None:
        self.backend = backend
        self.models = models or {}
        self.cache = cache
        self.max_attempts = max_attempts
        # The seed goes into the request, so it reaches both the model's
        # sampler and the cache key. Two seeds sharing a cached call would
        # make the seed axis measure nothing at all.
        self.seed = seed
        self.log: List[Call] = []

    def _resolve(self, slot: str) -> str:
        return self.models.get(slot, slot)

    def call(self, prompt: Prompt, **inputs) -> Call:
        request = render(prompt, **inputs)
        request["model"] = self._resolve(prompt.model)
        request["_prompt_id"] = prompt.id
        if self.seed is not None:
            request.setdefault("options", {})["seed"] = self.seed

        if self.cache is not None:
            hit = self.cache.get(prompt.id, request)
            if hit is not None:
                c = Call(prompt_id=prompt.id, model=request["model"],
                         data=hit["data"], raw=hit["raw"],
                         attempts=hit.get("attempts", 1),
                         first_attempt_parsed=hit.get("first", True),
                         seconds=0.0, cached=True)
                self.log.append(c)
                return c

        started = time.time()
        first_ok = False
        repaired = False
        last_text = ""
        data: Optional[dict] = None
        last_parsed: Optional[dict] = None
        attempt = 0
        for attempt in range(1, self.max_attempts + 1):
            req = json.loads(json.dumps(request))
            if attempt > 1:
                req["messages"][-1]["content"] += REPAIR_SUFFIX
            out = self.backend.complete(req)
            last_text = out["text"]
            parsed = extract_json(last_text)
            if parsed is not None:
                last_parsed = parsed
            if parsed is not None and not missing_fields(parsed, prompt.schema):
                data = parsed
                first_ok = attempt == 1
                break
        if data is None:
            # Last resort, and only for an absent bookkeeping field on a reply
            # that carries the substantive answer. Recorded, never silent.
            data = repair(last_parsed, prompt.schema)
            repaired = data is not None
        if data is None:
            raise LLMError(
                f"{prompt.id}: no schema-valid JSON after {self.max_attempts} "
                f"attempts; last reply began {last_text[:120]!r}")

        c = Call(prompt_id=prompt.id, model=request["model"], data=data,
                 raw=last_text, attempts=attempt, first_attempt_parsed=first_ok,
                 repaired=repaired, seconds=time.time() - started,
                 prompt_tokens=out.get("prompt_tokens", 0),
                 completion_tokens=out.get("completion_tokens", 0))
        if self.cache is not None:
            self.cache.put(prompt.id, request,
                           {"data": data, "raw": last_text,
                            "attempts": attempt, "first": first_ok})
        self.log.append(c)
        return c

    def stats(self) -> Dict[str, Dict[str, float]]:
        """Per prompt: first-attempt parse rate, mean latency, call count.
        This is what milestone M1 is gated on."""
        out: Dict[str, Dict[str, float]] = {}
        for c in self.log:
            s = out.setdefault(c.prompt_id, {"calls": 0, "first_ok": 0,
                                             "seconds": 0.0, "cached": 0,
                                             "repaired": 0})
            s["calls"] += 1
            s["first_ok"] += int(c.first_attempt_parsed)
            s["seconds"] += c.seconds
            s["cached"] += int(c.cached)
            s["repaired"] += int(c.repaired)
        for s in out.values():
            n = max(s["calls"], 1)
            s["first_attempt_parse_rate"] = s["first_ok"] / n
            s["mean_seconds"] = s["seconds"] / n
        return out
