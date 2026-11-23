"""Decision models: Jev and every other model behind the same API.

A decision model is not a chat model: it takes a `state` (the text) and
named `questions`, and answers each with a typed choice plus calibrated
probabilities, in one pass, without generating text.

    POST {url}
      {model, state, questions: {name: {type, instructions, criteria}}}
    -> {answers: {name: {type, choice, confidence, probabilities}}}

The same contract is served by OpenRouter's Decisions API for every decision
model it lists (`GET https://openrouter.ai/api/v1/models?output_modalities=decisions`:
TypeSafe Jev, Liquid d1, Perplexity Decider, Microsoft Decision, ...) and by
self-hosted servers of TypeSafe's System One API (`POST /v1/systemone`), such
as Kev (github.com/jaredpalmer/kev). So one detector covers them all: pick
the model with `model`, and the server with `url`.

Where models differ:
- Some answer only yes/no questions (`noul`). For those, set
  `question="noul"`; there is then no "unclear".
- Context windows range from about 8k tokens to 1M. The screener splits long
  text into chunks of `max_chars`; set it to fit the model (about 4 chars a
  token for plain text, less for dense text).
- A model may refuse to answer (`type: "refusal"`). That is an error verdict,
  never a pass.

When state overflows the model's limit, the server answers with an error
about length, which becomes a TOO_LONG verdict, and the screener halves the
chunk. Seen live from Jev (2026-09-29) for ~120k chars of base64:

    400 {"error":{"message":"HTTP 400: {\"detail\":{\"error_type\":\"max_tokens_exceeded\"}}","code":400}}

Neither TypeSafe nor OpenRouter documents this body, so any 400/413/422 that
talks about length counts, in case the wording changes. A wrong match only
costs a few split retries before on_error.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Literal

from ..standards import CRITERIA, QUESTION
from ..verdict import TOO_LONG, Verdict
from .base import Detector

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"

_TOO_LONG_STATUSES = {400, 413, 422}
_TOO_LONG_WORDS = re.compile(r"max_tokens|token limit|too many tokens|context length|too long", re.IGNORECASE)


def _is_openrouter(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


class DecisionDetector(Detector):
    """Asks a decision model one follows / breaks / unclear question per
    text: does it break the standard it is given?

    model      the decision model, e.g. "typesafe/jev-1.13" (the default),
               "liquid/d1.2", or whatever a self-hosted server expects
    url        OpenRouter's Decisions API (the default), or a self-hosted
               System One endpoint such as "http://127.0.0.1:8009/v1/systemone"
    api_key    sent as a Bearer token. For OpenRouter it defaults to the
               OPENROUTER_API_KEY environment variable; that key is never
               sent to any other host, so a self-hosted server gets a key
               only if you pass one
    question   "choice" (follows / breaks / unclear; the default) or "noul"
               (yes/no, for models that answer only that; no "unclear")
    breaks_at  your own cut-off on the probability of "breaks": at or above
               it the verdict is "breaks" with confidence 1.0, so the
               target's threshold no longer filters it; below it, the most
               likely other answer. The raw probabilities stay in `scores`.
               Use it for models whose probabilities rank well but whose
               top answer is too cautious for your rules; pick the value
               from shadow-mode logs. None (the default) keeps the model's
               own answer and confidence
    max_chars  the most text per call; the screener splits longer text
    name       shown in logs and verdicts; defaults to the model

    Every failure (no key, timeout, HTTP error, refusal, odd response) is an
    "error" verdict, never an exception.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        url: str = OPENROUTER_DECISIONS_URL,
        api_key: str | None = None,
        question: Literal["choice", "noul"] = "choice",
        breaks_at: float | None = None,
        timeout: float = 5.0,
        title: str = "taster-ai",
        referer: str | None = None,
        max_chars: int = 120_000,
        name: str | None = None,
    ):
        if question not in ("choice", "noul"):
            raise ValueError(f'question must be "choice" or "noul", not {question!r}')
        if breaks_at is not None and not 0.0 < float(breaks_at) <= 1.0:
            raise ValueError("breaks_at must be above 0 and at most 1")
        self.model = model
        self.url = url
        self.api_key = api_key
        self.question = question
        self.breaks_at = None if breaks_at is None else float(breaks_at)
        self.timeout = timeout
        self.title = title
        self.referer = referer
        self.max_chars = max_chars
        self.name = name or model

    def payload(self, text: str, standard: str) -> dict:
        instructions = QUESTION.format(standard=standard)
        if self.question == "choice":
            q = {"type": "choice", "instructions": instructions, "criteria": CRITERIA}
        else:
            q = {"type": "noul", "instructions": instructions,
                 "criteria": {"true": CRITERIA["breaks"], "false": CRITERIA["follows"]}}
        return {"model": self.model, "state": text.strip(), "questions": {"verdict": q}}

    def _key(self) -> str | None:
        if self.api_key:
            return self.api_key.strip()
        if _is_openrouter(self.url):  # the OpenRouter key goes to OpenRouter only
            return os.environ.get("OPENROUTER_API_KEY", "").strip() or None
        return None

    def detect(self, text: str, standard: str) -> Verdict:
        key = self._key()
        if key is None and _is_openrouter(self.url):
            return Verdict.failed(self.name, "no-openrouter-key")
        headers = {"Content-Type": "application/json", "X-Title": self.title}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        req = urllib.request.Request(
            self.url, data=json.dumps(self.payload(text, standard)).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
            return self._verdict(body["answers"]["verdict"])
        except urllib.error.HTTPError as e:
            if e.code in _TOO_LONG_STATUSES and _TOO_LONG_WORDS.search(_error_body(e)):
                return Verdict.failed(self.name, TOO_LONG)
            return Verdict.failed(self.name, f"http-{e.code}")
        except urllib.error.URLError as e:
            return Verdict.failed(self.name, f"url-{getattr(e, 'reason', e)}")
        except Exception as e:  # timeout, JSON, missing keys
            return Verdict.failed(self.name, type(e).__name__)

    def _verdict(self, answer: dict) -> Verdict:
        kind = answer.get("type", self.question)
        if kind == "refusal":
            return Verdict.failed(self.name, "refusal")
        if self.question == "noul":
            if kind != "noul":
                return Verdict.failed(self.name, f"unexpected-answer-type:{kind}")
            p = float(answer["noul"])
            scores = {"follows": 1.0 - p, "breaks": p}
            if p >= (0.5 if self.breaks_at is None else self.breaks_at):
                conf = 1.0 if self.breaks_at is not None else 2 * p - 1
                return Verdict(label="breaks", confidence=conf, scores=scores, detector=self.name)
            return Verdict(label="follows", confidence=max(0.0, 1 - 2 * p), scores=scores, detector=self.name)
        choice = str(answer.get("choice", "")).strip()
        if choice not in CRITERIA:
            return Verdict.failed(self.name, f"unexpected-verdict:{choice}")
        scores = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
        if self.breaks_at is not None:
            if scores.get("breaks", 0.0) >= self.breaks_at:
                return Verdict(label="breaks", confidence=1.0, scores=scores, detector=self.name)
            if choice == "breaks":  # the model's top answer, but under your cut-off
                others = {k: v for k, v in scores.items() if k in ("follows", "unclear")}
                choice = max(others, key=others.get) if others else "unclear"
                return Verdict(label=choice, confidence=0.0, scores=scores, detector=self.name)  # type: ignore[arg-type]
        return Verdict(
            label=choice,  # type: ignore[arg-type]
            confidence=float(answer.get("confidence", 0.0)),
            scores=scores,
            detector=self.name,
        )


def _error_body(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode(errors="replace")
    except Exception:  # no body
        return ""
