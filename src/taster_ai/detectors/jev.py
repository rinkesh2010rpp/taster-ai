"""Jev (TypeSafe) via OpenRouter's decisions endpoint.

Jev is a decision model, not a chat model: it takes a `state` (the text) and
named `questions`, and answers each with a typed choice plus calibrated
probabilities.

    POST /api/alpha/decisions
      {model, state, questions: {name: {type, instructions, criteria}}}
    -> {answers: {name: {type, choice, confidence, probabilities}}}

Jev's limit is 32k tokens of state, and there is no public way to count them
first. So chunks are sized for plain text (about 4 chars a token), and when
denser text overflows, the refusal becomes a TOO_LONG verdict and the
screener halves the chunk. Seen live (2026-09-29) for ~120k chars of base64:

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
import urllib.request
from ..standards import CRITERIA, QUESTION
from ..verdict import TOO_LONG, Verdict
from .base import Detector

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"

_TOO_LONG_STATUSES = {400, 413, 422}
_TOO_LONG_WORDS = re.compile(r"max_tokens|token limit|too many tokens|context length|too long", re.IGNORECASE)

class JevDetector(Detector):
    """Asks Jev one follows / breaks / unclear question per text: does it
    break the standard it is given?

    The API key comes from `api_key` or the OPENROUTER_API_KEY environment
    variable. Every failure (no key, timeout, HTTP error, odd response) is an
    "error" verdict, never an exception.
    """

    name = "jev"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = JEV_MODEL,
        url: str = JEV_URL,
        timeout: float = 5.0,
        title: str = "taster-ai",
        referer: str | None = None,
        max_chars: int = 120_000,
    ):
        self.api_key = api_key
        self.model = model
        self.url = url
        self.timeout = timeout
        self.title = title
        self.referer = referer
        self.max_chars = max_chars

    def payload(self, text: str, standard: str) -> dict:
        return {
            "model": self.model,
            "state": text.strip(),
            "questions": {
                "verdict": {
                    "type": "choice",
                    "instructions": QUESTION.format(standard=standard),
                    "criteria": CRITERIA,
                }
            },
        }

    def detect(self, text: str, standard: str) -> Verdict:
        api_key = (self.api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not api_key:
            return Verdict.failed(self.name, "no-openrouter-key")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": self.title,
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        req = urllib.request.Request(
            self.url, data=json.dumps(self.payload(text, standard)).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
            answer = body["answers"]["verdict"]
            choice = str(answer.get("choice", "")).strip()
            if choice not in CRITERIA:
                return Verdict.failed(self.name, f"unexpected-verdict:{choice}")
            return Verdict(
                label=choice,  # type: ignore[arg-type]
                confidence=float(answer.get("confidence", 0.0)),
                scores={str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()},
                detector=self.name,
            )
        except urllib.error.HTTPError as e:
            if e.code in _TOO_LONG_STATUSES and _TOO_LONG_WORDS.search(_error_body(e)):
                return Verdict.failed(self.name, TOO_LONG)
            return Verdict.failed(self.name, f"http-{e.code}")
        except urllib.error.URLError as e:
            return Verdict.failed(self.name, f"url-{getattr(e, 'reason', e)}")
        except Exception as e:  # timeout, JSON, missing keys
            return Verdict.failed(self.name, type(e).__name__)


def _error_body(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode(errors="replace")
    except Exception:  # no body
        return ""
