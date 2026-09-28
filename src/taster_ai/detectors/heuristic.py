"""A free, instant, offline detector for the obvious attacks.

It looks for a handful of patterns that real injections almost always use:
override phrases, fake system/owner messages, text addressed to "the AI
reading this", requests to hide things from the user or send secrets,
directives hidden in HTML comments, and invisible Unicode tag characters.

It is a baseline and a fallback, not a replacement for a model-based
detector: it misses paraphrased attacks, and its judgment of one pattern on
its own is only "unclear". Text inside double quotes is ignored, so an
article that *quotes* "ignore all previous instructions" isn't flagged.
"""

from __future__ import annotations

import re

from ..verdict import ToolContext, Verdict
from .base import Detector

_I = re.IGNORECASE
PATTERNS: dict[str, re.Pattern[str]] = {
    "override": re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
        r"\b(?:previous|prior|earlier|above|preceding|all|your|any|the)\b[^.\n]{0,25}?"
        r"\b(?:instructions?|prompts?|rules|directions|guidelines|directives)\b",
        _I,
    ),
    "new-task": re.compile(
        r"\b(?:your|the)\s+(?:new\s+)?(?:task|instructions?|goal|objective|role)\s+"
        r"(?:has|have|is|are)\s+(?:now\s+)?(?:changed|updated|replaced|overridden)\b"
        r"|\bnew\s+instructions?\s*:",
        _I,
    ),
    "fake-authority": re.compile(
        r"^\W*(?:system|developer|admin(?:istrator)?|operator|owner)\s+"
        r"(?:message|prompt|note|notice|instructions?|override)\b[^\n:]{0,40}:",
        _I | re.MULTILINE,
    ),
    "addresses-ai": re.compile(
        r"\b(?:ai|llms?|language\s+models?|assistants?|agents?|chatbots?)\b[^.\n]{0,30}?"
        r"\b(?:reading|processing|summari[sz]ing|parsing|viewing|browsing|seeing)\s+(?:this|these)\b",
        _I,
    ),
    "conceal": re.compile(
        r"\b(?:do\s+not|don't|never)\s+(?:tell|mention|inform|reveal|disclose|alert)\b[^.\n]{0,30}?"
        r"\b(?:the\s+)?(?:user|human|operator|anyone)\b",
        _I,
    ),
    "exfiltrate": re.compile(
        r"\b(?:send|e-?mail|post|upload|forward|leak|exfiltrate|reveal|print|output|share)\b[^.\n]{0,40}?"
        r"\b(?:api[\s_-]?keys?|passwords?|secrets?|access\s+tokens?|credentials|system\s+prompt|\.env)\b",
        _I,
    ),
    "pipe-to-shell": re.compile(r"\b(?:curl|wget)\b[^\n|]{0,120}\|\s*(?:sudo\s+)?(?:ba|z)?sh\b", _I),
}
# Unicode "tag" characters are invisible and have no normal use in text;
# bidi overrides can reorder what a human sees versus what a model reads.
_TAG_CHARS = re.compile("[\U000e0000-\U000e007f]")
_BIDI = re.compile("[‪-‮⁦-⁩]")
_HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)
_QUOTED = re.compile(r'"[^"\n]{1,400}"|“[^”\n]{1,400}”')


def _hits(text: str) -> set[str]:
    return {name for name, pattern in PATTERNS.items() if pattern.search(text)}


class HeuristicDetector(Detector):
    """Pattern-based detector: 0 signals -> clean, 1 -> unclear, 2+ -> injection."""

    name = "heuristic"
    max_chars = 200_000  # no model behind it, so no need to chunk

    def detect(self, text: str, context: ToolContext | None = None) -> Verdict:
        unquoted = _QUOTED.sub(" ", text)
        hits = _hits(unquoted)
        if any(_hits(comment) for comment in _HTML_COMMENT.findall(unquoted)):
            hits.add("hidden-in-comment")
        strong = 0
        if _TAG_CHARS.search(text):
            hits.add("invisible-tags")
            strong += 1  # counts double: there is no innocent use
        if _BIDI.search(text):
            hits.add("bidi-control")

        signals = len(hits) + strong
        if signals == 0:
            label, confidence = "clean", 0.6
        elif signals == 1:
            label, confidence = "unclear", 0.6
        elif signals == 2:
            label, confidence = "injection", 0.9
        else:
            label, confidence = "injection", 0.97
        rest = (1 - confidence) / 2
        scores = {k: (confidence if k == label else rest) for k in ("clean", "unclear", "injection")}
        return Verdict(label, confidence, scores, self.name, evidence=tuple(sorted(hits)))  # type: ignore[arg-type]
