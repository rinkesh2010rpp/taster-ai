"""The detector interface, and a detector that falls back to another."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace

from ..verdict import LABELS, ToolContext, Verdict


class Detector(ABC):
    """Judges whether a piece of text is a prompt injection.

    Subclasses implement `detect`. Callers use `safe_detect`, which never
    raises: any exception becomes an "error" verdict, so a broken detector
    can't break an agent's turn.
    """

    #: Short name, shown in logs and verdicts.
    name: str = "detector"
    #: The most text this detector takes in one call; the screener splits
    #: longer tool results into chunks of this size.
    max_chars: int = 8000

    @abstractmethod
    def detect(self, text: str, context: ToolContext | None = None) -> Verdict:
        """Judge `text`. May raise; `safe_detect` turns that into an error."""

    def safe_detect(self, text: str, context: ToolContext | None = None) -> Verdict:
        started = time.perf_counter()
        try:
            verdict = self.detect(text, context)
        except Exception as e:  # a detector bug must never break the turn
            return Verdict.failed(self.name, f"{type(e).__name__}: {e}", time.perf_counter() - started)
        if not isinstance(verdict, Verdict) or verdict.label not in LABELS:
            return Verdict.failed(self.name, f"bad-verdict:{verdict!r}", time.perf_counter() - started)
        if not verdict.latency_s:
            verdict = replace(verdict, latency_s=time.perf_counter() - started)
        return verdict


class FallbackDetector(Detector):
    """Use `primary`; when it can't give a verdict, ask `fallback` instead.

    Typical use: a strong remote detector (Jev) with the free local heuristic
    behind it, so an outage still catches the obvious attacks instead of
    passing everything through unscreened.
    """

    def __init__(self, primary: Detector, fallback: Detector):
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}|{fallback.name}"
        self.max_chars = min(primary.max_chars, fallback.max_chars)

    def detect(self, text: str, context: ToolContext | None = None) -> Verdict:
        verdict = self.primary.safe_detect(text, context)
        if verdict.ok:
            return verdict
        backup = self.fallback.safe_detect(text, context)
        if not backup.ok:
            return Verdict.failed(self.name, f"{verdict.error}; fallback: {backup.error}")
        return replace(backup, detector=f"{backup.detector} (fallback: {self.primary.name} {verdict.error})")
