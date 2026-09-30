"""The detector interface, a detector chain, and a wrapper for plain functions."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Callable

from ..verdict import LABELS, TOO_LONG, ToolContext, Verdict


class Detector(ABC):
    """Judges whether a piece of text is a prompt injection.

    Subclasses implement `detect`. Callers use `safe_detect`, which never
    raises: any exception becomes an "error" verdict, so a broken detector
    can't break an agent's turn.
    """

    #: Short name, shown in logs and verdicts.
    name: str = "detector"
    #: The most text this detector takes in one call; the screener splits
    #: longer tool results into chunks of this size. A detector whose real
    #: limit is in tokens can set this high and return an error verdict of
    #: TOO_LONG when a chunk overflows; the screener then halves it.
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
    """Try detectors in order; the first one that gives a verdict wins.

    Typical use: a strong remote detector first, a cheaper one behind it,
    and the free heuristic last, so an outage still catches the obvious
    attacks instead of passing everything through unscreened.

        FallbackDetector(JevDetector(), my_chat_model, HeuristicDetector())

    Chat models are wrapped in LLMDetector automatically.
    """

    def __init__(self, *detectors: Any):
        from . import as_detector  # late import: detectors/__init__ imports this module

        if len(detectors) < 2:
            raise ValueError("FallbackDetector needs at least two detectors")
        self.detectors = [as_detector(d) for d in detectors]
        self.name = "|".join(d.name for d in self.detectors)
        self.max_chars = min(d.max_chars for d in self.detectors)

    def detect(self, text: str, context: ToolContext | None = None) -> Verdict:
        errors = []
        for detector in self.detectors:
            verdict = detector.safe_detect(text, context)
            if verdict.too_long:  # let the screener split it for the whole chain, not the backup alone
                return Verdict.failed(self.name, TOO_LONG)
            if verdict.ok:
                if errors:  # say in the logs that a fallback answered
                    verdict = replace(verdict, detector=f"{verdict.detector} (fallback: {'; '.join(errors)})")
                return verdict
            errors.append(f"{detector.name} {verdict.error}")
        return Verdict.failed(self.name, "; ".join(errors))


class CallableDetector(Detector):
    """Wrap any function `text -> (label, confidence)` as a detector, for
    in-house models or SDKs Taster has no adapter for.

        CallableDetector(lambda text: my_classifier(text), name="in-house")
    """

    def __init__(self, fn: Callable[[str], tuple[str, float]], *, name: str = "callable", max_chars: int = 8000):
        self.fn = fn
        self.name = name
        self.max_chars = max_chars

    def detect(self, text: str, context: ToolContext | None = None) -> Verdict:
        label, confidence = self.fn(text)
        return Verdict(label, float(confidence), {label: float(confidence)}, self.name)  # type: ignore[arg-type]
