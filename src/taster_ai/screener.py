"""The engine: rule lookup, chunking, detection, combining, deciding, logging.

Framework adapters (LangChain, ...) call `Screener.screen` with a tool's name,
arguments and result text, then `render` the returned decision.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .actions import Action, Messages, decide, render
from .detectors import Detector, HeuristicDetector, as_detector
from .policy import Policy, Rule
from .sinks import Sink, log_sink
from .verdict import ToolContext, Verdict

logger = logging.getLogger("taster_ai")


@dataclass(frozen=True)
class Decision:
    """The outcome of screening one tool result."""

    tool: str
    tool_call_id: str | None
    rule: Rule
    verdict: Verdict
    #: What was actually done (always "passed" in shadow mode).
    action: Action
    #: What enforce mode would do; equals `action` in enforce mode.
    would_action: Action
    chars: int
    chunks: int
    input_hash: str
    latency_s: float

    def render(self, content: Any, messages: Messages | None = None) -> Any:
        """The content the model should see, or None to leave it unchanged."""
        return render(self.action, self.tool, self.verdict, content, messages)

    def record(self) -> dict[str, Any]:
        """A log-friendly dict. Holds a hash of the text, never the text."""
        v = self.verdict
        return {
            "tool": self.tool,
            "tool_call_id": self.tool_call_id,
            "rule": self.rule.tool,
            "mode": self.rule.mode,
            "label": v.label,
            "confidence": round(v.confidence, 4),
            "scores": dict(v.scores),
            "detector": v.detector,
            "error": v.error,
            "evidence": list(v.evidence),
            "action": self.action,
            "would_action": self.would_action,
            "chars": self.chars,
            "chunks": self.chunks,
            "latency_s": round(self.latency_s, 3),
            "input_hash": self.input_hash,
        }


def combine(verdicts: list[Verdict]) -> Verdict:
    """One verdict for a result from its chunks' verdicts, suspicious first:
    the most confident injection wins; otherwise any failure makes the whole
    result unscreened; otherwise any unclear chunk makes it unclear;
    otherwise it's clean."""
    ok = [v for v in verdicts if v.ok]
    flagged = [v for v in ok if v.label == "injection"]
    if flagged:
        return max(flagged, key=lambda v: v.confidence)
    if len(ok) < len(verdicts):
        return next(v for v in verdicts if not v.ok)
    unclear = [v for v in ok if v.label == "unclear"]
    if unclear:
        return max(unclear, key=lambda v: v.confidence)
    return min(ok, key=lambda v: v.confidence)


def chunk(text: str, size: int, overlap: int) -> list[str]:
    """Split into `size`-char pieces that overlap by `overlap` chars, so an
    injection straddling a boundary is still seen whole by one chunk."""
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    pieces = []
    for start in range(0, len(text), step):
        pieces.append(text[start : start + size])
        if start + size >= len(text):
            break
    return pieces


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Screener:
    """Screens tool results according to a policy.

    policy      the rules (what to screen, what to do)
    detector    the default judge: a Detector, or a LangChain chat model
                (wrapped in LLMDetector). Defaults to the free
                HeuristicDetector so it works with no setup.
    detectors   extra judges by name, for rules that say `detector: <name>`
    sinks       callables that receive each decision's record; defaults to
                [log_sink]. Pass [] for none.
    messages    the wording of withheld/wrapped results
    overlap     chars shared by neighbouring chunks of a long result
    max_chunks  results needing more chunks than this count as unscreenable
                (the rule's on_error applies) rather than being half-checked
    max_splits  how many times a chunk the detector refuses as too long is
                halved before giving up (the rule's on_error applies)
    cache_size  verdicts remembered by text hash, so a page seen twice is
                judged once; 0 disables
    """

    def __init__(
        self,
        policy: Policy,
        detector: Any = None,
        *,
        detectors: Mapping[str, Any] | None = None,
        sinks: Iterable[Sink] | None = None,
        messages: Messages | None = None,
        overlap: int = 200,
        max_chunks: int = 8,
        max_splits: int = 4,
        cache_size: int = 1024,
    ):
        self.policy = policy
        self.detector = as_detector(detector) if detector is not None else HeuristicDetector()
        self.detectors = {name: as_detector(d) for name, d in (detectors or {}).items()}
        missing = sorted({r.detector for r in policy.rules if r.mode != "pass" and r.detector} - set(self.detectors))
        if missing:
            raise ValueError(f"rules name detectors that weren't given: {missing}; pass detectors={{name: ...}}")
        self.sinks = [log_sink] if sinks is None else list(sinks)
        self.messages = messages or Messages()
        self.overlap = overlap
        self.max_chunks = max_chunks
        self.max_splits = max_splits
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[int, str], Verdict] = OrderedDict()
        self._lock = threading.Lock()

    def detector_for(self, rule: Rule) -> Detector:
        return self.detectors[rule.detector] if rule.detector else self.detector

    def screen(
        self,
        tool: str,
        args: Mapping[str, Any] | None,
        text: str,
        tool_call_id: str | None = None,
    ) -> Decision | None:
        """Screen one tool result. Returns None when it isn't screened (a
        "pass" rule, or empty text). Never raises."""
        args = args or {}
        rule = self.policy.rule_for(tool, args)
        if rule.mode == "pass" or not text.strip():
            return None
        started = time.perf_counter()
        try:
            verdict, chunks = self._verdict(rule, text, ToolContext(tool, args, tool_call_id))
        except Exception as e:  # a bug here must not break the turn; on_error decides
            logger.exception("taster: screening failed for %s", tool)
            verdict, chunks = Verdict.failed("taster", f"internal-{type(e).__name__}"), 0
        would = decide(rule, verdict)
        decision = Decision(
            tool=tool,
            tool_call_id=tool_call_id,
            rule=rule,
            verdict=verdict,
            action=would if rule.mode == "enforce" else "passed",
            would_action=would,
            chars=len(text),
            chunks=chunks,
            input_hash=_hash(text),
            latency_s=time.perf_counter() - started,
        )
        self._emit(decision)
        return decision

    def _verdict(self, rule: Rule, text: str, context: ToolContext) -> tuple[Verdict, int]:
        detector = self.detector_for(rule)
        pieces = chunk(text, detector.max_chars, min(self.overlap, detector.max_chars // 2))
        if len(pieces) > self.max_chunks:
            return Verdict.failed(detector.name, f"too-large:{len(text)}-chars"), len(pieces)
        if len(pieces) == 1:
            return self._judge(detector, pieces[0], context, self.max_splits)
        with ThreadPoolExecutor(max_workers=len(pieces)) as pool:
            results = list(pool.map(lambda p: self._judge(detector, p, context, self.max_splits), pieces))
        return combine([v for v, _ in results]), sum(n for _, n in results)

    def _judge(self, detector: Detector, text: str, context: ToolContext, splits_left: int) -> tuple[Verdict, int]:
        """A verdict for one piece and how many pieces it took: a piece the
        detector refuses as too long is halved (with overlap) and each half
        judged, up to `splits_left` times deep."""
        verdict = self._detect(detector, text, context)
        if not verdict.too_long or splits_left <= 0 or len(text) < 2:
            return verdict, 1
        overlap = min(self.overlap, len(text) // 4)
        halves = chunk(text, (len(text) + overlap + 1) // 2, overlap)
        results = [self._judge(detector, half, context, splits_left - 1) for half in halves]
        return combine([v for v, _ in results]), sum(n for _, n in results)

    def _detect(self, detector: Detector, text: str, context: ToolContext) -> Verdict:
        key = (id(detector), _hash(text))
        if self.cache_size:
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
        verdict = detector.safe_detect(text, context)
        if verdict.ok and self.cache_size:
            with self._lock:
                self._cache[key] = verdict
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return verdict

    def _emit(self, decision: Decision) -> None:
        record = decision.record()
        for sink in self.sinks:
            try:
                sink(record)
            except Exception:  # never let logging break a turn
                logger.exception("taster: sink %r failed", sink)
