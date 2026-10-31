"""The engine: runs every check at its stage, and logs what each decided.

Framework adapters (LangChain, ...) call `Screener.screen_call` with a tool's
name and arguments before it runs (and skip the tool if the outcome is
blocked), then `Screener.screen_result` with its result text, and `render`
the returned outcome.

For each check whose scope has the stage, the first matching target picks
the mode; the text is split into pieces the judge can take, each piece is
judged against the check's standard (in parallel, cached by judge, standard
and text), and the most suspicious verdict wins. Checks
never affect each other: a call or result is blocked if any check in
enforce mode blocks it, and the agent is told every check that did.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .actions import Action, Messages, decide, decide_call
from .checks import Check, Injection, call_pieces
from .detectors import Detector, HeuristicDetector
from .policy import Policy, Scope, Target
from .sinks import Sink, log_sink
from .verdict import TOO_LARGE, ToolContext, Verdict

logger = logging.getLogger("taster_ai")


@dataclass(frozen=True)
class Decision:
    """What one check decided about one tool call or result."""

    check: Check
    stage: str
    tool: str
    tool_call_id: str | None
    target: Target
    verdict: Verdict
    #: What was actually done (always "passed" in shadow mode).
    action: Action
    #: What enforce mode would do; equals `action` in enforce mode.
    would_action: Action
    chars: int
    chunks: int
    input_hash: str
    latency_s: float

    @property
    def rule(self) -> Target:  # the 0.1 name
        return self.target

    @property
    def blocked(self) -> bool:
        """The call must not run, or the result must not be shown."""
        return self.action in ("blocked", "withheld")

    def render(self, content: Any, messages: Messages | None = None) -> Any:
        """The result the model should see, or None to leave it unchanged."""
        return Outcome(self.stage, self.tool, (self,)).render(content, messages)

    def message(self, messages: Messages | None = None) -> str:
        """What the agent is told in place of a blocked call's result."""
        return Outcome(self.stage, self.tool, (self,)).message(messages)

    def record(self) -> dict[str, Any]:
        """A log-friendly dict. Holds a hash of the text, never the text."""
        v = self.verdict
        return {
            "check": self.check.name,
            "stage": self.stage,
            "tool": self.tool,
            "tool_call_id": self.tool_call_id,
            "target": self.target.tool,
            "mode": self.target.mode,
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


@dataclass(frozen=True)
class Outcome:
    """What every check that ran decided about one tool call or result."""

    stage: str
    tool: str
    decisions: tuple[Decision, ...]

    @property
    def blocked(self) -> bool:
        return any(d.blocked for d in self.decisions)

    @property
    def failed(self) -> list[Decision]:
        """The checks that blocked it."""
        return [d for d in self.decisions if d.blocked]

    def failures(self, messages: Messages | None = None) -> str:
        m = messages or Messages()
        lines = []
        for n, d in enumerate(self.failed, 1):
            v = d.verdict
            fields = {"n": n, "check": d.check.name, "confidence": v.confidence, "error": v.error}
            if v.ok:
                lines.append(m.failure.format(why=d.check.reason(v), **fields))
            elif v.oversized and self.stage == "tool_call":
                lines.append(m.too_large.format(why="", **fields))
            else:
                lines.append(m.unchecked.format(why="", **fields))
        return "\n".join(lines)

    def message(self, messages: Messages | None = None) -> str:
        """What the agent is told in place of a blocked call's result."""
        m = messages or Messages()
        return m.blocked.format(tool=self.tool, failures=self.failures(m))

    def render(self, content: Any, messages: Messages | None = None) -> Any:
        """The result the model should see, or None to leave it unchanged:
        withheld if any check withheld it, else wrapped as untrusted if any
        check wrapped or couldn't screen it."""
        m = messages or Messages()
        if self.blocked:
            return m.withheld.format(tool=self.tool, failures=self.failures(m))
        marked = [d for d in self.decisions if d.action in ("wrapped", "labelled")]
        if not marked:
            return None
        why = "; ".join(
            f"{d.check.name}: {d.verdict.label}, confidence {d.verdict.confidence:.2f}"
            if d.verdict.ok
            else f"{d.check.name}: not screened: {d.verdict.error}"
            for d in marked
        )
        header = m.wrap_header.format(tool=self.tool, why=why)
        footer = m.wrap_footer.format(tool=self.tool, why=why)
        if isinstance(content, str):
            return header + content + footer
        return [{"type": "text", "text": header}, *content, {"type": "text", "text": footer}]


def combine(verdicts: list[Verdict]) -> Verdict:
    """One verdict from its pieces' verdicts, suspicious first: the most
    confident "breaks" wins; otherwise any failure makes the whole thing
    unchecked; otherwise any unclear piece makes it unclear; otherwise it's
    clean."""
    ok = [v for v in verdicts if v.ok]
    flagged = [v for v in ok if v.label == "breaks"]
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
    """Runs checks on tool calls and tool results.

    checks      the checks to run (Injection, Constitution, your own Check)
    policy, detector, detectors
                the 0.1 way, still supported: one injection check on tool
                results with these targets, judged by `detector` (default:
                the free HeuristicDetector), with `detectors` by name. Pass
                these or `checks`, not both.
    sinks       callables that receive each decision's record; defaults to
                [log_sink]. Pass [] for none.
    messages    the wording of what the agent is told
    overlap     chars shared by neighbouring pieces of long text
    max_chunks  text needing more pieces than this counts as uncheckable
                (the target's on_error applies) rather than half-checked
    max_splits  how many times text the judge refuses as too long is cut
                smaller before giving up (the target's on_error applies)
    cache_size  verdicts remembered by judge, standard and text hash, so a
                page seen twice is judged once; 0 disables
    """

    def __init__(
        self,
        policy: Policy | None = None,
        detector: Any = None,
        *,
        checks: Iterable[Check] | None = None,
        detectors: Mapping[str, Any] | None = None,
        sinks: Iterable[Sink] | None = None,
        messages: Messages | None = None,
        overlap: int = 200,
        max_chunks: int = 8,
        max_splits: int = 4,
        cache_size: int = 1024,
    ):
        if checks is None:
            if policy is None:
                raise ValueError("pass checks=[...] (or a policy)")
            judge = detector if detector is not None else HeuristicDetector()
            checks = [Injection(Scope(tool_result=policy), judge, detectors=detectors)]
        elif policy is not None or detector is not None or detectors:
            raise ValueError("pass checks=[...], or policy/detector/detectors, not both")
        self.checks = list(checks)
        if not self.checks:
            raise ValueError("pass at least one check")
        names = [c.name for c in self.checks]
        if len(set(names)) < len(names):
            raise ValueError(f"check names must be unique, got {names}")
        self.sinks = [log_sink] if sinks is None else list(sinks)
        self.messages = messages or Messages()
        self.overlap = overlap
        self.max_chunks = max_chunks
        self.max_splits = max_splits
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[int, str, str], Verdict] = OrderedDict()
        self._lock = threading.Lock()

    def screen_call(
        self, tool: str, args: Mapping[str, Any] | None, tool_call_id: str | None = None
    ) -> Outcome | None:
        """Check one tool call before it runs. Returns None when no check
        runs on it. Never raises."""
        return self._run("tool_call", tool, args or {}, "", tool_call_id)

    def screen_result(
        self, tool: str, args: Mapping[str, Any] | None, text: str, tool_call_id: str | None = None
    ) -> Outcome | None:
        """Check one tool result before the model sees it. Returns None when
        no check runs on it (or the text is empty). Never raises."""
        if not text.strip():
            return None
        return self._run("tool_result", tool, args or {}, text, tool_call_id)

    def screen(
        self, tool: str, args: Mapping[str, Any] | None, text: str, tool_call_id: str | None = None
    ) -> Decision | None:
        """The 0.1 call: screen a result and return the first check's
        decision. Use screen_result to get every check's."""
        outcome = self.screen_result(tool, args, text, tool_call_id)
        return outcome.decisions[0] if outcome else None

    def _run(self, stage: str, tool: str, args: Mapping[str, Any], text: str, tool_call_id: str | None):
        jobs = []
        for check in self.checks:
            targets = check.scope.for_stage(stage)
            target = targets.rule_for(tool, args) if targets is not None else None
            if target is not None and target.mode != "pass":
                jobs.append((check, target))
        if not jobs:
            return None
        context = ToolContext(tool, args, tool_call_id)
        run = lambda job: self._decide(stage, job[0], job[1], context, text)  # noqa: E731
        if len(jobs) == 1:
            decisions = [run(jobs[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                decisions = list(pool.map(run, jobs))
        for d in decisions:
            self._emit(d)
        return Outcome(stage, tool, tuple(decisions))

    def _decide(self, stage: str, check: Check, target: Target, context: ToolContext, text: str) -> Decision:
        started = time.perf_counter()
        if stage == "tool_call":
            text = f"{context.tool} {json.dumps(dict(context.args), ensure_ascii=False, default=str)}"
        try:
            judge = check.judge_for(target)
            if stage == "tool_call":
                verdict, chunks = self._call_verdict(judge, context, check.standard)
            else:
                verdict, chunks = self._verdict(judge, text, check.standard)
        except Exception as e:  # a bug here must not break the turn; on_error decides
            logger.exception("taster: check %s failed for %s", check.name, context.tool)
            verdict, chunks = Verdict.failed("taster", f"internal-{type(e).__name__}"), 0
        would = decide_call(target, verdict) if stage == "tool_call" else decide(target, verdict)
        return Decision(
            check=check,
            stage=stage,
            tool=context.tool,
            tool_call_id=context.tool_call_id,
            target=target,
            verdict=verdict,
            action=would if target.mode == "enforce" else "passed",
            would_action=would,
            chars=len(text),
            chunks=chunks,
            input_hash=_hash(text),
            latency_s=time.perf_counter() - started,
        )

    def _verdict(self, detector: Detector, text: str, standard: str) -> tuple[Verdict, int]:
        pieces = chunk(text, detector.max_chars, min(self.overlap, detector.max_chars // 2))
        if len(pieces) > self.max_chunks:
            return Verdict.failed(detector.name, f"{TOO_LARGE}:{len(text)}-chars"), len(pieces)
        results = self._each(pieces, lambda p: self._judge(detector, p, standard, self.max_splits))
        return combine([v for v, _ in results]), sum(n for _, n in results)

    def _call_verdict(self, detector: Detector, context: ToolContext, standard: str) -> tuple[Verdict, int]:
        """Judge a call in pieces that each carry the tool's name and short
        arguments; if the judge refuses a piece as too long, cut the pieces
        smaller and judge again, up to max_splits times."""
        size = detector.max_chars
        for splits_left in range(self.max_splits, -1, -1):
            pieces = call_pieces(context.tool, context.args, size, min(self.overlap, size // 4))
            if len(pieces) > self.max_chunks:
                return Verdict.failed(detector.name, f"{TOO_LARGE}:{len(pieces)}-pieces"), len(pieces)
            verdicts = self._each(pieces, lambda p: self._detect(detector, p, standard))
            if not any(v.too_long for v in verdicts) or splits_left == 0 or size < 400:
                return combine(verdicts), len(pieces)
            size //= 2
        raise AssertionError("unreachable")

    def _each(self, pieces: list[str], fn) -> list:
        if len(pieces) == 1:
            return [fn(pieces[0])]
        with ThreadPoolExecutor(max_workers=len(pieces)) as pool:
            return list(pool.map(fn, pieces))

    def _judge(self, detector: Detector, text: str, standard: str, splits_left: int) -> tuple[Verdict, int]:
        """A verdict for one piece and how many pieces it took: a piece the
        detector refuses as too long is halved (with overlap) and each half
        judged, up to `splits_left` times deep."""
        verdict = self._detect(detector, text, standard)
        if not verdict.too_long or splits_left <= 0 or len(text) < 2:
            return verdict, 1
        overlap = min(self.overlap, len(text) // 4)
        halves = chunk(text, (len(text) + overlap + 1) // 2, overlap)
        results = [self._judge(detector, half, standard, splits_left - 1) for half in halves]
        return combine([v for v, _ in results]), sum(n for _, n in results)

    def _detect(self, detector: Detector, text: str, standard: str) -> Verdict:
        key = (id(detector), _hash(standard), _hash(text))
        if self.cache_size:
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
        verdict = detector.safe_detect(text, standard)
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
