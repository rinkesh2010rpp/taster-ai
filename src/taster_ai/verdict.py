"""The answer every judge gives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Label = Literal["follows", "breaks", "unclear", "error"]
LABELS: tuple[str, ...] = ("follows", "breaks", "unclear", "error")
#: The 0.1 labels, still accepted from judges written for 0.1.
ALIASES = {"clean": "follows", "injection": "breaks"}

#: The `error` of a judge that refused text as too long for it. The
#: screener then splits the text in half and asks again.
TOO_LONG = "too-long"
#: The start of the `error` for text needing more pieces than the screener
#: allows (`max_chunks`).
TOO_LARGE = "too-large"


@dataclass(frozen=True)
class Verdict:
    """One judge's answer: does this text follow the standard?

    `label` is "follows", "breaks" or "unclear", or "error" when the judge
    could not answer (timeout, HTTP error, missing key); `error` then says
    why. `evidence` is optional detail: the first item is the reason the
    agent is told, e.g. "writes an API key to a file".
    """

    label: Label
    confidence: float = 0.0
    scores: Mapping[str, float] = field(default_factory=dict)
    detector: str = ""
    error: str | None = None
    latency_s: float = 0.0
    evidence: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "label", ALIASES.get(self.label, self.label))
        object.__setattr__(self, "scores", {ALIASES.get(k, k): v for k, v in self.scores.items()})

    @property
    def ok(self) -> bool:
        return self.label != "error"

    @property
    def too_long(self) -> bool:
        return self.error == TOO_LONG

    @property
    def oversized(self) -> bool:
        """Failed because of the input's size, which whoever wrote it controls."""
        return self.too_long or (self.error or "").startswith(TOO_LARGE)

    @classmethod
    def failed(cls, detector: str, error: str, latency_s: float = 0.0) -> Verdict:
        return cls("error", 0.0, {}, detector, error, latency_s)


@dataclass(frozen=True)
class ToolContext:
    """Which tool call a decision is about (for logs; judges never see it)."""

    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = None
