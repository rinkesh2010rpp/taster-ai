"""The answer every detector gives, and what it is asked about."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Label = Literal["clean", "injection", "unclear", "error"]
LABELS: tuple[str, ...] = ("clean", "injection", "unclear", "error")

#: The `error` of a detector that refused text as too long for it. The
#: screener then splits the text in half and asks again.
TOO_LONG = "too-long"


@dataclass(frozen=True)
class Verdict:
    """One detector's judgment of one piece of text.

    `label` is "error" when the detector could not judge (timeout, HTTP error,
    missing key); `error` then says why. `evidence` is optional detail for the
    logs, e.g. which heuristic patterns matched.
    """

    label: Label
    confidence: float = 0.0
    scores: Mapping[str, float] = field(default_factory=dict)
    detector: str = ""
    error: str | None = None
    latency_s: float = 0.0
    evidence: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.label != "error"

    @property
    def too_long(self) -> bool:
        return self.error == TOO_LONG

    @classmethod
    def failed(cls, detector: str, error: str, latency_s: float = 0.0) -> Verdict:
        return cls("error", 0.0, {}, detector, error, latency_s)


@dataclass(frozen=True)
class ToolContext:
    """Which tool call produced the text being screened."""

    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = None
