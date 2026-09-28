"""Taster: antivirus for your AI agent.

Every tool result is tasted before your agent swallows it. The core has no
dependencies; framework adapters live in `taster_ai.adapters`.
"""

from typing import Any

from .actions import Messages
from .detectors import CallableDetector, Detector, FallbackDetector, HeuristicDetector, JevDetector
from .policy import Policy, Rule
from .screener import Decision, Screener
from .sinks import BackgroundSink, JsonlSink, log_sink, print_sink
from .verdict import ToolContext, Verdict

__version__ = "0.1.0.dev1"

__all__ = [
    "BackgroundSink",
    "CallableDetector",
    "Decision",
    "Detector",
    "FallbackDetector",
    "HeuristicDetector",
    "JevDetector",
    "JsonlSink",
    "LLMDetector",
    "Messages",
    "Policy",
    "Rule",
    "Screener",
    "ToolContext",
    "Verdict",
    "log_sink",
    "print_sink",
]


def __getattr__(name: str) -> Any:
    # LLMDetector needs langchain-core; import it only when asked for.
    if name == "LLMDetector":
        from .detectors.llm import LLMDetector

        return LLMDetector
    raise AttributeError(name)
