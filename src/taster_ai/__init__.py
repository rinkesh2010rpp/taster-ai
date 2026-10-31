"""Taster: checks on what your AI agent does and what it takes in.

Every tool call is checked before it runs, and every tool result before your
agent swallows it, by the checks you choose (Injection, Constitution, your
own Check). The core has no dependencies; framework adapters live in
`taster_ai.adapters`.
"""

from typing import Any

from .actions import Messages
from .checks import Check, Constitution, Injection, checks_from_dict, checks_from_env, checks_from_json, checks_from_yaml
from .detectors import CallableDetector, Detector, FallbackDetector, HeuristicDetector, JevDetector
from .scope import Scope, Target
from .screener import Decision, Outcome, Screener
from .standards import INJECTION_STANDARD
from .sinks import BackgroundSink, JsonlSink, log_sink, print_sink
from .verdict import TOO_LARGE, TOO_LONG, Verdict

__version__ = "0.2.0.dev0"

__all__ = [
    "BackgroundSink",
    "CallableDetector",
    "Check",
    "Constitution",
    "Decision",
    "Detector",
    "FallbackDetector",
    "HeuristicDetector",
    "INJECTION_STANDARD",
    "Injection",
    "JevDetector",
    "JsonlSink",
    "LLMDetector",
    "Messages",
    "Outcome",
    "Scope",
    "Screener",
    "TOO_LARGE",
    "TOO_LONG",
    "Target",
    "Verdict",
    "checks_from_dict",
    "checks_from_env",
    "checks_from_json",
    "checks_from_yaml",
    "log_sink",
    "print_sink",
]


def __getattr__(name: str) -> Any:
    # LLMDetector needs langchain-core; import it only when asked for.
    if name == "LLMDetector":
        from .detectors.llm import LLMDetector

        return LLMDetector
    raise AttributeError(name)
